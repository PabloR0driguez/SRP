class PruningInterface:
    def __init__(self, model, pruning_dataloader):
        self.nn = model
        self.dl = pruning_dataloader
        self.device = None
        self.original_param_count = None
        self.base_top1 = None
        self.final_masks = {}
        self.copy_metrics = {}
        self.logical_sparsity = None
        self.top1_gate = None
        self.mlp_importance = None
        self.att_importance = None
        self.mlp_importance_list = None

    def fit(self, device, args):

        #Imports
        import numpy as np
        import torch
        import torch.nn as nn
        import copy
        import gc

        #Helpers
        @torch.no_grad()
        def eval_accuracy(model, device, loader, max_batches=None):
            model.eval()
            correct, total = 0, 0
            for bi, (x, y) in enumerate(loader):
                if (max_batches is not None) and (bi >= max_batches):
                    break
                x, y = x.to(device), y.to(device)
                out = model(x)
                if hasattr(out, "logits"):
                    out = out.logits
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
            return correct / max(1, total)
        #Bonsai
        def get_param_count(model, exclude=()):
            return sum(p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude))
        #---
        def clear_all_forward_hooks(model):
            for _, m in model.named_modules():
                if hasattr(m, "_forward_hooks") and isinstance(m._forward_hooks, dict):
                    m._forward_hooks.clear()
                for attr in ("_my_hooks", "_my_fc1_hooks", "_my_qkv_hooks"):
                    if hasattr(m, attr):
                        handles = getattr(m, attr)
                        try:
                            for h in handles:
                                try: h.remove()
                                except Exception: pass
                        except TypeError:
                            try: handles.remove()
                            except Exception: pass
                        setattr(m, attr, [])
        #---
        def attach_vit_mlp_caches_only(model):
            clear_all_forward_hooks(model)
            for name, mod in model.named_modules():
                if name.endswith('mlp') and hasattr(mod, 'fc1') and hasattr(mod, 'fc2'):
                    mod.main_mask = None
                    mod.temp_mask = None
                    mod.intermed_cache = None
                    mod.skip_computation = False
                    mod.prune_method = "wanda"
                    mod.intermediate_size = mod.fc1.out_features

                    def _fc1_hook(subm, inp, out, target_module=mod):
                        x = out
                        if x.dim() == 3:
                            x = x.abs().mean(dim=(0,1))
                        else:
                            x = x.abs().mean(dim=0)
                        target_module.intermed_cache = x.view(1, 1, -1)

                    if hasattr(mod.fc1, "_my_fc1_hooks"):
                        for h in mod.fc1._my_fc1_hooks:
                            try: h.remove()
                            except Exception: pass
                    h = mod.fc1.register_forward_hook(_fc1_hook)
                    mod.fc1._my_fc1_hooks = [h]
        #---
        def reattach_mlp_fc1_hooks(model):
            if hasattr(model, "_mlp_cache_handles"):
                try:
                    for h in model._mlp_cache_handles:
                        try: h.remove()
                        except Exception: pass
                except Exception:
                    pass
                model._mlp_cache_handles = []
            attach_vit_mlp_caches_only(model)


        def compute_mlp_importance_linear(args_loc, model, loader, prior_by_name, module_map):
            import torch, numpy as np
            device_local = next(model.parameters()).device

            #we get priors by using hooks
            #we get fixed and to use indexes
            #we simulate the prune
            #we do forward passes so that y=accuracy
            #we use ridge to get importances
            #simulate turning off parts without actually turning off
            def _ensure_fc1_gate(_model):
                for _n, m in _model.named_modules():
                    if hasattr(m, "fc1"):
                        if getattr(m.fc1, "_bonsai_gate", None) is None:
                            def _gate(_, __, out):
                                mk = getattr(m, "temp_mask", None)
                                if mk is None:
                                    return out
                                return out * mk.to(out.dtype).to(out.device)
                            m.fc1._bonsai_gate = m.fc1.register_forward_hook(_gate)
            _ensure_fc1_gate(model)

            #per mask get an accuracy to use as target in the regression
            @torch.no_grad()
            def _acc(_model, _loader, _maxb):
                ok, tot = 0, 0
                bi = 0
                for data in _loader:
                    if bi >= _maxb:
                        break
                    x, y = data
                    x = x.to(device_local); y = y.to(device_local)
                    out = _model(x)
                    if hasattr(out, "logits"):
                        out = out.logits
                    pred = out.argmax(1)
                    ok += (pred == y).sum().item()
                    tot += y.numel()
                    bi += 1
                if tot == 0:
                    return 0.0
                return ok / float(tot)

            def _ridge(X, y, lam):
                Xt = X.T
                XtX = Xt @ X
                U = XtX.shape[0]
                I = torch.eye(U, dtype=X.dtype, device=X.device)
                w = torch.linalg.solve(XtX + lam * I, Xt @ y.view(-1, 1))
                return w.view(-1)

            #params
            mpi= int(getattr(args_loc, "masks_per_iter", 20))
            mb= int(getattr(args_loc, "eval_batches", 10))
            eps= float(getattr(args_loc, "perturb_frac", 0.05))
            lam= float(getattr(args_loc, "ridge_lambda", 1e-3))

            #initialize main_mask X and Y
            X_dict, y_dict = {}, {}
            for name, mod in module_map.items():
                H = int(mod.fc1.out_features)
                if getattr(mod, "main_mask", None) is None or mod.main_mask.numel() != H:
                    mod.main_mask = torch.ones(1, 1, H, device=device_local, dtype=torch.float16)
                _p, _fix, use_idx = prior_by_name[name]
                U = int(use_idx.size)
                X_dict[name] = torch.empty(0, U, dtype=torch.float32, device=device_local)
                y_dict[name] = torch.empty(0, dtype=torch.float32, device=device_local)
            #get X and Y with random masks
            half = mpi // 2
            if half < 1:
                half = 1

            it = 0
            while it < half:
                #mask m
                for name, mod in module_map.items():
                    _p, fixed_idx, use_idx = prior_by_name[name]
                    H = int(mod.fc1.out_features)
                    mm = mod.main_mask.view(-1).detach().cpu().numpy()
                    act_idx = np.where(mm > 0.0)[0]
                    cand = act_idx
                    if fixed_idx.size > 0:
                        cand = np.setdiff1d(act_idx, fixed_idx, assume_unique=False)

                    mvec = mm.copy()
                    if cand.size > 0:
                        k = int(eps * float(mm.sum()))
                        if k < 1:
                            k = 1
                        choose = np.random.choice(cand, size=min(k, cand.size), replace=False)
                        mvec[choose] = 0.0
                    if fixed_idx.size > 0:
                        mvec[fixed_idx] = 1.0

                    mod.temp_mask = torch.tensor(mvec, device=device_local, dtype=mod.main_mask.dtype).view(1, 1, -1)

                y1 = _acc(model, loader, mb)

                for name, mod in module_map.items():
                    _p, _fix, use_idx = prior_by_name[name]
                    cur = (mod.temp_mask.view(-1).detach().to(torch.float32).cpu().numpy())[use_idx]
                    xrow = torch.from_numpy(cur).to(device_local).view(1, -1)
                    X_dict[name] = torch.vstack((X_dict[name], xrow))
                    y_dict[name] = torch.cat((y_dict[name], torch.tensor([y1], device=device_local)))

                #
                for name, mod in module_map.items():
                    cur = mod.temp_mask.view(-1).detach().cpu().numpy()
                    _p, fixed_idx, _u = prior_by_name[name]
                    comp = 1.0 - cur
                    if fixed_idx.size > 0:
                        comp[fixed_idx] = 1.0
                    mod.temp_mask = torch.tensor(comp, device=device_local, dtype=mod.main_mask.dtype).view(1, 1, -1)

                y2 = _acc(model, loader, mb)

                for name, mod in module_map.items():
                    _p, _fix, use_idx = prior_by_name[name]
                    cur = (mod.temp_mask.view(-1).detach().to(torch.float32).cpu().numpy())[use_idx]
                    xrow = torch.from_numpy(cur).to(device_local).view(1, -1)
                    X_dict[name] = torch.vstack((X_dict[name], xrow))
                    y_dict[name] = torch.cat((y_dict[name], torch.tensor([y2], device=device_local)))

                #clean
                for _n2, m2 in module_map.items():
                    m2.temp_mask = None

                it += 1

            #ridge
            importance_scores = {}
            for name, mod in module_map.items():
                X = X_dict[name]
                y = y_dict[name]
                _p, fixed_idx, use_idx = prior_by_name[name]
                H = int(mod.fc1.out_features)

                if X.shape[0] == 0:
                    #no data yet
                    imp = torch.zeros(H, dtype=torch.float32, device=device_local)
                    importance_scores[name] = imp.view(1, 1, -1).detach().cpu()
                    continue

                w = _ridge(X, y, lam)  # [U]
                imp = torch.zeros(H, dtype=torch.float32, device=device_local)
                i = 0
                U = int(use_idx.size)
                while i < U:
                    idx = int(use_idx[i])
                    imp[idx] = w[i]
                    i += 1

                importance_scores[name] = imp.view(1, 1, -1).detach().cpu()

            return importance_scores



        ##Helpers end
        #Bonsai adapted
        def run_data_to_sampling_proba(info, module, pfrac,importance_out=None, module_name=None):
            if (not isinstance(info, dict)) or ('in' not in info):
                H = int(module.fc1.out_features)
                device_local = next(module.parameters()).device
                avg_act_magnitudes = torch.ones(1, 1, H, device=device_local, dtype=torch.float32)
            else:
                avg_act_magnitudes = info['in'][1] / info['in'][0]
            if (importance_out is not None) and (module_name is not None):
                importance_out[module_name] = avg_act_magnitudes.detach().clone()
            sampling_proba = avg_act_magnitudes.detach().cpu().squeeze().numpy()
            H = sampling_proba.shape[0]

            if pfrac is None:
                fixed_indices, use_indices = np.array([], dtype=int), np.arange(H)
            else:
                num_keep_static = int(H * (1.0 -  pfrac))
                order = np.argsort(-sampling_proba)
                fixed_indices, use_indices = order[:num_keep_static], order[num_keep_static:]

            sampling_proba = sampling_proba.max() - sampling_proba
            if sampling_proba.sum() == 0:
                sampling_proba = np.ones_like(sampling_proba)
            sampling_proba[fixed_indices] = 0

            mm = getattr(module, 'main_mask', None)
            if (mm is None) or (mm.numel() != H):
                device_local = next(module.parameters()).device
                module.main_mask = torch.ones(1, 1, H, device=device_local, dtype=torch.float16)
                mm = module.main_mask

            sampling_proba *= mm.detach().cpu().float().squeeze().numpy()
            s = sampling_proba.sum()
            sampling_proba = sampling_proba / s if s > 0 else np.ones_like(sampling_proba) / H
            return sampling_proba, fixed_indices, use_indices


        #Bonsai adapted
        def build_mlp_masks(args_loc, model, loader):
            from collections import defaultdict
            info_cache, hook_handles = defaultdict(dict), {}
            module_map = {}
            importance_scores = {}
            #hooks
            def hook_fn(module_name, info_cache_ref):
                def hook(module, in_, out_):
                    cache = getattr(module, "intermed_cache", None)
                    if cache is None:
                        return
                    flat_in = cache.detach().float()
                    module.intermed_cache = None
                    if 'in' not in info_cache_ref[module_name]:
                        info_cache_ref[module_name]['in'] = [1, flat_in]
                    else:
                        info_cache_ref[module_name]['in'] = [
                            info_cache_ref[module_name]['in'][0] + 1,
                            info_cache_ref[module_name]['in'][1].add_(flat_in)
                        ]
                return hook

            #save hooks on each MLP
            for name, mod in model.named_modules():
                if name.endswith('mlp') and hasattr(mod, 'fc1') and hasattr(mod, 'fc2') and not getattr(mod, 'skip_computation', False):
                    hh = mod.register_forward_hook(hook_fn(name, info_cache))
                    hook_handles[name] = hh
                    module_map[name] = mod

            # Warm-up for the prior
            device_local = next(model.parameters()).device
            _ = eval_accuracy(model, device_local, loader, max_batches=getattr(args_loc, "warmup_batches", 10))

            #now get prior
            prior_by_name = {}  # name is (sampling_proba, fixed_idx, use_idx)
            for name, mod in module_map.items():
                pfrac = getattr(args_loc, "prune_frac", 0.05)
                sampling_proba, fixed_indices, use_indices = run_data_to_sampling_proba(
                    info_cache[name], mod, pfrac,
                    importance_out=importance_scores, module_name=name
                )

                H = int(mod.fc1.out_features)
                if (getattr(mod, 'main_mask', None) is None) or (mod.main_mask.numel() != H):
                    mod.main_mask = torch.ones(1, 1, H, device=next(mod.parameters()).device, dtype=torch.float16)

                prior_by_name[name] = (sampling_proba, fixed_indices, use_indices)


            imp_linear = compute_mlp_importance_linear(args_loc=args_loc,model=model,loader=loader,prior_by_name=prior_by_name,module_map=module_map)

            # save imp scores
            for name, imp in imp_linear.items():
                importance_scores[name] = imp

            #build the masks
            mask_dict = {}
            pfrac_now = float(getattr(args_loc, "prune_frac", 0.05))

            for name, mod in module_map.items():
                H = int(mod.fc1.out_features)
                mm = mod.main_mask.view(-1).detach().to(torch.float32)
                scores = importance_scores[name].to(device_local).view(-1)

                #
                act_idx = torch.nonzero(mm > 0.0, as_tuple=False).view(-1)

                #
                _p, fixed_idx, use_idx = prior_by_name[name]
                if fixed_idx.size > 0:
                    fm = torch.zeros(H, dtype=torch.bool, device=mm.device)
                    j = 0
                    K = int(fixed_idx.size)
                    while j < K:
                        fm[int(fixed_idx[j])] = True
                        j += 1
                    act_idx = act_idx[~fm[act_idx]]

                if act_idx.numel() > 0:
                    k = int(pfrac_now * float(mm.sum().item()))
                    if k < 1:
                        k = 1
                    #
                    vals = scores[act_idx]
                    _, order = torch.sort(vals)
                    k = min(k, act_idx.numel())
                    kill = act_idx[order[:k]]
                    mm[kill] = 0.0

                mod.main_mask = mm.to(mod.main_mask.device, dtype=mod.main_mask.dtype).view(1, 1, -1)
                mask_dict[name] = mod.main_mask

            #remove hooks
            for h in hook_handles.values():
                try:
                    h.remove()
                except Exception:
                    pass

            return mask_dict, importance_scores

        try:
            from transformers.pytorch_utils import prune_linear_layer
        except Exception:
            prune_linear_layer = None
        #Bonsai adapted
        def prune_mlp_vit(mask_, module: nn.Module):
            module.temp_mask = None
            module.intermed_cache = None
            if mask_.mean() == 0:
                module.fc1 = nn.Identity()
                module.fc2 = nn.Identity()
                module.intermediate_size = 0
                module.skip_computation = True
                gc.collect(); torch.cuda.empty_cache()
                return
            keep = mask_.squeeze().nonzero().squeeze()
            if keep.numel() == 1:
                keep = keep.view(1)
            if prune_linear_layer is None:
                with torch.no_grad():
                    module.fc1.weight = nn.Parameter(module.fc1.weight[keep].contiguous())
                    if module.fc1.bias is not None:
                        module.fc1.bias = nn.Parameter(module.fc1.bias[keep].contiguous())
                    module.fc2.weight = nn.Parameter(module.fc2.weight[:, keep].contiguous())
            else:
                module.fc1 = prune_linear_layer(module.fc1, keep)
                module.fc2 = prune_linear_layer(module.fc2, keep, dim=1)
            module.intermediate_size = keep.numel()
            gc.collect(); torch.cuda.empty_cache()
        #---
        def prune_model_mlp_only(mask_info, model):
            for name, mod in model.named_modules():
                if name in mask_info and name.endswith('mlp') and hasattr(mod, 'fc1') and hasattr(mod, 'fc2'):
                    prune_mlp_vit(mask_info[name], mod)
            gc.collect(); torch.cuda.empty_cache()
        #---

        #now we use a model copy to remove neurons and see best masks without actually damaging the model
        def final_mlp_masks_from_pruned_copy(model, loader, args_loc, target=None, tol=0.01, max_iters=8,
                                             report_eval=True, eval_batches=10):
            device_local = next(model.parameters()).device
            target_val = float(getattr(args_loc, "sparsity_ratio", None) if target is None else target)

            m = copy.deepcopy(model).to(device_local).eval()
            attach_vit_mlp_caches_only(m)

            alive, H0 = {}, {}
            for name, mod in m.named_modules():
                if name.endswith('mlp') and hasattr(mod, 'fc1') and hasattr(mod, 'fc2'):
                    H0[name] = int(mod.fc1.out_features)
                    alive[name] = list(range(H0[name]))
            #---
            def logical_sparsity():
                kept = sum(len(v) for v in alive.values())
                total = sum(H0.values()) if len(H0) else 1
                return 1.0 - kept / total

            traj_spars, traj_top1 = [], []
            cur_s = logical_sparsity()
            it = 1
            last_imp_scores = {}

            #importance_original before pruning, save importances before pruning
            importance_original = None
            while True:
                if (abs(cur_s - target_val) <= tol) or (cur_s >= target_val) or (it > max_iters):
                    break
                pf_now = getattr(args_loc, "prune_frac", 0.02)
                if cur_s + pf_now > target_val:
                    setattr(args_loc, "prune_frac", max(1e-3, target_val - cur_s))

                #print(f"[Iter {it}] cur_s={cur_s:.3f} target={target_val:.3f} prune_frac={getattr(args_loc,'prune_frac'):.3f}")
                reattach_mlp_fc1_hooks(m)

                mask_info, imp_scores  = build_mlp_masks(args_loc, m, loader)
                importance_last = {}
                for name, imp in imp_scores.items():
                    if name not in H0:
                        continue
                    H = H0[name]
                    prev = alive.get(name, [])
                    flat = imp.view(-1).detach().cpu()# (H_cur,)
                    trans = torch.zeros(1, 1, H)# (1,1,H0)
                    take = min(len(prev), flat.numel())
                    if take > 0:
                        #mapping back to original indexes
                        trans.view(-1)[torch.as_tensor(prev[:take])] = flat[:take]
                    importance_last[name] = trans
                last_imp_scores = importance_last

                #save the importances
                if importance_original is None:
                    importance_original = {
                        k: v.detach().clone() for k, v in importance_last.items()
                    }

                for name, mk in mask_info.items():
                    if name not in alive:
                        continue
                    prev = alive[name]
                    if len(prev) == 0:
                        continue
                    keep_cur = torch.nonzero(mk.view(-1) > 0, as_tuple=False).view(-1).tolist()
                    keep_cur = [i for i in keep_cur if i < len(prev)]
                    alive[name] = [prev[i] for i in keep_cur]

                prune_model_mlp_only(mask_info, m)

                cur_s = logical_sparsity()
                if report_eval:
                    top1_it = eval_accuracy(m, device_local, loader, max_batches=eval_batches)
                    traj_top1.append(float(top1_it))
                traj_spars.append(float(cur_s))
                it += 1
            #print(f"FINAL sparsity(MLP)={cur_s:.3f} (objective={target_val:.3f})")
            top1_copy_final = None
            if report_eval:
                top1_copy_final = float(eval_accuracy(m, device_local, loader, max_batches=eval_batches))

            final_masks = {}
            for name, H in H0.items():
                mask = torch.zeros(H, dtype=torch.float16, device=device_local)
                if len(alive[name]) > 0:
                    mask[alive[name]] = 1.0
                final_masks[name] = mask.view(1, 1, -1)

            #checking by layer
            #print("\n Masks per layer :")
            tot_keep, tot_H = 0, 0
            for name, msk in final_masks.items():
                H = msk.numel()
                k = int(msk.detach().float().sum().item())
                #print(f"  {name:20s} keep={k:4d} / H={H:4d}  -> spars={1 - k/H:.3f}")
                tot_keep += k; tot_H += H
            #print(f"Cheking sparsity (MLP global) with final_masks: {1 - tot_keep/max(tot_H,1):.3f}")

            #turn importances dict into a matriz as needed
            importance_original_matrix = None
            if importance_original is not None and len(importance_original) > 0:
                rows = []
                #sort key blocks.i.mlp by index i
                keys = sorted(importance_original.keys())
                for k in keys:
                    v = importance_original[k].view(1, -1).to(device_local)
                    rows.append(v)
                if len(rows) > 0:
                    importance_original_matrix = torch.cat(rows, dim=0)  # (num_blocks, H_original)
            metrics = {
                "top1_copy_final": top1_copy_final,
                "traj_sparsity": traj_spars,
                "traj_top1": traj_top1,
                "importance_last": last_imp_scores,
                "importance_original": importance_original,
                "importance_original_matrix": importance_original_matrix,
            }

            del m
            torch.cuda.empty_cache()
            return final_masks, metrics
        #---
        def apply_runtime_gates(model, masks):
            device_local = next(model.parameters()).device
            applied, missing = 0, []
            for name, mk in masks.items():
                found = False
                for mod_name, mod in model.named_modules():
                    if mod_name == name and mod_name.endswith('mlp') and hasattr(mod, 'fc1'):
                        found = True
                        mask = mk.to(device_local).view(1,1,-1)
                        if hasattr(mod.fc1, "_gate_handle") and mod.fc1._gate_handle is not None:
                            try: mod.fc1._gate_handle.remove()
                            except: pass
                            mod.fc1._gate_handle = None
                        def _gate(subm, inp, out, mask=mask):
                            return out * (mask if out.dim()==3 else mask.view(-1))
                        mod.fc1._gate_handle = mod.fc1.register_forward_hook(_gate)
                        applied += 1
                        break
                if not found:
                    missing.append(name)
            #if missing:
            #    print("Layer not found:", missing)
            #print(f"Zeroed applied to {applied}  MLP layers")
            return applied, missing
        #---


        import torch

        def vector_to_masks_fc(vec_1x1x4E, fc1_weight, fc2_weight): #why?
            # vec_1x1x4E: tensor (1,1,4E)
            # fc1_weight: tensor (4E, E)
            # fc2_weight: tensor (E, 4E)
            vec = (vec_1x1x4E != 0).reshape(-1).to(torch.bool)

            #expand to 2d
            rows_fc1, cols_fc1 = fc1_weight.shape   # (4E, E)
            rows_fc2, cols_fc2 = fc2_weight.shape   # (E, 4E)

            fc1_mask_2d = vec[:, None].expand(rows_fc1, cols_fc1)  # (4E, E)
            fc2_mask_2d = vec[None, :].expand(rows_fc2, cols_fc2)  # (E, 4E)

            return fc1_mask_2d, fc2_mask_2d


        def build_translated_mask(masks_dict, model):
            out = []
            i = 0
            total = len(model.blocks)
            while i < total:
                key = f"blocks.{i}.mlp"
                vec = masks_dict[key]
                fc1_w = model.blocks[i].mlp.fc1.weight  # (4E, E)
                fc2_w = model.blocks[i].mlp.fc2.weight  # (E, 4E)
                fc1_mask_2d, fc2_mask_2d = vector_to_masks_fc(vec, fc1_w, fc2_w)
                par = [fc1_mask_2d.to(torch.bool), fc2_mask_2d.to(torch.bool)]
                out.append(par)

                i = i + 1

            return out

        #Execution
        #Brief Explanation:
        #First we attach the hooks and count parameters to count as frame of reference for sparsity
        #Get base accuracy
        #Then using final_mlp_masks_from_pruned_copy we deepcopy the model and gathers importances scores and returns masks (final_masks), accuracy and importance scores (importance_last)
        # #First we make a copy of the model and attach hooks on it
        # #We measure the original width and the index of unpruned points (at first all are) and calculate "sparsity" (zeroes)
        # #While "sparsity"< objective:
        # # #Get scores and returns masks and importance scores (mask_info) and (imp_scores)
        # # #Because of the previously obtained indexes of every point, we convert the importance scores in the pruned copy back to the original matrix size. Since the method is iterative, the index in iteration2 will be different from what they were at first (as well as dimensions), but this way we recover them.
        #Translate the masks dictionary into a matrix
        #Use apply_runtime_gates to zero out based on the mask, but is currently left commented as we intend to only get scores and not affecting the model
        #Calculate using final_masks the "sparsity" (zeroes)
        self.device = device
        mode = getattr(args, "mode", "gating")

        # Baseline
        attach_vit_mlp_caches_only(self.nn)
        self.original_param_count = get_param_count(self.nn)
        print("Param Original:", self.original_param_count)
        self.base_top1 = eval_accuracy(self.nn, self.device, self.dl, max_batches=getattr(args, "warmup_batches", 10))
        print(f"Accuracy base (sample): {self.base_top1:.4f}")

        #Build Mask from the copy
        #print("Building Masks in copy")
        self.final_masks, self.copy_metrics = final_mlp_masks_from_pruned_copy(
            self.nn, self.dl, args,
            target=getattr(args, "sparsity_ratio", 0.20),
            tol=getattr(args, "tol", 0.01),
            max_iters=getattr(args, "max_iters", 8),
            report_eval=True,
            eval_batches=getattr(args, "eval_batches", 10)
        )   #FINAL MASKS ARE HERE

        final_masks_translated = build_translated_mask(self.final_masks, self.nn)

        self.mask_fc_width = final_masks_translated

        #checking first block
        #try:
        #    _fc1, _fc2 = final_masks_translated[0]
        #    print("mask_fc[0] shapes:", tuple(_fc1.shape), tuple(_fc2.shape),"| dtypes:", _fc1.dtype, _fc2.dtype)
        #except Exception as _e:
        #    print("CANT check mask_fc[0]:", _e)

        #print("Final_masks ready")
        #if self.copy_metrics.get("top1_copy_final") is not None:
        #    print(f"Final Accuracy with pruning: {self.copy_metrics['top1_copy_final']:.4f}")

        # Calcular sparsidad lógica a partir de las máscaras (independiente del modo)
        kept, total = 0.0, 0.0
        for m in self.final_masks.values():
            v = m.detach().float().view(-1)
            kept += v.sum().item()
            total += v.numel()
        self.logical_sparsity = 1.0 - kept / max(total, 1e-6)

        self.top1_gate = None
        self.top1_pruned = None

        if mode == "gating":
            apply_runtime_gates(self.nn, self.final_masks)
            self.top1_gate = eval_accuracy(self.nn, self.device, self.dl, max_batches=getattr(args, "eval_batches", 10))
        elif mode == "pruning":
            prune_model_mlp_only(self.final_masks, self.nn)
            self.top1_pruned = eval_accuracy(self.nn, self.device, self.dl, max_batches=getattr(args, "eval_batches", 10))
        else:
            apply_runtime_gates(self.nn, self.final_masks)
            self.top1_gate = eval_accuracy(self.nn, self.device, self.dl, max_batches=getattr(args, "eval_batches", 10))

        #Returns
        return {
            "params_original": self.original_param_count,
            "top1_base": self.base_top1,
            "top1_copy_final": self.copy_metrics.get("top1_copy_final", None),
            "logical_sparsity": self.logical_sparsity,
            "top1_gate": self.top1_gate,
            "top1_pruned": self.top1_pruned,
            "mask_fc_width": self.mask_fc_width,
            "importance_last": self.copy_metrics.get("importance_last", {}),
            "importance_original": self.copy_metrics.get("importance_original", {}),
            "importance_original_matrix": self.copy_metrics.get("importance_original_matrix", None),
        }

    def compute_importances(self, device, args):

        import copy
        import torch
        args_loc = copy.deepcopy(args)
        args_loc.mode = "importances"
        results = self.fit(device, args_loc)
        imp_dict = results.get("importance_original", {})
        if (not isinstance(imp_dict, dict)) or len(imp_dict) == 0:
            self.mlp_importance = None
            return None
        ffn_importances = {}

        if hasattr(self.nn, "blocks"):
            num_blocks = len(self.nn.blocks)
        else:
            num_blocks = 0

        i = 0
        while i < num_blocks:
            layer_name = f"blocks.{i}.mlp"
            if layer_name in imp_dict:
                tens = imp_dict[layer_name]
                flat = tens.view(-1).detach().cpu().tolist()

                j = 0
                while j < len(flat):
                    key_ij = f"{i}:{j}"
                    ffn_importances[key_ij] = float(flat[j])
                    j = j + 1

            i = i + 1

        json_out = {"ffn": ffn_importances}
        self.mlp_importance = json_out

        return json_out
