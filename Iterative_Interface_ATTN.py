class PruningInterface:
    def __init__(self, model, loader):
        self.nn = model
        self.dl = loader
        self.device = None
        self.original_param_count = None
        self.base_top1 = None
        self.top1_gate = None
        self.logical_sparsity = None

        self.final_masks = {}
        self.copy_metrics = {}

        self.mask_fc_width = None
        self._gate_handles = {}
        self.att_importance_list = None
        #IMPORTANT INFO ABOUT ARGS: 
        #SimpleNamespace object
        #to get importances for all neurons, set prune_frac = 1.0
        #prune_frac, sparsity_ratio, tol, max_iters,
        #masks_per_iter, eval_batches, warmup_batches,
        #perturb_frac, ridge_lambda, mode='importances' (if we dont want to affect the model but rather just get importances)


    def fit(self, device, args):
        #Execution
        import numpy as np
        import torch
        import torch.nn as nn
        import copy
        import gc
        self.device = device
        self.mode = getattr(args, "mode", "gating")
        #helpers
        #helper for head sparsity
        def _heads_sparsity_from_masks(self, mask_dict):
            kept, total = 0.0, 0.0
            for m in mask_dict.values():
                v = m.detach().float().view(-1)
                kept += v.sum().item()
                total += v.numel() #number of elements
            return 1.0 - kept / max(total, 1e-6)

        #helper for parameters count
        def _get_param_count(self, model: torch.nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p is not None)

        #helper to get accuracy
        @torch.no_grad()
        def _eval_accuracy(self, model, device, loader, max_batches=10):
            model.eval()
            correct, total = 0, 0
            batches = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                #adaptation to what the model outpus
                if hasattr(out, "logits"):
                    out = out.logits
                if isinstance(out, (tuple, list)):
                    out = out[0]
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
                batches += 1
                if batches >= max_batches:
                    break
            return correct / max(1, total)

        #forward pass based importance with regression
        def compute_attn_importance_linear(args_loc, model, loader, prior_by_name, module_map):
            device_local = next(model.parameters()).device
            #simulate turning off heads
            def _ensure_attn_head_gate(_model):
                for _n, m in _model.named_modules():
                    if hasattr(m, "qkv") and hasattr(m, "num_heads") and hasattr(m, "head_dim"):
                        if getattr(m.qkv, "_bonsai_gate", None) is None:  #avoid getting duplicated hooks
                            def _gate(subm, inp, out, parent=m):   #forward hook
                                # out: (B, N, 3*D) con D=H*d
                                mk = getattr(parent, "temp_head_mask", None)  # (1,1,H) con 0/1
                                if mk is None:
                                    return out
                                B, N, threeD = out.shape
                                H = int(getattr(parent, "num_heads"))
                                d = int(getattr(parent, "head_dim"))
                                # (B,N,3,H,d) * (1,1,1,H,1)
                                qkv = out.view(B, N, 3, H, d)
                                gate = mk.to(out.dtype).to(out.device).view(1, 1, 1, H, 1)
                                qkv = qkv * gate #apply the gate to the head
                                return qkv.view(B, N, threeD)
                            m.qkv._bonsai_gate = m.qkv.register_forward_hook(_gate)
            _ensure_attn_head_gate(model)

            #get accuracy with the simulated masks
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
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    pred = out.argmax(1)
                    ok  += (pred == y).sum().item()
                    tot += y.numel()
                    bi  += 1
                return 0.0 if tot == 0 else ok / float(tot)

            def _ridge(X, y, lam):
                Xt = X.T
                XtX = Xt @ X
                U = XtX.shape[0]
                I = torch.eye(U, dtype=X.dtype, device=X.device)
                w = torch.linalg.solve(XtX + lam * I, Xt @ y.view(-1, 1))
                return w.view(-1)

            #some parameters
            mpi= int(getattr(args_loc, "masks_per_iter", 8))
            mb= int(getattr(args_loc, "eval_batches", 10))
            eps= float(getattr(args_loc, "perturb_frac", 0.05))
            lam= float(getattr(args_loc, "ridge_lambda", 1e-3))


            #now we go through the mask options to get their accuracy, apply regression and so get importances
            X_dict, y_dict = {}, {}
            for name, mod in module_map.items():
                H = int(mod.num_heads)
                if getattr(mod, "main_mask", None) is None or mod.main_mask.numel() != H:
                    mod.main_mask = torch.ones(1, 1, H, device=device_local, dtype=torch.float16)
                _p, _fix, use_idx = prior_by_name[name]
                U = int(use_idx.size)
                X_dict[name] = torch.empty(0,U,dtype=torch.float32, device=device_local)
                y_dict[name] = torch.empty(0, dtype=torch.float32, device=device_local)

            #now collecting samples
            half = max(1, mpi // 2)
            it = 0
            while it < half:
                for name, mod in module_map.items():
                    _p, fixed_idx, use_idx = prior_by_name[name]#here we take the fixed and to use indexes
                    H = int(mod.num_heads)#total heads
                    mm = mod.main_mask.view(-1).detach().cpu().numpy()#currently active heads

                    #candidates
                    act_idx = np.where(mm > 0.0)[0]
                    cand = np.setdiff1d(act_idx, fixed_idx, assume_unique=False) if fixed_idx.size > 0 else act_idx

                    mvec = mm.copy()
                    if cand.size > 0:
                        k = int(eps * float(mm.sum()))#turn off
                        k = max(1, min(k, cand.size))
                        choose = np.random.choice(cand, size=k, replace=False)
                        mvec[choose] = 0.0

                    if fixed_idx.size > 0:
                        mvec[fixed_idx] = 1.0#make fixed ones 1s the mask
                    mod.temp_head_mask = torch.tensor(mvec, device=device_local, dtype=mod.main_mask.dtype).view(1, 1, -1)

                #get accuracy for this particular temporary mask
                y1 = _acc(model, loader, mb)

                #we check with the candidates off and on respectively
                for name, mod in module_map.items():
                    _p, _fix, use_idx = prior_by_name[name]
                    cur = (mod.temp_head_mask.view(-1).detach().to(torch.float32).cpu().numpy())[use_idx]
                    xrow = torch.from_numpy(cur).to(device_local).view(1, -1)
                    X_dict[name] = torch.vstack((X_dict[name], xrow))
                    y_dict[name] = torch.cat((y_dict[name], torch.tensor([y1], device=device_local)))

                for name, mod in module_map.items():
                    _p, fixed_idx, _u = prior_by_name[name]
                    cur = mod.temp_head_mask.view(-1).detach().cpu().numpy()
                    comp = 1.0 - cur
                    if fixed_idx.size > 0:
                        comp[fixed_idx] = 1.0
                    mod.temp_head_mask = torch.tensor(comp, device=device_local, dtype=mod.main_mask.dtype).view(1, 1, -1)

                y2 = _acc(model, loader, mb)

                #
                for name, mod in module_map.items():
                    _p, _fix, use_idx = prior_by_name[name]
                    cur = (mod.temp_head_mask.view(-1).detach().to(torch.float32).cpu().numpy())[use_idx]
                    xrow = torch.from_numpy(cur).to(device_local).view(1, -1)
                    X_dict[name] = torch.vstack((X_dict[name], xrow))
                    y_dict[name] = torch.cat((y_dict[name], torch.tensor([y2], device=device_local)))

                #cleanup
                for _n2, m2 in module_map.items():
                    m2.temp_head_mask = None

                it += 1


            #apply ridge
            importance_scores = {}
            for name, mod in module_map.items():
                X = X_dict[name]   # [S, U]
                y = y_dict[name]   # [S]
                _p, fixed_idx, use_idx = prior_by_name[name]
                H = int(mod.num_heads)

                if X.shape[0] == 0:
                    imp = torch.zeros(H, dtype=torch.float32, device=device_local)
                    importance_scores[name] = imp.view(1, 1, -1).detach().cpu()
                    continue

                w = _ridge(X, y, lam)                         # [U]
                imp = torch.zeros(H, dtype=torch.float32, device=device_local)
                U  = int(use_idx.size)
                i = 0
                while i < U:
                    idx = int(use_idx[i])
                    imp[idx] = w[i]
                    i += 1
                importance_scores[name] = imp.view(1, 1, -1).detach().cpu()

            return importance_scores
        #

        #main functions
        #forward-pass importances


        def run_data_to_sampling_proba(info, module, pfrac, importance_out=None, module_name=None):
            #similar to MLP version
            if (not isinstance(info, dict)) or ('in' not in info):
                H = int(module.num_heads)
                device_local = next(module.parameters()).device
                avg_act_magnitudes = torch.ones(1, 1, H, device=device_local, dtype=torch.float32)
            else:
                avg_act_magnitudes = info['in'][1] / info['in'][0]
            if (importance_out is not None) and (module_name is not None):
                importance_out[module_name] = avg_act_magnitudes.detach().clone()
            sampling_proba = avg_act_magnitudes.detach().cpu().squeeze().numpy()
            H = sampling_proba.shape[0]

            #which are fixed and which prunable?
            if pfrac is None:
                fixed_indices, use_indices = np.array([], dtype=int), np.arange(H)
            else:
                num_keep_static = int(H * (1.0 - pfrac))
                order = np.argsort(-sampling_proba)
                fixed_indices, use_indices = order[:num_keep_static], order[num_keep_static:]

            #least important are more likely to be pruned
            sampling_proba = sampling_proba.max() - sampling_proba
            if sampling_proba.sum() == 0:
                sampling_proba = np.ones_like(sampling_proba)
            #
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



        def build_attn_masks(self, args_loc, model, loader):

            #get importances per head, taken by the hook
            #turns off les important ones
            from collections import defaultdict

            info_cache, hook_handles = defaultdict(dict), {}
            module_map = {}
            importance_scores = {}
            mask_dict = {}

            #hooks to accumulate importances per head
            def hook_fn(module_name, info_cache, get_heads, get_head_dim):
                def hook(module, in_, out_):
                    if out_ is None:
                        return
                    # out_: (B, N, D)
                    try:
                        B, N, D = out_.shape
                    except Exception:
                        return
                    H = int(get_heads(module))
                    d = int(get_head_dim(module)) if hasattr(module, "head_dim") else (D // H)
                    if H * d != D or H <= 0 or d <= 0:
                        return
                    #(B,N,D) to (B,N,H,d)
                    y = out_.detach()
                    y = y.view(B, N, H, d)
                    #imp per head: mean(y)
                    imp = y.abs().mean(dim=(0, 1, 3)).float()  # (H,)
                    if 'in' not in info_cache[module_name]:
                        info_cache[module_name]['in'] = [1, imp]
                    else:
                        info_cache[module_name]['in'][0] += 1
                        info_cache[module_name]['in'][1].add_(imp)
                return hook

            #now we go per head collecting hooks
            for name, mod in model.named_modules():
                if name.endswith('attn') and hasattr(mod, 'num_heads') and not getattr(mod, 'skip_computation', False):
                    hh = mod.register_forward_hook(
                        hook_fn(name, info_cache, lambda m: m.num_heads, lambda m: getattr(m, "head_dim", None))
                    )
                    hook_handles[name] = hh
                    module_map[name] = mod

            #forward pass to get the activations
            device_local = next(model.parameters()).device
            _ = _eval_accuracy(self, model, device_local, loader, max_batches=getattr(args_loc, "warmup_batches", 10))

            prior_by_name = {}
            base_pfrac = float(getattr(args_loc, "prune_frac", 0.02))
            for name, mod in module_map.items():
                try:
                    sampling_proba, fixed_idx, use_idx = run_data_to_sampling_proba(info_cache[name], mod, base_pfrac)
                except Exception:
                    sampling_proba, fixed_idx, use_idx = run_data_to_sampling_proba(info_cache[name], mod, base_pfrac) #forcing for testing
                    #potential error management
                    #H = int(mod.num_heads)
                    #sampling_proba = np.ones(H, dtype=np.float32) / max(H, 1)
                    #fixed_idx= np.array([], dtype=np.int64)
                    #use_idx= np.arange(H, dtype=np.int64)
                sampling_proba= np.asarray(sampling_proba, dtype=np.float32)
                fixed_idx= np.asarray(fixed_idx, dtype=np.int64)
                use_idx= np.asarray(use_idx, dtype=np.int64)

                prior_by_name[name] = (sampling_proba, fixed_idx, use_idx)
            imp_linear = compute_attn_importance_linear(args_loc, model, loader, prior_by_name, module_map) #we compute importances

            #get importances and build mask
            for name, mod in module_map.items():
                H = int(mod.num_heads)

                #starting mask is all 1s
                mm = getattr(mod, 'main_mask', None)
                if (mm is None) or (mm.numel() != H):
                    mod.main_mask = torch.ones(1, 1, H, device=device_local, dtype=torch.float16)
                    mm = mod.main_mask

                alive_vec = mm.detach().float().view(-1).cpu().numpy()
                non_zero = np.where(alive_vec > 0.5)[0]
                num_alive = len(non_zero)

                imp = imp_linear.get(name, torch.zeros(1,1,H)).view(-1).detach().cpu().numpy()
                importance_scores[name] = torch.tensor(imp, dtype=torch.float32)

                if num_alive <= 1:
                    mask_dict[name] = mm
                    continue

                #adaptative prune_frac
                target= float(getattr(args_loc, "sparsity_ratio", 0.20))
                tmp_all= dict(mask_dict)
                tmp_all[name]= mm
                current_spars= _heads_sparsity_from_masks(self, tmp_all)
                gap = max(0.0, target - current_spars)
                base_pfrac= float(getattr(args_loc, "prune_frac", 0.02))
                eff_pfrac= max(0.005, min(base_pfrac, 0.6 * gap))

                #how many heads to turn off? (>=1 y < num_alive)
                num_to_zero= int(np.ceil(eff_pfrac * num_alive))
                num_to_zero= min(max(1, num_to_zero), num_alive - 1)

                imp_alive = imp[non_zero]
                k = min(num_to_zero, len(non_zero))
                take_rel = np.argpartition(imp_alive, kth=k - 1)[:k]
                chosen = non_zero[take_rel]

                mm_np = alive_vec.copy()
                mm_np[chosen] = 0.0
                mod.main_mask = torch.tensor(mm_np.reshape(1, 1, H), dtype=torch.float16, device=device_local)
                mask_dict[name] = mod.main_mask

            #remove hooks
            for h in hook_handles.values():
                try:
                    h.remove()
                except Exception:
                    pass
            return mask_dict, importance_scores

        #gating
        def _clear_runtime_head_gates(self, model):
            #remove prev hooks
            for name, handle in list(self._gate_handles.items()):
                try:
                    handle.remove()
                except Exception:
                    pass
                self._gate_handles.pop(name, None)

        def _apply_runtime_head_gates(self, model, mask_dict):
            _clear_runtime_head_gates(self, model)
            #adds a hook
            def make_gate_hook(mask_tensor, get_heads, get_head_dim):
                def hook(module, in_, out_):
                    if out_ is None:
                        return out_
                    try:
                        B, N, D = out_.shape
                    except Exception:
                        return out_
                    H = int(get_heads(module))
                    d = int(get_head_dim(module)) if hasattr(module, "head_dim") else (D // H)
                    if H * d != D or H <= 0 or d <= 0:
                        return out_
                    gate = mask_tensor.to(out_.device, dtype=out_.dtype)  # (1,1,H)
                    y = out_.view(B, N, H, d)
                    y = y * gate.view(1, 1, H, 1)
                    return y.view(B, N, D)
                return hook

            for name, mod in model.named_modules():
                if name.endswith('attn') and hasattr(mod, 'num_heads'):
                    m = mask_dict.get(name, None)
                    if m is None:
                        continue
                    hh = mod.register_forward_hook(
                        make_gate_hook(m.detach().clone(), lambda m: m.num_heads, lambda m: getattr(m, "head_dim", None))
                    )
                    self._gate_handles[name] = hh


        #build the attn masks ON A COPY of the model
        def final_attn_masks_from_pruned_copy(self, model, loader, args_loc,target=None, tol=0.01, max_iters=20, report_eval=True, eval_batches=10):

            #make a copy of model and apply hooks
            m = copy.deepcopy(model).to(next(model.parameters()).device)

            it = 0
            global_masks = {}
            history = []
            #save importances before pruning
            importance_original_dict = None
            importance_original_matrix = None

            while it < max_iters:
                mask_info, imp_scores = build_attn_masks(self, args_loc, m, loader)
                if it == 0:
                    importance_original_dict = imp_scores
                    if isinstance(imp_scores, dict) and len(imp_scores) > 0:
                        #sort key blocks.i.attn by index i
                        def _block_idx(name):
                            try:
                                return int(name.split('.')[1])
                            except Exception:
                                return 0
                        keys_sorted = sorted(imp_scores.keys(), key=_block_idx)
                        rows = []
                        for k in keys_sorted:
                            v = imp_scores[k]
                            if isinstance(v, torch.Tensor):
                                rows.append(v.view(-1))  # (H,)
                            else:
                                rows.append(torch.tensor(v, dtype=torch.float32).view(-1))
                        # matrix [num_blocks, H] with original imps
                        importance_original_matrix = torch.stack(rows, dim=0)

                #use AND to combine masks from the iteration
                for k, newm in mask_info.items():
                    if k not in global_masks:
                        global_masks[k] = newm.detach().clone()
                    else:
                        prev = global_masks[k].detach().clone().to(newm.device)
                        global_masks[k] = (prev * newm).to(newm.dtype)

                #apply the gating IN THE COPY
                _apply_runtime_head_gates(self, m, global_masks)
                s = _heads_sparsity_from_masks(self, global_masks)
                it += 1
                base_pfrac = float(getattr(args_loc, "prune_frac", 0.02))
                gap = max(0.0, float(target) - float(s))
                eff_pfrac = max(0.005, min(base_pfrac, 0.6 * gap))
                history.append((it, s, eff_pfrac))
                if s + 1e-12 >= float(target) or it >= max_iters:
                    break

            top1_copy_final = None
            if report_eval:
                top1_copy_final = _eval_accuracy(self, m, next(m.parameters()).device, loader, max_batches=eval_batches)
            out_metrics = {
                "iterations": it,
                "sparsity_final": _heads_sparsity_from_masks(self, global_masks),
                # última iteración (post-poda en la copia)
                "importance_last": imp_scores,
                # importancias originales (primera iteración, modelo denso)
                "importance_original": importance_original_dict,
                "importance_original_matrix": importance_original_matrix,
                "history": history,
                "top1_copy_final": top1_copy_final,
            }
            return global_masks, out_metrics

        #EXECUTE
        # Baseline
        self.original_param_count = _get_param_count(self, self.nn)
        self.base_top1 = _eval_accuracy(self, self.nn, self.device, self.dl, max_batches=getattr(args, "warmup_batches", 10))
        #Build Mask from the copy
        self.final_masks, self.copy_metrics = final_attn_masks_from_pruned_copy(
            self, self.nn, self.dl, args,
            target=getattr(args, "sparsity_ratio", 0.20),
            tol=getattr(args, "tol", 0.01),
            max_iters=getattr(args, "max_iters", 8),
            report_eval=True,
            eval_batches=getattr(args, "eval_batches", 10)
        )   # FINAL MASKS ARE HERE

        #importances before pruning (NEEDED)
        importance_original_dict = self.copy_metrics.get("importance_original", {})
        importance_original_matrix = self.copy_metrics.get("importance_original_matrix", None)

        # importances after pruning (not needed)
        importance_last_dict = self.copy_metrics.get("importance_last", {})

        self.importance_original_dict = importance_original_dict
        self.importance_original_matrix = importance_original_matrix
        self.importance_last_dict = importance_last_dict
        self.logical_sparsity = _heads_sparsity_from_masks(self, self.final_masks)

        mode = getattr(args, "mode", "gating")
        self.mode = mode

        if mode == "gating":
            #Apply to real model (gating)
            _apply_runtime_head_gates(self, self.nn, self.final_masks)
            self.top1_gate = _eval_accuracy(self, self.nn, self.device, self.dl,
                                            max_batches=getattr(args, "eval_batches", 10))

        elif mode == "importances":
            #just get importances
            _clear_runtime_head_gates(self, self.nn)
            self.top1_gate = None

            # in case we only need to extract importances, we will also normalize
            _clear_runtime_head_gates(self, self.nn)
            self.top1_gate = None

            imp_mat = self.copy_metrics.get("importance_original_matrix", None)
            if imp_mat is not None:
                max_abs = imp_mat.abs().max()
                if max_abs > 0:
                    norm_mat = imp_mat / max_abs
                    self.copy_metrics["importance_original_matrix"] = norm_mat

                    imp_dict = self.copy_metrics.get("importance_original", {})
                    if isinstance(imp_dict, dict) and len(imp_dict) > 0:
                        keys = sorted(imp_dict.keys())
                        for i, k in enumerate(keys):
                            v_old = imp_dict[k]
                            imp_dict[k] = norm_mat[i].view_as(v_old)
                        self.copy_metrics["importance_original"] = imp_dict

        elif mode == "pruning":
            print("unavailable")
        else:
            #
            _apply_runtime_head_gates(self, self.nn, self.final_masks)
            self.top1_gate = _eval_accuracy(self, self.nn, self.device, self.dl,
                                            max_batches=getattr(args, "eval_batches", 10))

        return {
            "params_original": self.original_param_count,
            "top1_base": self.base_top1,
            "top1_copy_final": self.copy_metrics.get("top1_copy_final", None),
            "logical_sparsity": self.logical_sparsity,
            "top1_gate": self.top1_gate,
            "mask_fc_width": self.mask_fc_width,
            "importance_last": self.copy_metrics.get("importance_last", {}),
            "importance_original": importance_original_dict,
            "importance_original_matrix": importance_original_matrix,
        }


    def compute_importances(self, device, args):

        import copy
        args_loc = copy.deepcopy(args)
        args_loc.mode = "importances"

        results = self.fit(device, args_loc)
        imp_mat = results.get("importance_original_matrix", None)

        if imp_mat is None:
            self.att_importance_list = []
            return self.att_importance_list

        self.att_importance_list = []

        for row in imp_mat:
            self.att_importance_list.append(row.detach())

        return self.att_importance_list
