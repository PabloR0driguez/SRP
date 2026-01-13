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
        self.importance_scores = None

    def fit(self, device, args):

        #Imports
        import numpy as np
        import torch
        import torch.nn as nn
        import copy
        import gc

        #Helpers
        (print("Fitting"))
        # CHANGE (DeiT distilled compatibility): handle models that return (logits, distill_logits)
        def _get_logits(out):
            if isinstance(out, (tuple, list)):
                if (len(out) == 2) and hasattr(out[0], "shape") and hasattr(out[1], "shape") and (out[0].shape == out[1].shape):
                    out = (out[0] + out[1]) / 2.0
                else:
                    out = out[0]
            if hasattr(out, "logits"):
                out = out.logits
            return out

        @torch.no_grad()
        def eval_accuracy(model, device, loader, max_batches=None):
            model.eval()
            correct, total = 0, 0
            for bi, (x, y) in enumerate(loader):
                if (max_batches is not None) and (bi >= max_batches):
                    break
                x, y = x.to(device), y.to(device)
                out = _get_logits(model(x))  # CHANGE
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
                                #return out * mk.to(out.dtype).to(out.device)
                                return out * mk.to(out.dtype)
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
                    out = _get_logits(_model(x))  # CHANGE
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
            eval_cache = [] #this is in order to reduce the running time, we used to use _acc and takes too much time
            bi = 0
            for data in loader:
                if bi >= mb:
                    break
                x, y = data
                eval_cache.append((x, y))
                bi += 1
            @torch.no_grad()
            def _acc_cached(_model, _cache):
                ok = 0
                tot = 0
                i = 0
                while i < len(_cache):
                    x_cpu, y_cpu = _cache[i]
                    x = x_cpu.to(device_local, non_blocking=True)
                    y = y_cpu.to(device_local, non_blocking=True)
                    out = _get_logits(_model(x))
                    pred = out.argmax(1)
                    ok += (pred == y).sum().item()
                    tot += y.numel()
                    i += 1
                if tot == 0:
                    return 0.0
                return ok / float(tot)

            #build X/y for each layer
            out_scores = {}
            for name, mod in module_map.items():
                print("name--compute_mlp_importance_linear")
                print(name)
                #print("----------------------------")
                sampling_proba, fixed_indices, use_indices = prior_by_name[name]
                U = int(len(use_indices))
                if U == 0:
                    out_scores[name] = torch.zeros_like(mod.main_mask).view(-1)
                    continue

                X_rows = []
                y_rows = []
                # base mask only lives on use_indices
                base_mask = (mod.main_mask.squeeze().float().detach().cpu().numpy())[use_indices]
                base_mask = np.maximum(base_mask, 0)
                if base_mask.sum() == 0:
                    base_mask = np.ones_like(base_mask)
                base_mask = base_mask / base_mask.sum()
                print("we have base mask")
                k = 0
                while k < (mpi // 2):
                    print("we are inside while k < (mpi // 2) with k " + str(k))
                    # sample random mask with eps fraction turned off
                    mask = np.ones((1, 1, mod.intermediate_size), dtype=np.float32)
                    # only consider currently alive indices (main_mask>0)
                    alive = np.squeeze(mod.main_mask.detach().cpu().numpy()).nonzero()[0]
                    if alive.size == 0:
                        break
                    # choose number to zero
                    num_to_zero = int(eps * alive.size) + 1
                    # sample from alive using sampling_proba
                    probs = sampling_proba[alive]
                    if probs.sum() <= 0:
                        probs = np.ones_like(probs) / float(len(probs))
                    else:
                        probs = probs / probs.sum()
                    chosen = np.random.choice(alive, size=min(num_to_zero, alive.size), replace=False, p=probs)
                    mask[:, :, chosen] = 0.0
                    # force fixed indices alive
                    if fixed_indices is not None and len(fixed_indices) > 0:
                        mask[:, :, fixed_indices] = 1.0

                    mod.temp_mask = torch.tensor(mask, device=device_local, dtype=mod.main_mask.dtype)
                    acc1 = _acc_cached(model, eval_cache)

                    # complement
                    mask2 = np.ones((1, 1, mod.intermediate_size), dtype=np.float32)
                    mask2[:, :, alive] = 1.0
                    mask2[:, :, chosen] = 0.0
                    if fixed_indices is not None and len(fixed_indices) > 0:
                        mask2[:, :, fixed_indices] = 1.0
                    mod.temp_mask = torch.tensor(mask2, device=device_local, dtype=mod.main_mask.dtype)
                    acc2 = _acc_cached(model, eval_cache)

                    X_rows.append(torch.tensor(mask.squeeze()[use_indices], device=device_local, dtype=torch.float32))
                    y_rows.append(torch.tensor(acc1, device=device_local, dtype=torch.float32))

                    X_rows.append(torch.tensor(mask2.squeeze()[use_indices], device=device_local, dtype=torch.float32))
                    y_rows.append(torch.tensor(acc2, device=device_local, dtype=torch.float32))

                    k += 1

                if len(X_rows) == 0:
                    out_scores[name] = torch.zeros((mod.intermediate_size,), device=device_local, dtype=torch.float32)
                    continue

                X = torch.stack(X_rows, dim=0)  # [M, U]
                y = torch.stack(y_rows, dim=0)  # [M]
                w = _ridge(X, y, lam)

                # map weights back to full intermediate_size
                full = torch.zeros((mod.intermediate_size,), device=device_local, dtype=torch.float32)
                full[torch.tensor(use_indices, device=device_local)] = w
                out_scores[name] = full

                # reset temp mask
                mod.temp_mask = None
                print("full")
                print(full.detach().float().cpu().tolist(), flush=True)
                print("----------------------------------")

            print("out scores")
            print(out_scores)
            return out_scores

        #--- identify MLP modules in ViT
        def get_vit_mlp_module_map(model):
            module_map = {}
            for name, mod in model.named_modules():
                if name.endswith("mlp") and hasattr(mod, "fc1") and hasattr(mod, "fc2"):
                    module_map[name] = mod
            return module_map

        #--- build priors
        def build_mlp_priors(args_loc, model, loader, module_map):
            import numpy as np
            # (1) attach hooks to cache fc1 activations
            reattach_mlp_fc1_hooks(model)

            # (2) run warmup batches to fill intermed_cache
            wb = int(getattr(args_loc, "warmup_batches", 10))
            model.eval()
            bi = 0
            for data in loader:
                if bi >= wb:
                    break
                x, y = data
                x = x.to(device)
                _ = model(x)
                bi += 1

            # (3) build sampling proba from cached intermed_cache
            prior_by_name = {}
            for name, mod in module_map.items():
                if mod.intermed_cache is None:
                    H = mod.fc1.out_features
                    sampling_proba = np.ones((H,), dtype=np.float32) / float(H)
                    fixed_indices = np.array([], dtype=np.int64)
                    use_indices = np.arange(H, dtype=np.int64)
                else:
                    v = mod.intermed_cache.detach().cpu().squeeze().numpy()
                    if v.ndim == 0:
                        H = mod.fc1.out_features
                        sampling_proba = np.ones((H,), dtype=np.float32) / float(H)
                    else:
                        H = v.shape[0]
                        sampling_proba = v.astype(np.float32)
                        sampling_proba = sampling_proba - sampling_proba.min()
                        if sampling_proba.sum() <= 0:
                            sampling_proba = np.ones_like(sampling_proba)
                        sampling_proba = sampling_proba / sampling_proba.sum()

                    # prune_frac in this algorithm defines how many are "fixed"
                    pfrac = float(getattr(args_loc, "prune_frac", 0.05))
                    num_keep_static = int(len(sampling_proba) * (1.0 - 2 * pfrac))
                    if num_keep_static < 0:
                        num_keep_static = 0
                    sorted_ = np.argsort(-sampling_proba)
                    fixed_indices = sorted_[:num_keep_static]
                    use_indices = sorted_[num_keep_static:]

                    # invert for "turn-off probability"
                    sampling_proba = sampling_proba.max() - sampling_proba
                    sampling_proba[fixed_indices] = 0
                    if sampling_proba.sum() <= 0:
                        sampling_proba = np.ones_like(sampling_proba)
                    sampling_proba = sampling_proba / sampling_proba.sum()

                prior_by_name[name] = (sampling_proba, fixed_indices, use_indices)

            return prior_by_name


        def build_mlp_priors_dummy(module_map):
            import numpy as np
            prior_by_name = {}
            for name, mod in module_map.items():
                H = mod.fc1.out_features
                sampling_proba = np.ones((H,), dtype=np.float32) / float(H)
                fixed_indices = np.array([], dtype=np.int64)
                use_indices = np.arange(H, dtype=np.int64)
                prior_by_name[name] = (sampling_proba, fixed_indices, use_indices)
            return prior_by_name


        #--- init masks in mlp modules
        def init_main_masks(module_map):
            for name, mod in module_map.items():
                H = mod.fc1.out_features
                mod.main_mask = torch.ones((1, 1, H), device=device, dtype=torch.float16)
                mod.temp_mask = None

        # 0) baseline
        self.device = device
        self.nn = self.nn.to(device)
        self.nn.eval()

        self.base_top1 = eval_accuracy(self.nn, device, self.dl, max_batches=int(getattr(args, "eval_batches", 10)))

        # 1) Copy model
        model_copy = copy.deepcopy(self.nn).to(device)
        model_copy.eval()

        # 2) attach mlp caches and init masks
        attach_vit_mlp_caches_only(model_copy)
        module_map = get_vit_mlp_module_map(model_copy)
        init_main_masks(module_map)

        # 3) iterative pruning loop
        self.original_param_count = get_param_count(model_copy)

        target_sparsity = float(getattr(args, "sparsity_ratio", 0.2))
        tol = float(getattr(args, "tol", 0.01))
        max_iters = int(getattr(args, "max_iters", 4))
        prune_frac = float(getattr(args, "prune_frac", 0.05))

        it = 0
        while it < max_iters:

            # compute current logical sparsity: fraction turned off in mlps
            avgs = 0.0
            cnt = 0
            for name, mod in module_map.items():
                if mod.main_mask is not None:
                    avgs += (1.0 - mod.main_mask.mean().item())
                    cnt += 1
            cur_sparsity = avgs / max(1, cnt)
            self.logical_sparsity = cur_sparsity

            if (abs(cur_sparsity - target_sparsity) < tol) or (cur_sparsity > target_sparsity):
                break

            # adjust prune_frac if overshooting
            if (cur_sparsity + prune_frac) > target_sparsity:
                old_prune_frac = prune_frac
                prune_frac = abs(target_sparsity - cur_sparsity)
                # keep args in sync for prior building
                try:
                    args.prune_frac = prune_frac
                except Exception:
                    pass
                print('We have updated the prune fraction {:.3f} -> {:.3f} to avoid overshooting'.format(old_prune_frac, prune_frac))

            print('Gathering statistics for pruning')

            # build priors
            #previously we calculated priors, but since we are using this code only to extract importance scores for all the models we dont calculate priors
            #priors were meant to provide a safety net in which paramets to keep, but since to extract importances it must all go in one shot and everything needs a score, there is no sense in getting priors
            mode = getattr(args, "mode", "importances")
            if mode == "importances":
                # dummy priors: no warmup forward, no hooks, no OOM aquí
                prior_by_name = build_mlp_priors_dummy(module_map)
                print("getting dummy priors")
            else:
                prior_by_name = build_mlp_priors(args, model_copy, self.dl, module_map)

            # compute importances
            imp_by_name = compute_mlp_importance_linear(args, model_copy, self.dl, prior_by_name, module_map)
            self.importance_scores = imp_by_name
            # update masks based on prune_frac quantile
            for name, mod in module_map.items():
                imp = imp_by_name[name].detach()
                print("name--update masks")
                print(name)
                print("-------------------")
                # prevent pruning fixed indices by setting to INF
                _, fixed_indices, _ = prior_by_name[name]
                if fixed_indices is not None and len(fixed_indices) > 0:
                    imp = imp.clone()
                    imp[torch.tensor(fixed_indices, device=device)] = float('inf')

                if mod.main_mask is not None:
                    alive = (mod.main_mask.squeeze() > 0).nonzero().squeeze()
                    if alive.numel() == 0:
                        continue
                    qt = torch.quantile(imp[alive].float(), prune_frac)
                else:
                    qt = torch.quantile(imp.float(), prune_frac)

                new_mask = (imp > qt).float().half().view(1, 1, -1)

                if mod.main_mask is not None:
                    mod.main_mask *= new_mask
                else:
                    mod.main_mask = new_mask
                print("mod.main_mask")
                print(mod.main_mask)

            # clear caches
            for name, mod in module_map.items():
                mod.intermed_cache = None
                mod.temp_mask = None

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            it += 1

        # collect final masks
        self.final_masks = {name: mod.main_mask.detach().clone() for name, mod in module_map.items()}

        # store importance list
        self.mlp_importance_list = [self.final_masks[k].detach().clone().squeeze().float() for k in self.final_masks.keys()]
        print("importances!!!!")
        print(self.mlp_importance_list)

        return self.final_masks, imp_by_name #, self.importance_scores

    def compute_importances_old(self, device, args):
        import torch
        print("Computing Importances")
        self.fit(device, args)

        ffn = {}
        # preserve ordering by block index if possible
        names = list(self.final_masks.keys())
        def _idx(nm):
            try:
                return int(nm.split(".")[1])
            except Exception:
                return 999999
        names = sorted(names, key=_idx)

        for nm in names:
            mk = self.final_masks[nm].detach().squeeze().float().cpu().tolist()
            try:
                bidx = int(nm.split(".")[1])
            except Exception:
                bidx = 0
            j = 0
            while j < len(mk):
                ffn[f"{bidx}:{j}"] = float(mk[j])
                j += 1

        json_out = {"ffn": ffn}
        self.mlp_importance = json_out
        return json_out


    def compute_importances(self, device, args):
        # force "importances" mode (NO pruning)
        args_local = args
        setattr(args_local, "mode", "importances")

        masks, importance_scores = self.fit(device, args_local)

        ffn_importances = {}

        names = list(importance_scores.keys())

        def _block_idx(nm):
            try:
                return int(nm.split(".")[1])
            except:
                return 999999

        names = sorted(names, key=_block_idx)

        i = 0
        while i < len(names):
            name = names[i]
            tens = importance_scores[name]
            if tens is None:
                i += 1
                continue

            try:
                bidx = int(name.split(".")[1])
            except:
                bidx = i

            flat = tens.view(-1).detach().cpu().tolist()

            j = 0
            while j < len(flat):
                key_ij = f"{bidx}:{j}"
                ffn_importances[key_ij] = float(flat[j])
                j += 1

            i += 1

        json_out = {"ffn": ffn_importances}
        self.mlp_importance = json_out
        return json_out

