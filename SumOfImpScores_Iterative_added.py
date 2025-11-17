import torch
import torch.nn as nn
from time import time
from abc import ABC, abstractmethod
from Iterative_Interface_MLP import PruningInterface as Iterat_MLP_PruningInterface
from Iterative_Interface_ATTN import PruningInterface as Iterat_ATTN_PruningInterface



# Model Definitions

class Attn:
    def __init__(self, emb_dim, head_dim, n_heads):
        # Attention weights: output_dim x input_dim
        self.q = torch.randn((n_heads * head_dim, emb_dim))
        self.k = torch.randn((n_heads * head_dim, emb_dim))
        self.v = torch.randn((n_heads * head_dim, emb_dim))
        self.p = torch.randn((emb_dim, n_heads * head_dim))  # proj is transposed

        # MLP weights
        self.fc1 = torch.randn((4 * emb_dim, emb_dim))  # hidden x input
        self.fc2 = torch.randn((emb_dim, 4 * emb_dim))  # output x hidden


class VisionModel:
    def __init__(self, emb_dim, head_dim, n_heads):
        self.de = emb_dim
        self.dh = head_dim
        self.nh = n_heads
        self.nb = 12

        self.bs = []
        for _ in range(self.nb):
            self.bs.append(Attn(emb_dim, head_dim, n_heads))

    def count_parameters(self):
        # Count total parameters in model
        total = 0
        for block in self.bs:
            total += block.q.numel() + block.k.numel() + block.v.numel() + block.p.numel()
            total += block.fc1.numel() + block.fc2.numel()
        return total

    def count_nonzero_parameters(self):
        # Count non-pruned parameters - to calculate sparsity
        nonzero = 0
        for block in self.bs:
            nonzero += (block.q != 0).sum().item()
            nonzero += (block.k != 0).sum().item()
            nonzero += (block.v != 0).sum().item()
            nonzero += (block.p != 0).sum().item()
            nonzero += (block.fc1 != 0).sum().item()
            nonzero += (block.fc2 != 0).sum().item()
        return nonzero


# PRUNING METHOD INTERFACE

class PruningMethod(ABC):
    """
    UNIFIED INTERFACE for plugging in ANY pruning method

    To plug in your own method :
    1. Inherit from this class
    2. Implement compute_importance_scores()
    3. Set self.mlp_neuron_importance and self.att_importance

    SCORE FORMAT:

    For MLP (ALWAYS neuron-level):
    - mlp_neuron_importance: List[Tensor] of shape (hidden_dim,) per block
      One score per hidden unit in fc1/fc2

    For Attention (depends on pruning_granularity):
    - If 'width': List[Tensor] of shape (n_heads*head_dim,) per block
    - If 'depth': Tensor of shape (n_blocks,)
    - If 'head': List[Tensor] of shape (n_heads,) per block

    All scores MUST be normalized to [0, 1] where 1 = most important
    """

    def __init__(self, model, attention_granularity='width', dataloader=None):
        """
        Args:
            model: VisionModel instance
            attention_granularity: How to prune attention - 'width', 'depth', or 'head'
            dataloader: Optional dataloader for methods that need data
        """
        self.model = model
        self.attention_granularity = attention_granularity.lower()
        self.dataloader = dataloader

        if self.attention_granularity not in ['width', 'depth', 'head']:
            raise ValueError(f"Invalid attention_granularity: {attention_granularity}")

        # Importance scores (set by compute_importance_scores)
        self.mlp_neuron_importance = None  # ALWAYS neuron-level for MLP
        self.att_importance = None  # Granularity depends on attention_granularity

    @abstractmethod
    def compute_importance_scores(self):
        """
        Compute importance scores using YOUR algorithm

        MUST set:
        - self.mlp_neuron_importance: ALWAYS neuron-level scores for MLP
          List of tensors, shape (4*emb_dim,) per block

        - self.att_importance: Attention scores at specified granularity
          Format depends on self.attention_granularity:
          - 'width': List of tensors, shape (n_heads*head_dim,) per block
          - 'depth': Tensor of shape (n_blocks,)
          - 'head': List of tensors, shape (n_heads,) per block

        All scores MUST be in range [0, 1]

        Example for a gradient-based method:
            # MLP neuron importance (ALWAYS compute this)
            for block in self.model.bs:
                mlp_score = compute_gradient_importance(block.fc1, block.fc2)
                self.mlp_neuron_importance.append(mlp_score)

            # Attention importance (at chosen granularity)
            if self.attention_granularity == 'width':
                for block in self.model.bs:
                    att_score = compute_gradient_importance(block.q, block.k, ...)
                    self.att_importance.append(att_score)
        """
        pass

    def get_mlp_importance(self):
        if self.mlp_neuron_importance is None:
            self.compute_importance_scores()
        return self.mlp_neuron_importance

    def get_att_importance(self):
        if self.att_importance is None:
            self.compute_importance_scores()
        return self.att_importance


# EXAMPLE IMPLEMENTATIONS

class Method_WidthWidth(PruningMethod):
    """
    Example: Prunes both MLP and Attention at neuron-level (width)

    YOUR METHOD HERE: Replace with 2SSP, SNP, HAS implementation
    This shows the pattern for width-width pruning
    """

    def __init__(self, model, dataloader=None, criterion='l1'):
        super().__init__(model, attention_granularity='width', dataloader=dataloader)
        self.criterion = criterion  # 'l1', 'l2', 'gradient', etc.

    def compute_importance_scores(self):
        """Compute neuron importance for BOTH MLP and Attention"""
        self.mlp_neuron_importance = []
        self.att_importance = []

        for block in self.model.bs:
            # MLP neuron importance (REQUIRED for all methods)
            if self.criterion == 'l1':
                mlp_score = torch.sum(torch.abs(block.fc1), dim=1)
                mlp_score += torch.sum(torch.abs(block.fc2), dim=0)
            elif self.criterion == 'l2':
                mlp_score = torch.sum(block.fc1 ** 2, dim=1)
                mlp_score += torch.sum(block.fc2 ** 2, dim=0)
            else:  # gradient-based, variance, etc.
                mlp_score = torch.var(block.fc1, dim=1) + torch.var(block.fc2, dim=0)

            mlp_score = mlp_score / (mlp_score.max() + 1e-8)
            self.mlp_neuron_importance.append(mlp_score)

            # Attention neuron importance
            if self.criterion == 'l1':
                att_score = torch.sum(torch.abs(block.q), dim=1)
                att_score += torch.sum(torch.abs(block.k), dim=1)
                att_score += torch.sum(torch.abs(block.v), dim=1)
                att_score += torch.sum(torch.abs(block.p), dim=0)
            elif self.criterion == 'l2':
                att_score = torch.sum(block.q ** 2, dim=1)
                att_score += torch.sum(block.k ** 2, dim=1)
                att_score += torch.sum(block.v ** 2, dim=1)
                att_score += torch.sum(block.p ** 2, dim=0)
            else:
                att_score = torch.var(block.q, dim=1) + torch.var(block.k, dim=1)
                att_score += torch.var(block.v, dim=1) + torch.var(block.p, dim=0)

            att_score = att_score / (att_score.max() + 1e-8)
            self.att_importance.append(att_score)


class Method_DepthWidth(PruningMethod):
    """
    Example: Prunes MLP at neuron-level, Attention at block-level

    YOUR METHOD HERE: Replace with 2SSP, SNP, HAS implementation
    This shows the pattern for depth-width pruning
    """

    def __init__(self, model, dataloader=None, criterion='magnitude'):
        super().__init__(model, attention_granularity='depth', dataloader=dataloader)
        self.criterion = criterion

    def compute_importance_scores(self):
        """Compute MLP neuron importance + Attention block importance"""
        self.mlp_neuron_importance = []

        # MLP: Neuron-level (REQUIRED)
        for block in self.model.bs:
            if self.criterion == 'magnitude':
                mlp_score = torch.sum(torch.abs(block.fc1), dim=1)
                mlp_score += torch.sum(torch.abs(block.fc2), dim=0)
            else:  # variance, gradient, etc.
                mlp_score = torch.var(block.fc1, dim=1) + torch.var(block.fc2, dim=0)

            mlp_score = mlp_score / (mlp_score.max() + 1e-8)
            self.mlp_neuron_importance.append(mlp_score)

        # Attention: Block-level
        att_block_scores = []
        for block in self.model.bs:
            if self.criterion == 'magnitude':
                block_score = torch.abs(block.q).sum() + torch.abs(block.k).sum()
                block_score += torch.abs(block.v).sum() + torch.abs(block.p).sum()
            else:
                block_score = torch.var(block.q) + torch.var(block.k)
                block_score += torch.var(block.v) + torch.var(block.p)
            att_block_scores.append(block_score.item())

        self.att_importance = torch.tensor(att_block_scores)
        self.att_importance = self.att_importance / (self.att_importance.max() + 1e-8)


class Method_HeadWidth(PruningMethod):
    """
    Example: Prunes MLP at neuron-level, Attention at head-level

    YOUR METHOD HERE: Replace with 2SSP, SNP, HAS implementation
    This shows the pattern for head-width pruning
    """

    def __init__(self, model, dataloader=None, criterion='magnitude'):
        super().__init__(model, attention_granularity='head', dataloader=dataloader)
        self.criterion = criterion

    def compute_importance_scores(self):
        """Compute MLP neuron importance + Attention head importance"""
        self.mlp_neuron_importance = []
        self.att_importance = []

        # MLP: Neuron-level (REQUIRED)
        for block in self.model.bs:
            if self.criterion == 'magnitude':
                mlp_score = torch.sum(torch.abs(block.fc1), dim=1)
                mlp_score += torch.sum(torch.abs(block.fc2), dim=0)
            else:
                mlp_score = torch.var(block.fc1, dim=1) + torch.var(block.fc2, dim=0)

            mlp_score = mlp_score / (mlp_score.max() + 1e-8)
            self.mlp_neuron_importance.append(mlp_score)

        # Attention: Head-level
        nh = self.model.nh
        dh = self.model.dh

        for block in self.model.bs:
            head_scores = torch.zeros(nh)

            for h in range(nh):
                start = h * dh
                end = (h + 1) * dh

                if self.criterion == 'magnitude':
                    score = torch.abs(block.q[start:end]).sum()
                    score += torch.abs(block.k[start:end]).sum()
                    score += torch.abs(block.v[start:end]).sum()
                    score += torch.abs(block.p[:, start:end]).sum()
                else:
                    score = torch.var(block.q[start:end])
                    score += torch.var(block.k[start:end])
                    score += torch.var(block.v[start:end])
                    score += torch.var(block.p[:, start:end])

                head_scores[h] = score.item()

            head_scores = head_scores / (head_scores.max() + 1e-8)
            self.att_importance.append(head_scores)




class Method_Bonsai_Iterative_Interface_MLP(PruningMethod):

    def __init__(self,model,real_model,real_loader,device,  args, attention_granularity='width',dataloader=None):
        super().__init__(model, attention_granularity=attention_granularity,dataloader=dataloader)
        self.real_model = real_model
        self.real_loader = real_loader
        self.device = device
        self.pruning_args = args
        self._done = False
        #IMPORTANT INFO ABOUT ARGS: 
        #SimpleNamespace object
        #to get importances for all neurons, set prune_frac = 1.0
        #prune_frac, sparsity_ratio, tol, max_iters,
        #masks_per_iter, eval_batches, warmup_batches,
        #perturb_frac, ridge_lambda, mode='importances' (if we dont want to affect the model but rather just get importances)
    def compute_importance_scores(self):
        if self._done:
            return
        from Iterative_Interface_MLP import PruningInterface as Iterat_MLP_PruningInterface
        import torch
        #my pruning interface
        iface = Iterat_MLP_PruningInterface(self.real_model, self.real_loader)
        mlp_importance = iface.compute_importances(self.device, self.pruning_args)
        self.mlp_neuron_importance = [row.detach().float().view(-1) for row in mlp_importance]

        #not using attention so just None
        self.att_importance = None
        self._done = True



class Method_Bonsai_Iterative_Interface_Attn(PruningMethod):

    def __init__(self,model,real_model,real_loader,device,args, attention_granularity='head',dataloader=None):
        super().__init__(model, attention_granularity=attention_granularity, dataloader=dataloader)
        self.real_model = real_model
        self.real_loader = real_loader
        self.device = device
        self.pruning_args = args
        self._done = False

    def compute_importance_scores(self):
        if self._done:
            return
        from Iterative_Interface_ATTN import PruningInterface as Iterat_ATTN_PruningInterface
        import torch

        iface = Iterat_ATTN_PruningInterface(self.real_model, self.real_loader)
        att_importance = iface.compute_importances(self.device, self.pruning_args)
        self.att_importance = [row.detach().float().view(-1) for row in att_importance]

        # not using MLP here, so no MLP scores
        self.mlp_neuron_importance = None
        self._done = True


















# SUM OF IMPORTANCE SCORES PRUNING

class SumOfImportanceScoresPruning:
    """
    Combines multiple pruning methods by summing normalized importance scores

    KEY ARCHITECTURE:
    - ALL methods compute MLP neuron importance (their own way)
    - Each method computes Attention importance at its chosen granularity
    - Scores are summed, then used to create structured masks
    """

    def __init__(self, model, pruning_methods):
        """
        Args:
            model: VisionModel instance
            pruning_methods: List of PruningMethod instances
        """
        self.model = model
        self.methods = pruning_methods

        # Compute all scores
        for method in self.methods:
            method.compute_importance_scores()

    def _expand_att_scores_to_neurons(self, method, block_idx):
        """
        Expand attention importance scores to per-neuron format

        Returns:
            Tensor of shape (n_heads*head_dim,) - neuron-level scores
        """
        if method.attention_granularity == 'width':
            # Already in neuron format
            return method.get_att_importance()[block_idx]

        elif method.attention_granularity == 'depth':
            # Expand block-level score to all neurons
            block_score = method.get_att_importance()[block_idx].item()
            return torch.full((self.model.nh * self.model.dh,), block_score)

        elif method.attention_granularity == 'head':
            # Expand head-level scores to neurons
            head_scores = method.get_att_importance()[block_idx]
            neuron_scores = torch.zeros(self.model.nh * self.model.dh)
            for h in range(self.model.nh):
                start = h * self.model.dh
                end = (h + 1) * self.model.dh
                neuron_scores[start:end] = head_scores[h]
            return neuron_scores

    def combine_scores(self):
        """
        Sum importance scores from all methods

        Returns:
            combined_mlp_scores: List of tensors, shape (hidden_dim,) per block
            combined_att_scores: List of tensors, shape (n_heads*head_dim,) per block
        """
        nb = self.model.nb
        mlp_hidden_dim = 4 * self.model.de
        att_hidden_dim = self.model.nh * self.model.dh

        combined_mlp_scores = [torch.zeros(mlp_hidden_dim) for _ in range(nb)]
        combined_att_scores = [torch.zeros(att_hidden_dim) for _ in range(nb)]

        # Sum MLP scores from ALL methods (each computes its own way)
        for method in self.methods:
            mlp_importance = method.get_mlp_importance()
            for block_idx in range(nb):
                combined_mlp_scores[block_idx] += mlp_importance[block_idx]

        # Sum Attention scores from all methods (expand to neuron-level first)
        for method in self.methods:
            for block_idx in range(nb):
                att_neuron_scores = self._expand_att_scores_to_neurons(method, block_idx)
                combined_att_scores[block_idx] += att_neuron_scores

        return combined_mlp_scores, combined_att_scores

    def create_masks(self, target_sparsity):
        """
        Create structured pruning masks

        Args:
            target_sparsity: Fraction of parameters to prune (0 to 1)

        Returns:
            mlp_masks: List[List[Tensor]] - [fc1, fc2] masks per block
            att_masks: List[List[Tensor]] - [q, k, v, p] masks per block
        """
        combined_mlp, combined_att = self.combine_scores()

        # Create masks
        mlp_masks = self._create_mlp_masks(combined_mlp, target_sparsity)
        att_masks = self._create_att_masks(combined_att, target_sparsity)

        return mlp_masks, att_masks

    def _create_mlp_masks(self, combined_scores, target_sparsity):
        """Create neuron-level masks for MLP"""
        all_scores = torch.cat(combined_scores)
        sorted_scores, _ = torch.sort(all_scores)
        threshold_idx = int(target_sparsity * len(sorted_scores))
        threshold = sorted_scores[threshold_idx].item()

        masks = []
        for i, block in enumerate(self.model.bs):
            neuron_mask = (combined_scores[i] > threshold).float()

            fc1_mask = neuron_mask.unsqueeze(1).expand_as(block.fc1)
            fc2_mask = neuron_mask.unsqueeze(0).expand_as(block.fc2)

            masks.append([fc1_mask, fc2_mask])

        return masks

    def _create_att_masks(self, combined_scores, target_sparsity):
        """Create neuron-level masks for attention"""
        all_scores = torch.cat(combined_scores)
        sorted_scores, _ = torch.sort(all_scores)
        threshold_idx = int(target_sparsity * len(sorted_scores))
        threshold = sorted_scores[threshold_idx].item()

        masks = []
        for i, block in enumerate(self.model.bs):
            neuron_mask = (combined_scores[i] > threshold).float()

            q_mask = neuron_mask.unsqueeze(1).expand_as(block.q)
            k_mask = neuron_mask.unsqueeze(1).expand_as(block.k)
            v_mask = neuron_mask.unsqueeze(1).expand_as(block.v)
            p_mask = neuron_mask.unsqueeze(0).expand_as(block.p)

            masks.append([q_mask, k_mask, v_mask, p_mask])

        return masks

    def apply_masks(self, mlp_masks, att_masks):
        """Apply pruning masks to model"""
        for i, block in enumerate(self.model.bs):
            block.fc1 = block.fc1 * mlp_masks[i][0]
            block.fc2 = block.fc2 * mlp_masks[i][1]
            block.q = block.q * att_masks[i][0]
            block.k = block.k * att_masks[i][1]
            block.v = block.v * att_masks[i][2]
            block.p = block.p * att_masks[i][3]

    def get_sparsity_stats(self, mlp_masks, att_masks):
        """Calculate achieved sparsity"""
        mlp_total = sum(m[0].numel() + m[1].numel() for m in mlp_masks)
        mlp_pruned = sum((m[0] == 0).sum() + (m[1] == 0).sum() for m in mlp_masks)

        att_total = sum(sum(m.numel() for m in block) for block in att_masks)
        att_pruned = sum(sum((m == 0).sum() for m in block) for block in att_masks)

        mlp_sparsity = mlp_pruned.item() / mlp_total
        att_sparsity = att_pruned.item() / att_total
        total_sparsity = (mlp_pruned + att_pruned).item() / (mlp_total + att_total)

        return mlp_sparsity, att_sparsity, total_sparsity


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_model(model, mlp_masks, att_masks):
    """
    Evaluate pruned model performance

    Returns:
        accuracy: Simulated accuracy (in real use, run on validation set)
        latency: Estimated inference latency reduction
        params: Parameter statistics
    """
    # Apply masks
    pruner = SumOfImportanceScoresPruning(model, [])
    pruner.apply_masks(mlp_masks, att_masks)

    # Calculate parameter reduction
    total_params = model.count_parameters()
    active_params = model.count_nonzero_parameters()
    param_reduction = 1 - (active_params / total_params)

    # Simulate accuracy (in real scenario, run inference on validation set)
    # Accuracy typically degrades with pruning
    baseline_acc = 0.85
    simulated_acc = baseline_acc * (1 - 0.3 * param_reduction)  # Rough approximation

    # Estimate latency reduction (roughly proportional to FLOPs reduction)
    # For structured pruning, latency ≈ 1 - sparsity (assuming full neuron removal)
    latency_reduction = param_reduction * 0.8  # 80% of param reduction reflects in latency

    return {
        'accuracy': simulated_acc,
        'latency_reduction': latency_reduction,
        'total_params': total_params,
        'active_params': active_params,
        'sparsity': param_reduction
    }


# EXPERIMENTS

def run_experiment(emb_dim, head_dim, num_heads, target_sparsity, methods_config, name):
    """
    Run pruning experiment with given method configuration

    Args:
        methods_config: List of tuples (MethodClass, attention_granularity, kwargs)
        name: Experiment name
    """
    print(f"\n{'=' * 80}")
    print(f"Experiment: {name}")
    print(f"Target Sparsity: {target_sparsity * 100:.1f}%")
    print(f"{'=' * 80}")

    # Create model
    model = VisionModel(emb_dim, head_dim, num_heads)

    # Initialize methods
    methods = []
    for MethodClass, att_gran, kwargs in methods_config:
        methods.append(MethodClass(model, attention_granularity=att_gran, **kwargs))

    # Create pruner
    start = time()
    pruner = SumOfImportanceScoresPruning(model, methods)
    mlp_masks, att_masks = pruner.create_masks(target_sparsity)
    pruning_time = time() - start

    # Get sparsity stats
    mlp_sp, att_sp, total_sp = pruner.get_sparsity_stats(mlp_masks, att_masks)

    # Evaluate
    metrics = evaluate_model(model, mlp_masks, att_masks)

    print(f"Pruning Time: {pruning_time:.3f}s")
    print(f"\nSparsity Results:")
    print(f"  Target:    {target_sparsity * 100:5.1f}%")
    print(f"  Achieved:  {total_sp * 100:5.1f}%")
    print(f"  MLP:       {mlp_sp * 100:5.1f}%")
    print(f"  Attention: {att_sp * 100:5.1f}%")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Latency Reduction:  {metrics['latency_reduction'] * 100:.1f}%")
    print(f"  Active Parameters:  {metrics['active_params']:,} / {metrics['total_params']:,}")

    return {
        'name': name,
        'target': target_sparsity,
        'mlp_sp': mlp_sp,
        'att_sp': att_sp,
        'total_sp': total_sp,
        'accuracy': metrics['accuracy'],
        'latency_reduction': metrics['latency_reduction'],
        'pruning_time': pruning_time
    }


def run_all_experiments():
    """Run comprehensive evaluation of different method combinations"""

    emb_dim, head_dim, num_heads = 768, 64, 12
    target_sparsity = 0.5  # 50% pruning

    results = []

    print("\n" + "=" * 90)
    print(" " * 25 + "PRUNING METHOD EVALUATION")
    print("=" * 90)

    # Experiment 1: Single method (Width-Width, L1)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [(Method_WidthWidth, 'width', {'criterion': 'l1'})],
        "Single Method: Width-Width (L1)"
    ))

    # Experiment 2: Single method (Width-Width, L2)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [(Method_WidthWidth, 'width', {'criterion': 'l2'})],
        "Single Method: Width-Width (L2)"
    ))

    # Experiment 3: Single method (Depth-Width)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [(Method_DepthWidth, 'depth', {'criterion': 'magnitude'})],
        "Single Method: Depth-Width"
    ))

    # Experiment 4: Single method (Head-Width)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [(Method_HeadWidth, 'head', {'criterion': 'magnitude'})],
        "Single Method: Head-Width"
    ))

    # Experiment 5: Two methods (Width-Width L1 + Depth-Width)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [
            (Method_WidthWidth, 'width', {'criterion': 'l1'}),
            (Method_DepthWidth, 'depth', {'criterion': 'magnitude'})
        ],
        "Two Methods: Width-Width + Depth-Width"
    ))

    # Experiment 6: Two methods (Width-Width L1 + Head-Width)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [
            (Method_WidthWidth, 'width', {'criterion': 'l1'}),
            (Method_HeadWidth, 'head', {'criterion': 'magnitude'})
        ],
        "Two Methods: Width-Width + Head-Width"
    ))

    # Experiment 7: Three methods (All combinations)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [
            (Method_WidthWidth, 'width', {'criterion': 'l1'}),
            (Method_DepthWidth, 'depth', {'criterion': 'magnitude'}),
            (Method_HeadWidth, 'head', {'criterion': 'magnitude'})
        ],
        "Three Methods: Width+Depth+Head"
    ))

    # Experiment 8: Multiple Width methods (ensemble)
    results.append(run_experiment(
        emb_dim, head_dim, num_heads, target_sparsity,
        [
            (Method_WidthWidth, 'width', {'criterion': 'l1'}),
            (Method_WidthWidth, 'width', {'criterion': 'l2'}),
            (Method_WidthWidth, 'width', {'criterion': 'variance'})
        ],
        "Width Ensemble (L1+L2+Var)"
    ))

    # Print summary table
    print("\n" + "=" * 90)
    print(" " * 30 + "RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Method':<35} {'Acc':<8} {'Latency↓':<10} {'Sparsity':<10} {'Time(s)':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<35} {r['accuracy']:.4f}   "
              f"{r['latency_reduction'] * 100:5.1f}%     "
              f"{r['total_sp'] * 100:5.1f}%      "
              f"{r['pruning_time']:.3f}")
    print("=" * 90)

    return results

    # Print summary table
    print("\n" + "=" * 90)
    print(" " * 30 + "RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Method':<30} {'Acc':<8} {'Latency↓':<10} {'Sparsity':<10} {'Time(s)':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} {r['accuracy']:.4f}   "
              f"{r['latency_reduction'] * 100:5.1f}%     "
              f"{r['total_sp'] * 100:5.1f}%      "
              f"{r['pruning_time']:.3f}")
    print("=" * 90)

    return results

if __name__ == "__main__":
    results = run_all_experiments()