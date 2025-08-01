from dataclasses import dataclass, field
from sklearn.decomposition import PCA
import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half
import torch.nn.functional as F
from kvpress.presses.base_press import BasePress
import json


@dataclass
class QuarKPress(BasePress):

    key_channel_compression_ratio: float = 0.0
    value_channel_compression_ratio: float = 0.0
    window_size: int = 32
    max_capacity_prompt: int = field(init=False, default=None)
    threshold_ratio: float = 0.0
    pooling_ratio: float = 0.0
    mode: str = field(init=False, default=None)
    outpath: str = field(init=False, default=None)

    def compute_window_queries(self, module, hidden_states, position_embeddings):
        """
        Re-compute the last window_size query states
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -self.window_size :])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -self.window_size :])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, self.window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -self.window_size :], sin[:, -self.window_size :]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        return query_states

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If other similar presses are requested, we might create a generic compress method for dimension pruning
        to avoid code duplication.
        """
        if self.key_channel_compression_ratio == 0 and self.threshold_ratio == 0 and self.pooling_ratio == 0:
            print('error !!!!*****')
            return keys, values

        # Compute scores per dimension
        bsz, num_key_value_heads, q_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        queries = self.compute_window_queries(module, kwargs["hidden_states"], kwargs["position_embeddings"])
        queries_expand = queries.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size,module.head_dim).mean(2)
        pruned_keys = dynamic_score_selection_norm(queries_expand, keys, self.threshold_ratio, self.key_channel_compression_ratio, self.pooling_ratio)

        if self.value_channel_compression_ratio > 0:
            # Value compression using different strategy than keys
            pruned_values = compress_values(
                values, 
                queries_expand, 
                self.value_channel_compression_ratio,
                self.threshold_ratio,
                self.pooling_ratio
            )
        else:
            pruned_values = values

        return pruned_keys, pruned_values

    @property
    def compression_ratio(self):
        return self.key_channel_compression_ratio / 2

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

def compress_values(values, queries, compression_ratio, threshold_ratio=0, pooling_ratio=0):
    """
    Value compression with different strategy than keys.
    Values carry the actual information content, so we use different importance metrics.
    Each token's each dimension is evaluated independently.
    
    Args:
        values: (bsz, num_heads, seq_len, head_dim)
        queries: (bsz, num_heads, window_size, head_dim) 
        compression_ratio: ratio of dimensions to compress
        threshold_ratio: threshold for dynamic selection
        pooling_ratio: pooling strategy parameter
    """
    bsz, num_heads, seq_len, head_dim = values.shape
    
    value_magnitudes = torch.abs(values)  # (bsz, num_heads, seq_len, head_dim)
    importance_scores = value_magnitudes

    value_variance_across_seq = torch.var(values, dim=2, keepdim=True)  # (bsz, num_heads, 1, head_dim)
    value_variance_across_seq = value_variance_across_seq.expand(-1, -1, seq_len, -1)  # (bsz, num_heads, seq_len, head_dim)
    
    # Calculate variance across dimensions for each token
    value_variance_across_dim = torch.var(values, dim=3, keepdim=True)  # (bsz, num_heads, seq_len, 1)
    value_variance_across_dim = value_variance_across_dim.expand(-1, -1, -1, head_dim)  # (bsz, num_heads, seq_len, head_dim)
    
    # Combine both variance measures
    diversity_scores = (value_variance_across_seq + value_variance_across_dim) / 2
    diversity_scores = diversity_scores / (diversity_scores.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
    
    # Final importance combines multiple factors (maintains seq_len and head_dim)
    final_scores = importance_scores * 0.7 + diversity_scores * 0.3  # (bsz, num_heads, seq_len, head_dim)
    
    # Create mask based on importance scores (element-wise)
    # We can either use global threshold or per-token/per-head threshold
    if threshold_ratio > 0:
        # Dynamic threshold-based selection (element-wise)
        # Flatten for sorting while keeping track of original positions
        flat_scores = final_scores.view(bsz, num_heads, -1)  # (bsz, num_heads, seq_len * head_dim)
        sorted_scores, sorted_indices = torch.sort(flat_scores, dim=-1, descending=True)
        
        cumulative_scores = torch.cumsum(sorted_scores, dim=-1)
        total_scores = cumulative_scores[..., -1:]
        is_above_threshold = cumulative_scores > (threshold_ratio * total_scores)
        threshold_indices = is_above_threshold.to(torch.float32).argmax(dim=-1)  # (bsz, num_heads)
        
        # Create element-wise mask
        mask = torch.zeros_like(values, dtype=torch.bool)
        flat_mask = mask.view(bsz, num_heads, -1)
        
        for b in range(bsz):
            for h in range(num_heads):
                n_keep = max(1, threshold_indices[b, h].item() + 1)
                keep_indices = sorted_indices[b, h, :n_keep]
                flat_mask[b, h, keep_indices] = True
                
        mask = flat_mask.view(bsz, num_heads, seq_len, head_dim)
    else:
        # Fixed ratio selection (element-wise)
        _, sorted_indices = torch.sort(final_scores, dim=-1, descending=True)
        mask = torch.ones_like(values, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices[..., -int(compression_ratio * head_dim):], False)

    
    # Apply different reconstruction strategies based on pooling_ratio
    if pooling_ratio == 0.5:
        # PCA-based reconstruction for values
        return generate_value_pca_fill(values, mask)
    elif pooling_ratio == 0.3:
        # Interpolation-based reconstruction
        return generate_value_interpolated_fill(values, mask)
    elif pooling_ratio == 0.7:
        # Statistical distribution-based fill
        return generate_value_statistical_fill(values, mask, 'normal')
    elif pooling_ratio == 0.6:
        # Statistical distribution-based fill  
        return generate_value_statistical_fill(values, mask, 'exponential')
    elif pooling_ratio == 0.65:
        # Mean-based reconstruction for values
        return generate_value_mean_fill(values, mask)
    else:
        # Simple masking
        return values * mask

def generate_value_pca_fill(values, mask, n_components=8):
    """
    PCA-based reconstruction for value cache.
    Values need more careful reconstruction as they directly affect output.
    """
    recovered_values = values.clone()
    bsz, num_heads, seq_len, head_dim = values.shape
    
    for b in range(bsz):
        for h in range(num_heads):
            # Get valid (non-masked) dimensions across all sequence positions
            head_values = values[b, h]  # (seq_len, head_dim)
            head_mask = mask[b, h]      # (seq_len, head_dim)
            
            # Find dimensions that have valid data across most tokens
            valid_dim_ratio = head_mask.float().mean(dim=0)  # (head_dim,)
            reliable_dims = valid_dim_ratio > 0.5  # Dimensions valid in >50% of tokens
            
            if reliable_dims.sum() > n_components:
                reliable_data = head_values[:, reliable_dims]  # (seq_len, n_reliable)
                
                try:
                    # Apply PCA to reliable dimensions
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_components, reliable_data.shape[1]))
                    
                    # Fit PCA on all sequence positions
                    reliable_data_np = reliable_data.detach().cpu().numpy()
                    pca.fit(reliable_data_np)
                    
                    # Transform and inverse transform to get reconstructed values
                    transformed = pca.transform(reliable_data_np)
                    reconstructed = pca.inverse_transform(transformed)
                    reconstructed_tensor = torch.tensor(reconstructed, device=values.device, dtype=values.dtype)
                    
                    # Fill in the reliable dimensions
                    recovered_values[b, h, :, reliable_dims] = reconstructed_tensor
                    
                except Exception as e:
                    print(f"PCA failed for batch {b}, head {h}: {e}")
                    # Fallback to mean filling
                    mean_values = head_values[head_mask].mean()
                    recovered_values[b, h][~head_mask] = mean_values
            else:
                # Not enough reliable dimensions, use simple mean filling
                mean_values = head_values[head_mask].mean()
                recovered_values[b, h][~head_mask] = mean_values
    
    return recovered_values

def generate_value_interpolated_fill(values, mask):
    """
    Interpolation-based filling for value cache using tensor operations.
    Uses temporal and dimensional correlations efficiently.
    """
    bsz, num_heads, seq_len, head_dim = values.shape
    recovered_values = values.clone()
    missing_mask = ~mask
    
    # Method 1: Dimension-wise filling (vectorized across all dimensions)
    # Calculate mean for each dimension across all valid positions
    dim_valid_counts = mask.sum(dim=2, keepdim=True)  # (bsz, num_heads, 1, head_dim)
    dim_means = (values * mask).sum(dim=2, keepdim=True) / (dim_valid_counts + 1e-8)  # (bsz, num_heads, 1, head_dim)
    dim_fill = dim_means.expand(-1, -1, seq_len, -1)  # (bsz, num_heads, seq_len, head_dim)
    
    # Method 2: Token-wise filling (vectorized across all tokens)
    # Calculate mean for each token using valid dimensions
    token_valid_counts = mask.sum(dim=3, keepdim=True)  # (bsz, num_heads, seq_len, 1)
    token_means = (values * mask).sum(dim=3, keepdim=True) / (token_valid_counts + 1e-8)  # (bsz, num_heads, seq_len, 1)
    token_fill = token_means.expand(-1, -1, -1, head_dim)  # (bsz, num_heads, seq_len, head_dim)
    
    # Method 3: Spatial interpolation (use neighboring tokens)
    # Create shifted versions for temporal interpolation
    prev_values = torch.roll(values, shifts=1, dims=2)  # Previous token values
    next_values = torch.roll(values, shifts=-1, dims=2)  # Next token values
    prev_mask = torch.roll(mask, shifts=1, dims=2)      # Previous token mask
    next_mask = torch.roll(mask, shifts=-1, dims=2)     # Next token mask
    
    # Handle boundary conditions
    prev_values[:, :, 0, :] = 0
    next_values[:, :, -1, :] = 0
    prev_mask[:, :, 0, :] = False
    next_mask[:, :, -1, :] = False
    
    # Calculate interpolated values where both neighbors are valid
    both_valid = prev_mask & next_mask
    spatial_fill = torch.where(both_valid, 
                              (prev_values + next_values) / 2, 
                              torch.where(prev_mask, prev_values,
                                        torch.where(next_mask, next_values, values)))
    
    # Hierarchical filling strategy using tensor operations
    # Start with dimension-wise means
    fill_values = dim_fill.clone()
    
    # Override with token-wise means where token has valid data
    has_valid_token_data = token_valid_counts.squeeze(-1) > 0  # (bsz, num_heads, seq_len)
    has_valid_token_expanded = has_valid_token_data.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    fill_values = torch.where(has_valid_token_expanded, token_fill, fill_values)
    
    # Override with spatial interpolation where available
    has_spatial_data = (prev_mask | next_mask)
    fill_values = torch.where(has_spatial_data, spatial_fill, fill_values)
    
    # Apply filling only to missing positions
    recovered_values = torch.where(missing_mask, fill_values, recovered_values)
    
    return recovered_values

def generate_value_statistical_fill(values, mask, distribution='normal'):
    """
    Statistical distribution-based filling for value cache using tensor operations.
    """
    bsz, num_heads, seq_len, head_dim = values.shape
    recovered_values = values.clone()
    missing_mask = ~mask
    
    # Calculate statistics per head using tensor operations
    valid_values_masked = values * mask  # Zero out invalid positions
    valid_counts = mask.sum(dim=(2, 3), keepdim=True)  # (bsz, num_heads, 1, 1)
    
    # Calculate mean and other statistics per head
    head_means = valid_values_masked.sum(dim=(2, 3), keepdim=True) / (valid_counts + 1e-8)  # (bsz, num_heads, 1, 1)
    
    if distribution == 'normal':
        # Calculate variance per head
        squared_diff = (valid_values_masked - head_means.expand_as(values)) ** 2 * mask
        head_vars = squared_diff.sum(dim=(2, 3), keepdim=True) / (valid_counts + 1e-8)
        head_stds = torch.sqrt(head_vars).clamp(min=1e-6)
        
        # Expand to full shape for broadcasting
        means_expanded = head_means.expand(bsz, num_heads, seq_len, head_dim)
        stds_expanded = head_stds.expand(bsz, num_heads, seq_len, head_dim)
        
        # Generate normal distributed values for all missing positions
        normal_dist = torch.distributions.Normal(means_expanded, stds_expanded)
        fill_values = normal_dist.sample()
        
    elif distribution == 'exponential':
        # Calculate absolute mean for exponential distribution
        abs_valid = torch.abs(valid_values_masked)
        abs_means = abs_valid.sum(dim=(2, 3), keepdim=True) / (valid_counts + 1e-8)
        abs_means_clamped = abs_means.clamp(min=1e-6)
        rates = 1.0 / abs_means_clamped
        
        # Expand rates for broadcasting
        rates_expanded = rates.expand(bsz, num_heads, seq_len, head_dim)
        
        # Generate exponential distributed values
        exp_dist = torch.distributions.Exponential(rates_expanded)
        fill_values = exp_dist.sample()
        
        # Preserve signs based on original distribution per head
        sign_probs = (valid_values_masked > 0).float().sum(dim=(2, 3), keepdim=True) / (valid_counts + 1e-8)
        sign_probs_expanded = sign_probs.expand(bsz, num_heads, seq_len, head_dim)
        
        # Generate random signs
        signs = torch.bernoulli(sign_probs_expanded) * 2 - 1
        fill_values = fill_values * signs
    
    else:
        # Fallback to mean filling
        fill_values = head_means.expand(bsz, num_heads, seq_len, head_dim)
    
    # Apply filling only to missing positions
    recovered_values = torch.where(missing_mask, fill_values.to(values.dtype), recovered_values)
    
    return recovered_values

def generate_value_mean_fill(values, mask):
    """
    Mean-based filling for value cache using tensor operations.
    Strategy: Use mean of pruned (masked-out) values to fill missing positions.
    """
    bsz, num_heads, seq_len, head_dim = values.shape
    recovered_values = values.clone()
    missing_mask = ~mask  # Positions that need to be filled
    pruned_mask = ~mask   # Positions that were pruned (same as missing_mask)
    
    token_pruned_counts = pruned_mask.sum(dim=3, keepdim=True)  # (bsz, num_heads, seq_len, 1)
    token_pruned_means = (values * pruned_mask).sum(dim=3, keepdim=True) / (token_pruned_counts + 1e-8)  # (bsz, num_heads, seq_len, 1)
    
    # Create hierarchical filling strategy
    has_pruned_dims = token_pruned_counts.squeeze(-1) > 0  # (bsz, num_heads, seq_len)
    token_fill = token_pruned_means.expand(-1, -1, -1, head_dim)  # (bsz, num_heads, seq_len, head_dim)
    
    return torch.where(missing_mask, token_fill, recovered_values)

def dynamic_score_selection_norm(queries, keys, threshold_ratio=0, key_channel_compression_ratio=0, pooling_ratio=0):
    bsz, num_heads, seq_len, head_dim = keys.shape
    # queries_norm = torch.nn.functional.normalize(queries, dim=-1)
    # keys_norm = torch.nn.functional.normalize(keys, dim=-1)
    
    q_norm = torch.norm(queries, dim=-2, p=2).unsqueeze(-2)
    k_norm = torch.pow(keys, 2)
    # k_norm = torch.abs(keys)
    sorted_indices = torch.argsort(k_norm * q_norm, dim=-1, descending=True)
    contributions = torch.pow(keys, 2) * torch.norm(queries, dim=-2, p=2).unsqueeze(-2)
    # contributions = torch.abs(keys) * torch.norm(queries, dim=-2, p=2).unsqueeze(-2)
    group_indicator = torch.zeros(bsz, num_heads, seq_len).to(queries.device)
    topk_ratios = [key_channel_compression_ratio]
    topk = head_dim - int(key_channel_compression_ratio * head_dim)
    if threshold_ratio != 0:
        sorted_contributions = torch.gather(contributions, dim=-1, index=sorted_indices)
        cumulative_scores = torch.cumsum(sorted_contributions, dim=-1)
        is_above_threshold = cumulative_scores > (0.99 * cumulative_scores[..., -1]).unsqueeze(-1)
        threshold_indices = is_above_threshold.to(torch.float32).argmax(dim=-1)
        if threshold_ratio == 0.99:
            mask, topk_ratios, group_indicator = dynamic_group(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.990:
            mask, topk_ratios, group_indicator = dynamic_group0(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.991:
            mask, topk_ratios, group_indicator = dynamic_group1(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.992:
            mask, topk_ratios, group_indicator = dynamic_group2(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.9922:
            mask, topk_ratios, group_indicator = dynamic_group22(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.993:
            mask, topk_ratios, group_indicator = dynamic_group3(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.994:
            mask, topk_ratios, group_indicator = dynamic_group4(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.995:
            mask, topk_ratios, group_indicator = dynamic_group5(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.996:
            mask, topk_ratios, group_indicator = dynamic_group6(is_above_threshold, head_dim, keys, sorted_indices)
        elif threshold_ratio == 0.997:
            mask, topk_ratios, group_indicator = dynamic_group7(is_above_threshold, head_dim, keys, sorted_indices)
        else:
            mask = create_mask_by_threshold(sorted_indices, threshold_indices)

        num_selected_channels = torch.sum(mask).item()
        print(f"Number of selected channels: {num_selected_channels}")
        selected_channels_per_token = torch.sum(mask, dim=-1)
        print(f"Selected channels per token shape: {selected_channels_per_token}")
        print(f"Example: Selected channels for [batch=0, head=0, token=0]: {selected_channels_per_token[0, 0, 0].item()}")
    else:
        mask = torch.ones_like(keys, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices[..., -int(key_channel_compression_ratio * head_dim):], False)


    if pooling_ratio == 0.5:
        return generate_pca_fill(keys.to(torch.float32), mask).to(keys.dtype)
    elif pooling_ratio == 0.3:
        return generate_interpolated_fill(keys, mask)
    elif pooling_ratio == 0.9:
        return gamma(keys, ~mask)
    elif pooling_ratio == 0.7:
        return normal(keys, ~mask)
    elif pooling_ratio == 0.75:
        return normal_attn(contributions, q_norm, keys, ~mask, sorted_indices, topk)
    elif pooling_ratio == 0.755 or pooling_ratio == 0.754:
        return normal_attn(contributions, q_norm, keys, ~mask, sorted_indices, topk, False)
    elif pooling_ratio == 0.7555 or pooling_ratio == 0.7554:
        return normal_attn(contributions, q_norm, keys, ~mask, sorted_indices, topk, False, True)
    elif pooling_ratio == 0.6:
        return exponential(keys, ~mask)
    elif pooling_ratio == 0.65:
        return exponential_attn(contributions, q_norm, keys, ~mask, sorted_indices, topk)
    elif pooling_ratio == 0.654 or pooling_ratio == 0.6555:
        return exponential_attn(contributions, q_norm, keys, ~mask, sorted_indices, topk, False)
    elif pooling_ratio == 0.6556 or pooling_ratio == 0.6554:
        return exponential_attn(contributions, q_norm, keys, ~mask, sorted_indices, topk, False, True)
    elif pooling_ratio == threshold_ratio:
        return exponential_attn_grouped(contributions, q_norm, keys, ~mask, sorted_indices, group_indicator, topk_ratios, True, False)
    
    pruned_keys = keys * mask
    return pruned_keys


def dynamic_group(is_above_threshold, head_dim, keys, sorted_indices):
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.2 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.2 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.6 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.6 * head_dim)) & (first_above_threshold_idx <= int(0.8 * head_dim))] = 3
    group_indicator[first_above_threshold_idx > int(0.8 * head_dim)] = 4

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for group_id, ratio in zip([0, 1, 2, 3, 4], topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group22(is_above_threshold, head_dim, keys, sorted_indices):
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.2 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.2 * head_dim)) & (first_above_threshold_idx <= int(0.3 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.5 * head_dim))] = 3
    group_indicator[(first_above_threshold_idx > int(0.5 * head_dim)) & (first_above_threshold_idx <= int(0.6 * head_dim))] = 4
    group_indicator[(first_above_threshold_idx > int(0.6 * head_dim)) & (first_above_threshold_idx <= int(0.8 * head_dim))] = 5
    group_indicator[first_above_threshold_idx > int(0.8 * head_dim)] = 6

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group3(is_above_threshold, head_dim, keys, sorted_indices):
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.2 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.2 * head_dim)) & (first_above_threshold_idx <= int(0.3 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.5 * head_dim))] = 3
    group_indicator[first_above_threshold_idx > int(0.5 * head_dim)] = 4

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group4(is_above_threshold, head_dim, keys, sorted_indices):
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.3 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.5 * head_dim))] = 2
    group_indicator[first_above_threshold_idx > int(0.5 * head_dim)] = 3

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.3, 0.4, 0.5, 0.6]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group5(is_above_threshold, head_dim, keys, sorted_indices):
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.3 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.5 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.5 * head_dim)) & (first_above_threshold_idx <= int(0.6 * head_dim))] = 3
    group_indicator[first_above_threshold_idx > int(0.6 * head_dim)] = 4

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group0(is_above_threshold, head_dim, keys, sorted_indices):
    bsz, num_heads, seq_len, head_dim = keys.shape
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.2 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.2 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.6 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.6 * head_dim)) & (first_above_threshold_idx <= int(0.8 * head_dim))] = 3
    group_indicator[first_above_threshold_idx > int(0.8 * head_dim)] = 4

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    for group_id, ratio in zip([0, 1, 2, 3, 4], topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group1(is_above_threshold, head_dim, keys, sorted_indices):
    bsz, num_heads, seq_len, head_dim = keys.shape
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.25 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.25 * head_dim)) & (first_above_threshold_idx <= int(0.5 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.5 * head_dim)) & (first_above_threshold_idx <= int(0.75 * head_dim))] = 2
    group_indicator[first_above_threshold_idx > int(0.75 * head_dim)] = 3

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.25, 0.5, 0.75, 1.0]
    for group_id, ratio in zip([0, 1, 2, 3], topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group2(is_above_threshold, head_dim, keys, sorted_indices):
    bsz, num_heads, seq_len, head_dim = keys.shape
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.25 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.25 * head_dim)) & (first_above_threshold_idx <= int(0.3 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.35 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.35 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 3
    group_indicator[first_above_threshold_idx > int(0.4 * head_dim)] = 4

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.2, 0.3, 0.35, 0.4, 0.5]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group6(is_above_threshold, head_dim, keys, sorted_indices):
    bsz, num_heads, seq_len, head_dim = keys.shape
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.25 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.25 * head_dim)) & (first_above_threshold_idx <= int(0.3 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.35 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.35 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 3
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.6 * head_dim))] = 4
    group_indicator[first_above_threshold_idx > int(0.6 * head_dim)] = 5

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def dynamic_group7(is_above_threshold, head_dim, keys, sorted_indices):
    bsz, num_heads, seq_len, head_dim = keys.shape
    first_above_threshold_idx = torch.argmax(is_above_threshold.float(), dim=-1)
    group_indicator = torch.zeros_like(first_above_threshold_idx)

    group_indicator[first_above_threshold_idx <= int(0.3 * head_dim)] = 0
    group_indicator[(first_above_threshold_idx > int(0.3 * head_dim)) & (first_above_threshold_idx <= int(0.4 * head_dim))] = 1
    group_indicator[(first_above_threshold_idx > int(0.4 * head_dim)) & (first_above_threshold_idx <= int(0.5 * head_dim))] = 2
    group_indicator[(first_above_threshold_idx > int(0.5 * head_dim)) & (first_above_threshold_idx <= int(0.7 * head_dim))] = 3
    group_indicator[first_above_threshold_idx > int(0.7 * head_dim)] = 4

    total_mask = torch.zeros_like(keys, dtype=torch.bool)
    topk_ratios = [0.3, 0.35, 0.4, 0.45, 0.5]
    for group_id, ratio in zip(list(range(len(topk_ratios))), topk_ratios):
        group_mask = (group_indicator == group_id)
        
        if not group_mask.any():
            continue
        
        keep_channels = int(ratio * head_dim)
        
        top_indices = sorted_indices[..., :keep_channels]
        group_topk_mask = torch.zeros_like(keys, dtype=torch.bool).scatter_(-1, top_indices, True) 
        group_mask_expanded = group_mask.unsqueeze(-1).expand_as(total_mask)
        
        sub_mask = group_mask_expanded & group_topk_mask
        
        total_mask |= sub_mask
    return total_mask, topk_ratios, group_indicator

def exponential_attn_grouped(scores, queries, keys, mask, index, group_indicator, topk_ratios=[0.3, 0.5, 0.7], is_avg=True, is_up=False):
    bsz, num_heads, seq_len, head_dim = keys.shape
    device = keys.device
    
    new_values_total = torch.zeros_like(keys)
    
    for group_id, ratio in enumerate(topk_ratios):
        group_mask = (group_indicator == group_id)
        if not group_mask.any():
            continue
        
        group_mask_expanded = group_mask.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        group_scores = torch.where(group_mask_expanded, scores, torch.tensor(-float('inf'), device=device))
        group_queries = torch.where(group_mask_expanded, queries, torch.tensor(1.0, device=device))
        
        k = max(1, int(ratio * head_dim))
        threshold_group = torch.topk(group_scores, k=k, dim=-1)[0][..., -1, None]
        
        mask_low_group = group_scores < threshold_group
        masked_scores_group = torch.where(mask_low_group, group_scores, torch.tensor(0.0, device=device))
        
        if is_avg:
            mask_sum_group = mask_low_group.sum(dim=-1, keepdim=True)
            mask_sum_group = mask_sum_group + (mask_sum_group == 0) * 1
            avg_scores_group = masked_scores_group.sum(dim=-1, keepdim=True) / mask_sum_group
            new_values_group = torch.sqrt(avg_scores_group / group_queries)
        else:
            mean_group = group_scores.mean(dim=-1, keepdim=True)
            mean_group = torch.clamp(mean_group, min=1e-5)
            rate_group = 1.0 / mean_group
            
            exponential_dist = torch.distributions.exponential.Exponential(rate_group)
            values_group = exponential_dist.sample((head_dim,)).permute(1, 2, 3, 0)
            
            sorted_values_group = torch.sort(values_group, dim=-1, descending=True)[0]
            output_values_group = torch.zeros_like(values_group)
            output_values_group = output_values_group.scatter(dim=-1, index=index, src=sorted_values_group)
            
            threshold_expanded_group = threshold_group.expand_as(output_values_group)
            if is_up:
                values_group = torch.where(output_values_group > threshold_expanded_group, threshold_expanded_group, output_values_group)
            else:
                values_group = torch.where(output_values_group > threshold_expanded_group, avg_scores_group, output_values_group)
            
            new_values_group = torch.sqrt(values_group / group_queries)
        
        new_values_total = torch.where(group_mask_expanded, new_values_group, new_values_total)
    
    final_mask = torch.where(
        mask == 1, 
        torch.where(keys > 0, torch.tensor(1, dtype=keys.dtype), torch.tensor(-1, dtype=keys.dtype)),
        torch.tensor(0, dtype=keys.dtype)
    )
    return torch.where(
        final_mask == 1, new_values_total.abs(),
        torch.where(final_mask == -1, -new_values_total.abs(), keys)
    )

def exponential_attn(scores, queries, keys, mask, index, topk=63, is_avg=True, is_up=False):
    bsz, num_heads, seq_len, head_dim = keys.shape
    mask = torch.where(
        mask == 1, 
        torch.where(keys > 0, torch.tensor(1, dtype=keys.dtype), torch.tensor(-1, dtype=keys.dtype)),
        torch.tensor(0, dtype=keys.dtype)
    )
    threshold = torch.topk(scores, k=topk, dim=-1)[0][..., -1, None]
    mask_low = scores < threshold 
    masked_scores = torch.where(mask_low, scores, torch.tensor(0.0, device=scores.device)) 
    mask_sum = mask_low.sum(dim=-1, keepdim=True)
    mask_sum = mask_sum + (mask_sum == 0) * 1
    avg_scores = (masked_scores.sum(dim=-1, keepdim=True) / mask_sum).repeat(1, 1, 1, head_dim)
    if is_avg:
        new_values = torch.sqrt(avg_scores / queries)
    else:
        mean = scores.mean(dim=-1, keepdim=True)
        mean = torch.clamp(mean, min=1e-5)
        rate = 1.0 / mean
        
        exponential_dist = torch.distributions.exponential.Exponential(rate)
        values = exponential_dist.sample((scores.shape[-1],)).squeeze(-1).permute(1, 2, 3, 0)
        threshold_expanded = threshold.expand_as(values)
        sorted_values = torch.sort(values, dim=-1, descending=True)[0]
        output_values = torch.zeros_like(values) 
        output_values = output_values.scatter(dim=-1, index=index, src=sorted_values)
        if is_up:
            values = torch.where(output_values > threshold_expanded, threshold_expanded, output_values)
        else:
            values = torch.where(output_values > threshold_expanded, avg_scores, output_values)
        new_values = torch.sqrt(values / queries)

    return torch.where(
        mask == 1, new_values.abs(),                     
        torch.where(mask == -1, -new_values.abs(), keys)
    )

def exponential(keys, mask):
    mask = torch.where(
        mask == 1, 
        torch.where(keys > 0, torch.tensor(1, dtype=keys.dtype), torch.tensor(-1, dtype=keys.dtype)),
        torch.tensor(0, dtype=keys.dtype)
    )
    mean = keys.mean(dim=-1, keepdim=True)
    mean = torch.clamp(mean, min=1e-5)
    rate = 1.0 / mean
    
    exponential_dist = torch.distributions.exponential.Exponential(rate)
    new_values = exponential_dist.sample((keys.shape[-1],)).squeeze(-1).view(keys.shape[0], keys.shape[1], keys.shape[2], -1)
    return torch.where(
        mask == 1, new_values.abs(),                     
        torch.where(mask == -1, -new_values.abs(), keys)
    )

def normal_attn(scores, queries, keys, mask, index, topk=63, is_avg=True, is_up=False):
    bsz, num_heads, seq_len, head_dim = keys.shape
    mask = torch.where(
        mask == 1, 
        torch.where(keys > 0, torch.tensor(1, dtype=keys.dtype), torch.tensor(-1, dtype=keys.dtype)),
        torch.tensor(0, dtype=keys.dtype)
    )

    threshold = torch.topk(scores, k=topk, dim=-1)[0][..., -1, None]
    mask_low = scores < threshold 
    masked_scores = torch.where(mask_low, scores, torch.tensor(0.0, device=scores.device)) 
    mask_sum = mask_low.sum(dim=-1, keepdim=True)
    mask_sum = mask_sum + (mask_sum == 0) * 1
    avg_scores = (masked_scores.sum(dim=-1, keepdim=True) / mask_sum).repeat(1, 1, 1, head_dim)
    if is_avg:
        new_values = torch.sqrt(avg_scores / queries)
    else:
        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True)
        norm_dist = torch.distributions.normal.Normal(mean, std)
        values = norm_dist.sample((scores.shape[-1],)).squeeze(-1).permute(1, 2, 3, 0).abs()
        threshold_expanded = threshold.expand_as(values)
        sorted_values = torch.sort(values, dim=-1, descending=True)[0]
        output_values = torch.zeros_like(values) 
        output_values = output_values.scatter(dim=-1, index=index, src=sorted_values)
        if is_up:
            values = torch.where(output_values > threshold_expanded, threshold_expanded, output_values)
        else:
            values = torch.where(output_values > threshold_expanded, avg_scores, output_values)
        new_values = torch.sqrt(values / queries)

    return torch.where(
        mask == 1, new_values.abs(),                     
        torch.where(mask == -1, -new_values.abs(), keys)
    )

def normal(keys, mask):
    mask = torch.where(
        mask == 1, 
        torch.where(keys > 0, torch.tensor(1, dtype=keys.dtype), torch.tensor(-1, dtype=keys.dtype)),
        torch.tensor(0, dtype=keys.dtype)
    )
    mean = keys.mean(dim=-1, keepdim=True)
    std = keys.std(dim=-1, keepdim=True)
    mean = torch.clamp(mean, min=1e-5)
    std = torch.clamp(std, min=1e-5)
    norm_dist = torch.distributions.normal.Normal(mean, std)
    new_values = norm_dist.sample((keys.shape[-1],)).squeeze(-1).view(keys.shape[0], keys.shape[1], keys.shape[2], -1)
    return torch.where(
        mask == 1, new_values.abs(),                     
        torch.where(mask == -1, -new_values.abs(), keys)
    )

def gamma(keys, mask):
    min_val = keys.min(dim=(0, 1, 2), keepdim=True)
    if min_val < 0: 
        keys = keys - min_val + 1e-5

    mean = keys.mean(dim=(0, 1, 2), keepdim=True)
    variance = keys.var(dim=(0, 1, 2), keepdim=True)

    mean = torch.clamp(mean, min=1e-5) 
    variance = torch.clamp(variance, min=1e-5)

    shape = mean**2 / variance
    scale = variance / mean

    shape = shape.expand_as(keys)
    scale = scale.expand_as(keys)

    gamma_dist = torch.distributions.gamma.Gamma(shape, 1 / scale)
    new_values = gamma_dist.sample() + min_val - 1e-5
    return keys * mask + new_values * (~mask)


def generate_interpolated_fill(keys, mask):

    recovered_keys = keys.clone()
    bsz, num_heads, seq_len, head_dim = keys.shape
    # values_to_interpolate = keys[~mask]  
    surrounding_mean = torch.mean(keys[mask].view(bsz, num_heads, seq_len, -1), dim=-1)  
    surrounding_mean = surrounding_mean.unsqueeze(-1).expand_as(recovered_keys[~mask].view(bsz, num_heads, seq_len, -1))
    recovered_keys[~mask] = surrounding_mean.reshape(-1) 

    return recovered_keys

def generate_pca_fill(keys, mask, n_components=10):
    recovered_keys = keys.clone()  
    bsz, num_heads, seq_len, head_dim = keys.shape
    
    for b in range(bsz): 
        for h in range(num_heads):  
            sub_keys = keys[b, h]  # (seq_len, head_dim)
            sub_mask = mask[b, h]  # (seq_len, head_dim)
            
            nonzero_tokens = torch.nonzero(sub_mask.sum(dim=1), as_tuple=True)[0]
            if nonzero_tokens.numel() > 0:  
                valid_data = sub_keys[sub_mask]  
                valid_data = valid_data.view(sub_keys.size(0), -1)  

                if valid_data.size(0) > 1: 
                    pca_model = PCA(n_components=min(valid_data.size(-1), 10))  
                    pca_model.fit(valid_data.cpu().numpy())  


                    token_low_dims = pca_model.transform(valid_data[nonzero_tokens].cpu().numpy())
                    token_high_dims = pca_model.inverse_transform(token_low_dims)  # 恢复至原始空间
                    token_reconstructed = torch.tensor(token_high_dims, device=keys.device)
                    recovered_keys[b, h][~sub_mask] = token_reconstructed.view(-1).to(recovered_keys.dtype)

    return recovered_keys


