# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from sklearn.decomposition import PCA
import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half
import torch.nn.functional as F
from kvpress.presses.base_press import BasePress
import json


@dataclass
class AdaThinKPress(BasePress):
    """
    ThinK (https://arxiv.org/pdf/2407.21018) compresses the dimensions of the keys, and not the sequence length.
    Hence it can be combined with any other press that compresses the sequence length, e.g.
    press = ComposedPress([SnapKVPress(0.5), ThinKPress(0.5)])

    Here, we zero out the pruned dimensions resulting in no memory gain (the shape of the keys remains the same).
    To achieve memory savings, several options can be considered (see https://github.com/NVIDIA/kvpress/pull/18/),
    we might implement them in the future, especially if other similar presses are requested.

    This press has been reviewed by Yuhui Xu, first author of the ThinK paper.
    """

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

        # attention_scores = torch.matmul(queries, keys_expand.transpose(-1, -2))  # (bsz, num_heads, window_size, q_len)
        # attention_scores_norm = torch.norm(attention_scores, dim=-2)  # (bsz, num_heads, q_len)

        # # Step 3: Normalize attention scores
        # normalized_scores = attention_scores_norm / attention_scores_norm.sum(dim=-1, keepdim=True)  # Normalize along head_dim

        # # Step 4: Sort and dynamically select dimensions
        # sorted_indices = torch.argsort(normalized_scores, dim=-1, descending=True)  # (bsz, num_heads, head_dim)
        # cumulative_sum = torch.zeros_like(normalized_scores)  # Initialize cumulative sum
        # mask = torch.zeros_like(normalized_scores, dtype=torch.bool)
        # queries_norm = torch.pow(queries, 2).mean(dim=2)  # (bsz, num_heads, head_dim)
        # queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, module.head_dim).mean(2)
        # keys_norm = torch.pow(keys, 2).mean(dim=2)
        # key_scores = queries_norm * keys_norm  # (bsz, num_key_value_heads, head_dim)
        # breakpoint()
        queries = self.compute_window_queries(module, kwargs["hidden_states"], kwargs["position_embeddings"])
        queries_expand = queries.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size,module.head_dim).mean(2)
        # keys_expand = keys.unsqueeze(2).repeat(1, 1, num_key_value_groups, 1, 1).view(bsz, module.config.num_attention_heads, q_len, head_dim)
        # queries_norm = torch.pow(queries, 2).mean(dim=2) # (bsz, num_heads, self.window_size, head_dim)
        # queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, module.head_dim).mean(2).unsqueeze(-1)
        # breakpoint()
        # keys_squared = torch.pow(keys, 2)
        # key_scores = queries_norm * keys_squared
        # pruned_keys = dynamic_token_wise_dim_selection_norm(queries_expand, keys)
        # if self.pooling_ratio != 0:
        #     reshaped_keys = keys.view(-1, head_dim)  # Shape: (bsz * num_key_value_heads * q_len, head_dim)
        #     # mode = "max"
        #     compressed_dim = int(self.pooling_ratio * head_dim)
        #     kernel_size = int(1 / self.pooling_ratio)
        #     # print(self.mode)
        #     if self.mode == "avg":
        #         # Average Pooling
        #         pruned_keys = F.avg_pool1d(reshaped_keys.unsqueeze(1), kernel_size=kernel_size).squeeze(1)
        #     elif self.mode == "max":
        #         # Max Pooling
        #         pruned_keys, _ = F.max_pool1d(reshaped_keys.unsqueeze(1), kernel_size=kernel_size, return_indices=True)
        #         pruned_keys = pruned_keys.squeeze(1)
        #     else:
        #         raise ValueError(f"Invalid mode `{self.mode}`. Must be 'avg' or 'max'.")

        #     # Reshape pruned keys back to original batched shape
        #     pruned_keys = pruned_keys.view(bsz, num_key_value_heads, q_len, compressed_dim)

        #     # Unpooling (reverse the pooling operation)
        #     uncompressed_keys = F.interpolate(
        #         pruned_keys.view(-1, compressed_dim).unsqueeze(1),
        #         size=head_dim,
        #         mode="linear",
        #         align_corners=False
        #     ).squeeze(1)

        #     # Reshape uncompressed keys back to original batched shape
        #     pruned_keys = uncompressed_keys.view(bsz, num_key_value_heads, q_len, head_dim)
        #     # breakpoint()
        # else:
        # module.layer_idx
        # breakpoint()
        # bsz, num_heads, seq_len, head_dim = keys.shape
        # keys_dict = {f"{module.layer_idx}": {}}
        # for head_idx in range(num_heads):
        #     json_obj = {head_idx: keys[0, head_idx, :, :].tolist()}
        #     keys_dict[f"{module.layer_idx}"].update(json_obj)

        # output_file = f"{self.outpath}/keys.jsonl"
        # with open(output_file, "a+") as f:
        #     f.write(json.dumps(keys_dict) + "\n")
        pruned_keys = dynamic_score_selection_norm(queries_expand, keys, self.threshold_ratio, self.key_channel_compression_ratio, self.pooling_ratio)
        # pruned_keys = dynamic_score_selection(queries_expand, keys, self.threshold_ratio, self.key_channel_compression_ratio, self.pooling_ratio)
        
        # pruned_keys = prune_keys_to_norm(keys, queries_norm)
        # breakpoint()
        # Prune dimensions with the lowest scores by setting them to 0
        # n_pruned = int(head_dim * self.key_channel_compression_ratio)
        # indices = key_scores.topk(n_pruned, dim=-1, largest=False).indices
        # indices = indices.unsqueeze(2).expand(-1, -1, q_len, -1)
        # keys = keys.scatter_(-1, indices, 0)
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
    
    # Strategy 1: Value magnitude-based importance (per token, per dimension)
    # Keep both seq_len and head_dim dimensions for fine-grained control
    value_magnitudes = torch.abs(values)  # (bsz, num_heads, seq_len, head_dim)
    
    # # Strategy 2: Query-value interaction importance (per token, per dimension)
    # # Measure how much each value element contributes to query-value interactions
    # if queries.shape[2] > 0:  # if we have queries
    #     # Calculate interaction for each token-dimension pair
    #     queries_mean = queries.mean(dim=2)  # (bsz, num_heads, head_dim)
    #     queries_expanded = queries_mean.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bsz, num_heads, seq_len, head_dim)
        
    #     # Element-wise importance: how much each value element contributes
    #     interaction_scores = torch.abs(values * queries_expanded)  # (bsz, num_heads, seq_len, head_dim)
        
    #     # Combine magnitude and interaction scores
    #     importance_scores = value_magnitudes * 0.6 + interaction_scores * 0.4  # (bsz, num_heads, seq_len, head_dim)
    # else:
    importance_scores = value_magnitudes
    
    # Strategy 3: Information diversity preservation (per token, per dimension)
    # Calculate variance across sequence for each dimension
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
        # n_total_elements = seq_len * head_dim
        # n_keep = max(1, int(n_total_elements * (1 - compression_ratio)))
        
        # Flatten and sort
        # flat_scores = final_scores.view(bsz, num_heads, -1)
        _, sorted_indices = torch.sort(final_scores, dim=-1, descending=True)
        mask = torch.ones_like(values, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices[..., -int(compression_ratio * head_dim):], False)
        
        # # Create mask
        # mask = torch.zeros_like(values, dtype=torch.bool)
        # flat_mask = mask.view(bsz, num_heads, -1)
        
        # keep_indices = sorted_indices[..., :n_keep]
        # flat_mask.scatter_(-1, keep_indices, True)
        # mask = flat_mask.view(bsz, num_heads, seq_len, head_dim)
    
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
    
    # Strategy 1: Global mean of pruned values per head
    # Calculate mean of all pruned values for each head
    # pruned_valid_counts = pruned_mask.sum(dim=(2, 3), keepdim=True)  # (bsz, num_heads, 1, 1)
    # global_pruned_means = (values * pruned_mask).sum(dim=(2, 3), keepdim=True) / (pruned_valid_counts + 1e-8)  # (bsz, num_heads, 1, 1)
    
    # # Strategy 2: Dimension-wise mean of pruned values
    # # Calculate mean of pruned values for each dimension across sequence
    # dim_pruned_counts = pruned_mask.sum(dim=2, keepdim=True)  # (bsz, num_heads, 1, head_dim)
    # dim_pruned_means = (values * pruned_mask).sum(dim=2, keepdim=True) / (dim_pruned_counts + 1e-8)  # (bsz, num_heads, 1, head_dim)
    
    # Strategy 3: Token-wise mean of pruned values (MODIFIED)
    # For each token, use mean of its own pruned dimensions to fill missing positions
    token_pruned_counts = pruned_mask.sum(dim=3, keepdim=True)  # (bsz, num_heads, seq_len, 1)
    token_pruned_means = (values * pruned_mask).sum(dim=3, keepdim=True) / (token_pruned_counts + 1e-8)  # (bsz, num_heads, seq_len, 1)
    
    # Create hierarchical filling strategy
    # Priority 1: Use token's own pruned mean if token has pruned dimensions
    has_pruned_dims = token_pruned_counts.squeeze(-1) > 0  # (bsz, num_heads, seq_len)
    token_fill = token_pruned_means.expand(-1, -1, -1, head_dim)  # (bsz, num_heads, seq_len, head_dim)
    
    # # Priority 2: Use dimension-wise pruned mean for tokens with no pruned dimensions
    # dim_fill = dim_pruned_means.expand(-1, -1, seq_len, -1)  # (bsz, num_heads, seq_len, head_dim)
    
    # # Priority 3: Use global pruned mean as final fallback
    # global_fill = global_pruned_means.expand(-1, -1, seq_len, head_dim)  # (bsz, num_heads, seq_len, head_dim)
    
    # Apply hierarchical filling using tensor operations
    # Start with global pruned mean, then override with better estimates where available
    # fill_values = global_fill.clone()
    
    # # Override with dimension-wise pruned means where dimension has pruned data
    # dim_has_pruned_data = dim_pruned_counts.squeeze(2) > 0  # (bsz, num_heads, head_dim)
    # dim_has_pruned_expanded = dim_has_pruned_data.unsqueeze(2).expand(-1, -1, seq_len, -1)
    # fill_values = torch.where(dim_has_pruned_expanded, dim_fill, fill_values)
    
    # Override with token-wise pruned means where token has pruned dimensions
    # breakpoint()
    # has_pruned_dims_expanded = has_pruned_dims.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    # fill_values = torch.where(has_pruned_dims_expanded, token_fill, fill_values)
    
    # Apply the filling only to missing positions (mask == False)
    # recovered_values = torch.where(missing_mask, token_fill, recovered_values)
    
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

def dynamic_score_selection(queries, keys, threshold_ratio=0, key_channel_compression_ratio=0, pooling_ratio=0):
    bsz, num_heads, seq_len, head_dim = keys.shape
    queries_norm = torch.nn.functional.normalize(queries, dim=-1)
    keys_norm = torch.nn.functional.normalize(keys, dim=-1)
    attention_scores = torch.matmul(queries_norm, keys_norm.transpose(-1, -2))
    original_norm = torch.norm(attention_scores, dim=-2, p=2).unsqueeze(-1)

    dim_contributions = []
    contributions = []
    for d in range(queries.size(-1)):
        keys_masked = torch.zeros_like(keys_norm)
        keys_masked[..., d] = keys_norm[..., d]

        attention_scores_masked = torch.matmul(queries_norm, keys_masked.transpose(-1, -2))
        masked_norm = torch.norm(attention_scores_masked, dim=-2, p=2)
        contribution = masked_norm / original_norm.squeeze(-1)
        dim_contributions.append(masked_norm)
        contributions.append(contribution)
    dim_contributions = torch.stack(dim_contributions, dim=-1)
    sorted_indices = torch.argsort(dim_contributions, dim=-1, descending=True)
    if threshold_ratio != 0:
        contributions = torch.stack(contributions, dim=-1)
        sorted_contributions = torch.gather(contributions, dim=-1, index=sorted_indices)
        cumulative_scores = torch.cumsum(sorted_contributions, dim=-1)
        is_above_threshold = cumulative_scores > (threshold_ratio * cumulative_scores[..., -1]).unsqueeze(-1)
        threshold_indices = is_above_threshold.to(torch.float32).argmax(dim=-1)
        mask = create_mask_by_threshold(sorted_indices, threshold_indices)
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
        return normal_attn(dim_contributions, queries_norm, keys, ~mask, int(key_channel_compression_ratio * head_dim))
    elif pooling_ratio == 0.6:
        return exponential(keys, ~mask)
    elif pooling_ratio == 0.65:
        return exponential_attn(attention_scores, queries_norm, keys, ~mask)
    elif pooling_ratio == 0.99:
        return dynamic_group()
    num_selected_channels = torch.sum(mask).item()
    print(f"Number of selected channels: {num_selected_channels}")
    selected_channels_per_token = torch.sum(mask, dim=-1)
    print(f"Selected channels per token shape: {selected_channels_per_token}")
    print(f"Example: Selected channels for [batch=0, head=0, token=0]: {selected_channels_per_token[0, 0, 0].item()}")

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
    """
    对置零位置使用线性插值恢复。

    Args:
        keys (torch.Tensor): 原始数据 (bsz, num_heads, seq_len, head_dim)
        mask (torch.BoolTensor): 已经置零的位置 `False` 表示需要填充，`True` 表示保留

    Returns:
        torch.Tensor: 插值填充后的 keys 数据 (同样维度)
    """
    recovered_keys = keys.clone()
    bsz, num_heads, seq_len, head_dim = keys.shape
    # values_to_interpolate = keys[~mask]  # 取出所有被置零的部分
    surrounding_mean = torch.mean(keys[mask].view(bsz, num_heads, seq_len, -1), dim=-1)  # 取所有未被置零的部分的均值
    surrounding_mean = surrounding_mean.unsqueeze(-1).expand_as(recovered_keys[~mask].view(bsz, num_heads, seq_len, -1))
    recovered_keys[~mask] = surrounding_mean.reshape(-1) # 用均值填充所有置零处
    # for d in range(keys.size(-1)):  # 针对 head_dim 的每一维进行操作
    #     mask_dim = mask[..., d]  # 当前维度的掩码
    #     keys_dim = keys[..., d]  # 当前维度的特征值
    #     breakpoint()
    #     # 前后 token 的值进行插值 (矩阵化计算)
    #     prev_values = torch.roll(keys_dim, shifts=1, dims=2)  # 向前一个 token 的值
    #     next_values = torch.roll(keys_dim, shifts=-1, dims=2)  # 向后一个 token 的值
    #     interpolated_values = (prev_values + next_values) / 2  # 线性插值

    #     # 对掩码为 False 的位置进行填充
    #     recovered_keys[..., d][~mask_dim] = interpolated_values[~mask_dim]
    # for i in range(keys.size(-1)):  # 遍历最后一维
    #     mask_dim = mask[..., i]
    #     if torch.any(~mask_dim):  # 如果这一维有被置零的情况
    #         values_to_interpolate = keys[..., i][~mask_dim]  # 取出被置零的部分
    #         surrounding_mean = torch.mean(keys[..., i][mask_dim])  # 取未被置零的部分的均值
    #         recovered_keys[..., i][~mask_dim] = surrounding_mean  # 用均值填充置零处
    return recovered_keys

def generate_pca_fill(keys, mask, n_components=10):
    """
    使用非零部分数据的主成分重新生成置零数据。

    Args:
        keys (torch.Tensor): 原始数据 (bsz, num_heads, seq_len, head_dim)
        mask (torch.BoolTensor): 已经置零的位置 `False` 表示需要填充，`True` 表示保留

    Returns:
        torch.Tensor: 填充后的 keys 数据 (同样维度)
    """
    recovered_keys = keys.clone()  # 克隆原始数据以进行操作
    bsz, num_heads, seq_len, head_dim = keys.shape
    # keys_2d = keys.view(bsz * num_heads, seq_len, head_dim).permute(0, 2, 1).reshape(bsz * num_heads, head_dim * seq_len)
    # mask_2d = mask.view(bsz * num_heads, seq_len, head_dim).permute(0, 2, 1).reshape(bsz * num_heads, head_dim * seq_len)

    # # 找到所有有未置零值的 token 位置
    # nonzero_tokens = torch.nonzero(mask_2d.sum(dim=1), as_tuple=True)[0]

    # if nonzero_tokens.numel() > 0:  # 如果有未置零的 token
    #     valid_data = keys_2d[mask_2d].view(bsz * num_heads, seq_len, -1)  # 提取所有有效的上下文数据 (N, head_dim * seq_len)
    #     breakpoint()
    #     if valid_data.size(0) > 1:  # 如果有效数据不足，PCA 无法应用
    #         pca_model = PCA(n_components=min(valid_data.size(-1), 10))  # 限制主成分
    #         pca_model.fit(valid_data.cpu().numpy())  # 仅 fit 有效数据部分

    #         # 对所有有未置零值的 token 进行批量处理
    #         token_low_dims = pca_model.transform(valid_data[nonzero_tokens].cpu().numpy())
    #         token_high_dims = pca_model.inverse_transform(token_low_dims)  # 恢复至原始空间

    #         # 将重构的维度值赋回当前 token
    #         token_reconstructed = torch.tensor(token_high_dims, device=keys.device)
    #         breakpoint()
    #         recovered_keys_2d = recovered_keys.view(bsz * num_heads, seq_len, head_dim).permute(0, 2, 1).reshape(bsz * num_heads, head_dim * seq_len)
    #         recovered_keys_2d[~mask_2d] = token_reconstructed.view(-1).to(recovered_keys.dtype)

    # # 将恢复后的二维矩阵转换回原始的 batch 和 head 维度
    # recovered_keys = recovered_keys_2d.view(bsz, num_heads, head_dim, seq_len).permute(0, 1, 3, 2)
    # keys_reshaped = keys.view(-1, seq_len, head_dim)  # (bsz * num_heads, seq_len, head_dim)
    # mask_reshaped = mask.view(-1, seq_len, head_dim)  # (bsz * num_heads, seq_len, head_dim)
    
    for b in range(bsz):  # 遍历每个 batch
        for h in range(num_heads):  # 遍历每个 attention head
            sub_keys = keys[b, h]  # (seq_len, head_dim)
            sub_mask = mask[b, h]  # (seq_len, head_dim)
            
            nonzero_tokens = torch.nonzero(sub_mask.sum(dim=1), as_tuple=True)[0]
            if nonzero_tokens.numel() > 0:  # 如果有未置零的 token
                valid_data = sub_keys[sub_mask]  # 提取所有有效的上下文数据 (N, head_dim)
                valid_data = valid_data.view(sub_keys.size(0), -1)  # 确保二维矩阵 (N, head_dim)

                if valid_data.size(0) > 1:  # 如果有效数据不足，PCA 无法应用
                    pca_model = PCA(n_components=min(valid_data.size(-1), 10))  # 限制主成分
                    pca_model.fit(valid_data.cpu().numpy())  # 仅 fit 有效数据部分

                    # 对所有有未置零值的 token 进行批量处理
                    token_low_dims = pca_model.transform(valid_data[nonzero_tokens].cpu().numpy())
                    token_high_dims = pca_model.inverse_transform(token_low_dims)  # 恢复至原始空间
                    # 将重构的维度值赋回当前 token
                    token_reconstructed = torch.tensor(token_high_dims, device=keys.device)
                    recovered_keys[b, h][~sub_mask] = token_reconstructed.view(-1).to(recovered_keys.dtype)
    # recovered_keys = keys.clone()  # 克隆原始数据以进行操作
    # for b in range(keys.size(0)):  # 遍历 batch
    #     for h in range(keys.size(1)):  # 遍历 attention head
    #         sub_keys = keys[b, h]  # 当前 batch 和 head 的 keys (seq_len, head_dim)
    #         sub_mask = mask[b, h]  # 当前对应的有效掩码 (seq_len, head_dim)

    #         for t in range(sub_keys.size(0)):  # 遍历每个 token 的位置
    #             token_data = sub_keys[t]  # 当前 token 的特征数据 (head_dim,)
    #             token_mask = sub_mask[t]  # 当前 token 的掩码 (head_dim,)

    #             # 如果当前 token 有未置零的值，进行 PCA 动态重构
    #             if torch.any(token_mask):  # 检查是否有未置零数据
    #                 valid_data = sub_keys[sub_mask]  # 提取所有有效的上下文数据 (N, head_dim)
    #                 valid_data = valid_data.view(sub_keys.size(0), -1)  # 确保二维矩阵 (N, head_dim)
    #                 if valid_data.size(0) > 1:  # 如果有效数据不足，PCA 无法应用
    #                     pca_model = PCA(n_components=min(valid_data.size(-1), 10))  # 限制主成分
    #                     pca_model.fit(valid_data.cpu().numpy())  # 仅 fit 有效数据部分

    #                     # 当前 token 的置零位置动态恢复
    #                     missing_indices = torch.where(~token_mask)[0]  # 找到置零的位置 (head_dim 的维度索引)
    #                     if len(missing_indices) > 0:
    #                         token_low_dim = pca_model.transform(token_data[token_mask].cpu().numpy().reshape(1, -1))
    #                         token_high_dim = pca_model.inverse_transform(token_low_dim)  # 恢复至原始空间

    #                         # 将重构的维度值赋回当前 token
    #                         token_reconstructed = torch.tensor(token_high_dim, device=keys.device).squeeze(0)
    #                         recovered_keys[b, h, t, missing_indices] = token_reconstructed.to(recovered_keys.dtype)
    # breakpoint()
    return recovered_keys


def dynamic_token_wise_dim_selection_norm(queries, keys, threshold_ratio=0.999):
    """
    针对每个 token 动态筛选重要维度，保证每个 token 的乘积范数达到原始范数的 99%。

    Args:
        queries: Tensor of shape (bsz, num_heads, query_len, head_dim)
        keys: Tensor of shape (bsz, num_heads, key_len, head_dim)
        target_ratio: float, 保留原始范数的比例（默认 99%）

    Returns:
        compressed_queries: Tensor of shape (bsz, num_heads, query_len, selected_head_dim)
        compressed_keys: Tensor of shape (bsz, num_heads, key_len, selected_head_dim)
        mask: Boolean Tensor of shape (bsz, num_heads, query_len, head_dim), 表示每个 token 的筛选掩码
    """
    bsz, num_heads, seq_len, head_dim = keys.shape
    # Step 1: 计算原始范数
    queries_norm = torch.nn.functional.normalize(queries, dim=-1)
    keys_norm = torch.nn.functional.normalize(keys, dim=-1)
    attention_scores = torch.matmul(queries_norm, keys_norm.transpose(-1, -2))  # (bsz, num_heads, query_len, key_len)
    original_norm = torch.norm(attention_scores, dim=-2, p=2).unsqueeze(-1)  # (bsz, num_heads, key_len)
    # norms=torch.norm(torch.einsum('bhqd,bhkd->bhqk',queries,keys))
    # breakpoint()

    # Step 2: 计算每个维度的贡献
    dim_contributions = []
    for d in range(queries.size(-1)):
        keys_masked = torch.zeros_like(keys_norm)
        keys_masked[..., d] = keys_norm[..., d]  # 仅保留第 i 个维度的特征

        attention_scores_masked = torch.matmul(queries_norm, keys_masked.transpose(-1, -2))
        masked_norm = torch.norm(attention_scores_masked, dim=-2, p=2)  # (bsz, num_heads, key_len)
        dim_contributions.append(masked_norm)
    dim_contributions = torch.stack(dim_contributions, dim=-1)  # (bsz, num_heads, key_len, head_dim)
    sorted_indices = torch.argsort(dim_contributions, dim=-1, descending=True)  # 每个 token 按 channel 贡献排序 (bsz, num_heads, seq_len, head_dim)
    
    # Step 3: 动态选择 top-k channels
    # cumulative_sum = torch.zeros_like(dim_contributions)  # 用于存储累积和
    cumulative_ratio = torch.ones_like(dim_contributions)
    mask = torch.zeros_like(dim_contributions, dtype=torch.bool)  # 用于标记保留的 channel
    # prev_ratio = torch.zeros_like(original_norm)
    breakpoint()
    for i in range(head_dim):
        index = sorted_indices[:, :, :, i]  # 当前最高贡献的 channel 索引
        # breakpoint()
        selected_indices = sorted_indices[:, :, :, :i + 1]   
        keys_masked = torch.zeros_like(keys)
        keys_masked = keys_masked.scatter_(-1, selected_indices, keys.gather(-1, selected_indices))
        cur_score = torch.norm(torch.matmul(queries, keys_masked.transpose(-1, -2)), dim=-2, p=2).unsqueeze(-1)

        # cur_score = torch.sum(dim_contributions.gather(-1, sorted_indices[:, :, :, :i+1]), dim=-1).unsqueeze(-1)
        cur_ratio = cur_score / original_norm
        cumulative_ratio.scatter_(-1, index.unsqueeze(-1), cur_ratio)
        # cumulative_sum.scatter_add_(-1, index.unsqueeze(-1), cur_score)
        # current_ratio = cumulative_sum / original_norm # 当前累积平方和与原始平方和的比例
        # breakpoint()
        # mask = mask | (prev_ratio < threshold_ratio)
        mask = mask | (cumulative_ratio < threshold_ratio)  # 保留当前 channel
        # prev_ratio = current_ratio
        if torch.all((cur_score / original_norm) >= threshold_ratio):
            print('get threshold! Skip!')
            # mask = mask | (cumulative_ratio < threshold_ratio)
            # breakpoint()
            break
    sorted_cumulative_ratio = torch.gather(cumulative_ratio, dim=-1, index=sorted_indices)  # (bsz, num_heads, key_len, head_dim)
    is_above_threshold = sorted_cumulative_ratio > threshold_ratio  # (bsz, num_heads, key_len, head_dim)
    # 找到第一个满足条件的索引
    threshold_indices = is_above_threshold.to(torch.float32).argmax(dim=-1)  # (bsz, num_heads, key_len)
    breakpoint()
    mask = create_mask_by_threshold(sorted_indices, threshold_indices)
    
    num_selected_channels = torch.sum(mask).item()  # 统计所有位置中被选中的 channel 数量
    print(f"Number of selected channels: {num_selected_channels}")
    selected_channels_per_token = torch.sum(mask, dim=-1)  # (bsz, num_heads, seq_len)
    print(f"Selected channels per token shape: {selected_channels_per_token}")
    print(f"Example: Selected channels for [batch=0, head=0, token=0]: {selected_channels_per_token[0, 0, 0].item()}")
    
    # Step 4: 裁剪 keys
    pruned_keys = keys * mask # 将未保留的 channel 设置为 0
    return pruned_keys

def dynamic_token_wise_dim_selection(queries, keys, threshold_ratio=0.999):
    """
    针对每个 token 动态筛选重要维度，保证每个 token 的乘积范数达到原始范数的 99%。

    Args:
        queries: Tensor of shape (bsz, num_heads, query_len, head_dim)
        keys: Tensor of shape (bsz, num_heads, key_len, head_dim)
        target_ratio: float, 保留原始范数的比例（默认 99%）

    Returns:
        compressed_queries: Tensor of shape (bsz, num_heads, query_len, selected_head_dim)
        compressed_keys: Tensor of shape (bsz, num_heads, key_len, selected_head_dim)
        mask: Boolean Tensor of shape (bsz, num_heads, query_len, head_dim), 表示每个 token 的筛选掩码
    """
    bsz, num_heads, seq_len, head_dim = keys.shape
    # Step 1: 计算原始范数
    queries_norm = torch.nn.functional.normalize(queries, dim=-1)
    keys_norm = torch.nn.functional.normalize(keys, dim=-1)
    attention_scores = torch.matmul(queries, keys.transpose(-1, -2))  # (bsz, num_heads, query_len, key_len)
    original_norm = torch.norm(attention_scores, dim=-2, p=2).unsqueeze(-1)  # (bsz, num_heads, key_len)
    # norms=torch.norm(torch.einsum('bhqd,bhkd->bhqk',queries,keys))
    # breakpoint()

    # Step 2: 计算每个维度的贡献
    dim_contributions = []
    for d in range(queries.size(-1)):
        keys_masked = torch.zeros_like(keys)
        keys_masked[..., d] = keys[..., d]  # 仅保留第 i 个维度的特征

        attention_scores_masked = torch.matmul(queries, keys_masked.transpose(-1, -2))
        masked_norm = torch.norm(attention_scores_masked, dim=-2, p=2)  # (bsz, num_heads, key_len)
        dim_contributions.append(masked_norm)
    dim_contributions = torch.stack(dim_contributions, dim=-1)  # (bsz, num_heads, key_len, head_dim)
    sorted_indices = torch.argsort(dim_contributions, dim=-1, descending=True)  # 每个 token 按 channel 贡献排序 (bsz, num_heads, seq_len, head_dim)
    
    # Step 3: 动态选择 top-k channels
    # cumulative_sum = torch.zeros_like(dim_contributions)  # 用于存储累积和
    cumulative_ratio = torch.ones_like(dim_contributions)
    mask = torch.zeros_like(dim_contributions, dtype=torch.bool)  # 用于标记保留的 channel
    # prev_ratio = torch.zeros_like(original_norm)
    idx = 0
    for i in range(head_dim):
        index = sorted_indices[:, :, :, i]  # 当前最高贡献的 channel 索引
        # breakpoint()
        selected_indices = sorted_indices[:, :, :, :i + 1]   
        keys_masked = torch.zeros_like(keys)
        keys_masked = keys_masked.scatter_(-1, selected_indices, keys.gather(-1, selected_indices))
        cur_score = torch.norm(torch.matmul(queries, keys_masked.transpose(-1, -2)), dim=-2, p=2).unsqueeze(-1)

        # cur_score = torch.sum(dim_contributions.gather(-1, sorted_indices[:, :, :, :i+1]), dim=-1).unsqueeze(-1)
        cur_ratio = cur_score / original_norm
        cumulative_ratio.scatter_(-1, index.unsqueeze(-1), cur_ratio)
        # cumulative_sum.scatter_add_(-1, index.unsqueeze(-1), cur_score)
        # current_ratio = cumulative_sum / original_norm # 当前累积平方和与原始平方和的比例
        # breakpoint()
        # mask = mask | (prev_ratio < threshold_ratio)
        mask = mask | (cumulative_ratio < threshold_ratio)  # 保留当前 channel
        # prev_ratio = current_ratio
        if torch.all((cur_score / original_norm) >= threshold_ratio):
            print('get threshold! Skip!')
            # mask = mask | (cumulative_ratio < threshold_ratio)
            # breakpoint()
            break
    sorted_cumulative_ratio = torch.gather(cumulative_ratio, dim=-1, index=sorted_indices)  # (bsz, num_heads, key_len, head_dim)
    is_above_threshold = sorted_cumulative_ratio > threshold_ratio  # (bsz, num_heads, key_len, head_dim)
    # 找到第一个满足条件的索引
    threshold_indices = is_above_threshold.to(torch.float32).argmax(dim=-1)  # (bsz, num_heads, key_len)
    breakpoint()
    mask = create_mask_by_threshold(sorted_indices, threshold_indices)
    
    num_selected_channels = torch.sum(mask).item()  # 统计所有位置中被选中的 channel 数量
    print(f"Number of selected channels: {num_selected_channels}")
    selected_channels_per_token = torch.sum(mask, dim=-1)  # (bsz, num_heads, seq_len)
    print(f"Selected channels per token shape: {selected_channels_per_token}")
    print(f"Example: Selected channels for [batch=0, head=0, token=0]: {selected_channels_per_token[0, 0, 0].item()}")
    
    # Step 4: 裁剪 keys
    pruned_keys = keys * mask # 将未保留的 channel 设置为 0
    return pruned_keys

def create_mask_by_threshold(sorted_indices, threshold_indices):
    """
    根据 threshold_indices 和 sorted_indices 构造 mask，满足条件的及其之前的位置置为 1

    Args:
        sorted_indices: Tensor, 排序后的索引，形状 (bsz, num_heads, key_len, head_dim)
        threshold_indices: Tensor, 每个 token 的第一个满足条件的索引，形状 (bsz, num_heads, key_len)

    Returns:
        mask: Tensor, 布尔类型的 mask，形状为 (bsz, num_heads, key_len, head_dim)
    """
    bsz, num_heads, key_len, head_dim = sorted_indices.shape

    # 创建一个范围张量用于 mask 的比较，形状为 (1, 1, 1, head_dim)
    range_tensor = torch.arange(head_dim, device=sorted_indices.device).view(1, 1, 1, head_dim)

    # 将 threshold_indices 扩展为和 head_dim 可广播的形状
    threshold_indices_expanded = threshold_indices.unsqueeze(-1)  # (bsz, num_heads, key_len, 1)

    # 根据阈值构造 mask，范围内的置为 True
    mask = range_tensor <= threshold_indices_expanded  # (bsz, num_heads, key_len, head_dim)

    # mask 顺序仍基于原始排序，因此需要对其进行逆变换
    reverse_mask = torch.zeros_like(mask, dtype=torch.bool)  # 初始化全 False
    reverse_mask.scatter_(-1, sorted_indices, mask)  # 按照 sorted_indices 逆变换 mask

    return reverse_mask


def prune_keys(keys, queries, threshold_ratio=0.99):
    bsz, num_heads, seq_len, head_dim = keys.shape
    # 原始完整注意力分数
    attention_scores = torch.matmul(queries, keys.transpose(-1, -2))  # (bsz, num_heads, window_size, q_len)
    attention_scores_norm_original = torch.norm(attention_scores, dim=(-2, -1), p=2)  # (bsz, num_heads)

    # 记录原始注意力范数用于误差计算
    attention_norm_baseline = attention_scores_norm_original.mean()  # 一个标量，作为基准

    # keys_expand 是压缩的 keys，形状为 (bsz, num_heads, q_len, head_dim)
    # queries_expand 是压缩的 queries，形状为 (bsz, num_heads, window_size, head_dim)

    # 单独计算每个 head_dim 的注意力分数
    dim_attention_scores = []  # 存储每个 dim 的裁剪结果
    dim_attention_contributions = []  # 存储每个 dim 对总范数的贡献
    breakpoint()
    dim_contributions = []
    for d in range(module.head_dim):
        # 提取单维度的 queries 和 keys
        queries_single_dim = queries[..., d]  # (bsz, num_heads, query_len)
        keys_single_dim = keys[..., d]  # (bsz, num_heads, key_len)

        # 计算单维度的 attention_scores
        attention_scores_single_dim = torch.matmul(queries_single_dim.unsqueeze(-1), keys_single_dim.unsqueeze(-2))
        
        # 计算单维度的范数
        dim_norm = torch.norm(attention_scores_single_dim, dim=(-2, -1), p=2)  # (bsz, num_heads)
        dim_contributions.append(dim_norm)

    # 将每个维度的范数堆叠到一起
    dim_contributions = torch.stack(dim_contributions, dim=-1)  # (bsz, num_heads, head_dim)

    # 计算每个维度的平均贡献（用于排序）
    dim_contribution_mean = dim_contributions.mean(dim=(0, 1))  # 平均贡献 (head_dim,)

    sorted_indices = torch.argsort(dim_contribution_mean, descending=True)
    # # 将所有通道的结果堆叠起来
    # dim_attention_contributions = torch.stack(dim_attention_contributions, dim=-1)  # (bsz, num_heads, head_dim)

    # # 计算通道贡献的总体排序
    # dim_contribution_mean = dim_attention_contributions.mean(dim=(0, 1))  # 平均贡献 (head_dim,)
    # sorted_indices = torch.argsort(dim_contribution_mean, descending=True)  # 从高到低排序

    # # 初始化裁剪后的 keys 和累计误差
    # mask = torch.zeros_like(keys, dtype=torch.bool)  # (bsz, num_heads, q_len, head_dim)
    # current_attention_norm = 0.0

    # # 逐步裁剪通道
    # for i, dim_idx in enumerate(sorted_indices):
    #     # 开启当前维度的 mask
    #     mask[:, :, :, dim_idx] = True

    #     # 计算裁剪后的 keys
    #     keys_compressed = keys * mask.unsqueeze(-2)  # (bsz, num_heads, q_len, head_dim)

    #     # 计算裁剪后的注意力分数
    #     attention_scores_compressed = torch.matmul(queries, keys_compressed.transpose(-1, -2))
    #     current_attention_norm = torch.norm(attention_scores_compressed, dim=(-2, -1), p=2).mean()

    #     # 误差计算
    #     error_percentage = (attention_norm_baseline - current_attention_norm) / attention_norm_baseline

    #     # 判断是否满足误差阈值
    #     if error_percentage < 0.01:  # 如果误差小于 1%
    #         break

    # return keys * mask.unsqueeze(-2)


def prune_keys_to_norm(keys, queries_norm, threshold_ratio=0.99):
    """
    对每个 token 的 keys 动态裁剪 channel，保持范数为原始范数的 98%。
    keys: shape (bsz, num_heads, seq_len, head_dim)
    threshold_ratio: 保留范数比例，默认为 98%。
    返回: 裁剪后的 keys (bsz, num_heads, seq_len, head_dim)
    """
    bsz, num_heads, seq_len, head_dim = keys.shape
    # breakpoint()
    # Step 1: 计算原始 Frobenius 范数
    keys_squared = torch.pow(keys, 2)  # 每个 channel 的平方值
    # keys_norm = torch.matmul(queries_norm, keys_squared)
    # key_scores = queries_norm * keys_squared
    original_norm = keys_squared.sum(dim=-1, keepdim=True)  # 每个 token 的原始范数 (bsz, num_heads, seq_len, 1)
    
    # Step 2: 按 channel 的贡献排序
    sorted_indices = torch.argsort(keys_squared, dim=-1, descending=True)  # 每个 token 按 channel 贡献排序 (bsz, num_heads, seq_len, head_dim)
    
    # Step 3: 动态选择 top-k channels
    cumulative_sum = torch.zeros_like(keys_squared)  # 用于存储累积和
    mask = torch.zeros_like(keys_squared, dtype=torch.bool)  # 用于标记保留的 channel
    idx = 0
    for i in range(head_dim):
        index = sorted_indices[:, :, :, i]  # 当前最高贡献的 channel 索引
        # breakpoint()
        cur_score = torch.sum(keys_squared.gather(-1, sorted_indices[:, :, :, :i+1]), dim=-1).unsqueeze(-1)
        cumulative_sum.scatter_add_(-1, index.unsqueeze(-1), cur_score) 
        # cumulative_sum.scatter_add_(-1, index.unsqueeze(-1), keys_squared.gather(-1, index.unsqueeze(-1)))  # 更新累积和
        current_ratio = cumulative_sum / original_norm # 当前累积平方和与原始平方和的比例
        mask = mask | (current_ratio > threshold_ratio)  # 保留当前 channel
        if torch.all(cur_score / original_norm > threshold_ratio):
            print('get threshold! Skip!')
            # breakpoint()
            break

    
    num_selected_channels = torch.sum(mask).item()  # 统计所有位置中被选中的 channel 数量
    print(f"Number of selected channels: {num_selected_channels}")
    selected_channels_per_token = torch.sum(mask, dim=-1)  # (bsz, num_heads, seq_len)
    print(f"Selected channels per token shape: {selected_channels_per_token}")
    print(f"Example: Selected channels for [batch=0, head=0, token=0]: {selected_channels_per_token[0, 0, 0].item()}")
    # print(idx)
            # breakpoint()
        # if current_ratio >= threshold_ratio and idx == 0:
        #     idx = i + 1
        #     print(idx)
    # Step 4: 裁剪 keys
    pruned_keys = keys * mask.float()  # 将未保留的 channel 设置为 0
    return pruned_keys