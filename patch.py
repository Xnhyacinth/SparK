    
import logging
import math
from typing import Optional, Tuple
import sys
import torch
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from torch import nn

try:
    import xformers.ops
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")
    sys.exit(1)


def replace_llama_attn_with_xformers_attn():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # attention_interface: Callable = eager_attention_forward

    # if self.config._attn_implementation != "eager":
    #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
    #         logger.warning_once(
    #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
    #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
    #         )
    #     else:
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
        # input and output should be of form (bsz, q_len, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = xformers.ops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=None
        )
    else:
        # input and output should be of form (bsz, q_len, num_heads, head_dim)
        attn_output = xformers.ops.memory_efficient_attention(
            query_states,
            key_states,
            value_states,
            attn_bias=xformers.ops.LowerTriangularMask(),
        )
    attn_weights = None
    # attn_output, attn_weights = attention_interface(
    #     self,
    #     query_states,
    #     key_states,
    #     value_states,
    #     attention_mask,
    #     dropout=0.0 if not self.training else self.attention_dropout,
    #     scaling=self.scaling,
    #     **kwargs,
    # )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights