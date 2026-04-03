#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# PanguEmbedded-7B vllm-ascend adapter.
#
# Architecture: standard LLaMA-like Dense (GQA + SiLU + RMSNorm + RoPE),
# with bias=True on q/k/v/o projections.  vLLM's built-in LlamaForCausalLM
# does not load attention bias weights, so we override load_weights() to
# handle them explicitly.

from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaAttention, LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP


class PanguEmbeddedAttention(LlamaAttention):
    """Extends LlamaAttention to support bias on QKV and O projections.

    PanguEmbedded sets bias=True in its config, so q/k/v/o weight files
    contain both a weight tensor and a bias tensor.  The base LlamaAttention
    uses QKVParallelLinear which already supports bias; we just need to make
    sure it is constructed with bias=True.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config=None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config=None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        # Pass bias flags through to the base class constructor.
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            **kwargs,
        )


class PanguEmbeddedForCausalLM(nn.Module, SupportsPP):
    """vLLM model wrapper for PanguEmbedded-7B.

    Delegates the full model body to LlamaModel and overrides weight loading
    to handle the attention projection biases that are absent in vanilla Llama.
    """

    # Weight names that differ between the HF checkpoint and vLLM internals.
    # vLLM merges q/k/v into a single QKV tensor; we handle bias accordingly.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    # Modules that live only on specific pipeline-parallel ranks.
    _pp_skip_modules = {"lm_head"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = getattr(vllm_config, "quant_config", None)
        cache_config = vllm_config.cache_config

        self.config = config
        self.quant_config = quant_config

        # Reuse LlamaModel with bias=True forwarded via config.
        # LlamaModel reads config.attention_bias / config.bias to decide
        # whether to create biases in QKVParallelLinear / RowParallelLinear.
        # PanguEmbedded stores this flag as config.bias.
        if not hasattr(config, "attention_bias"):
            config.attention_bias = getattr(config, "bias", False)

        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, config.vocab_size, logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights, remapping HF attention bias names to vLLM names.

        HF stores:
            model.layers.N.self_attn.q_proj.{weight,bias}
            model.layers.N.self_attn.k_proj.{weight,bias}
            model.layers.N.self_attn.v_proj.{weight,bias}
            model.layers.N.self_attn.o_proj.{weight,bias}

        vLLM merges q/k/v into:
            model.layers.N.self_attn.qkv_proj.{weight,bias}
        and keeps o_proj separate.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # tie lm_head to embed_tokens when configured.
            if (
                self.config.tie_word_embeddings
                and "lm_head.weight" in name
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Match both weight and bias tensors for packed QKV.
                if weight_name not in name:
                    continue
                # Replace e.g. "q_proj.weight" → "qkv_proj.weight"
                #            or "q_proj.bias"   → "qkv_proj.bias"
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
