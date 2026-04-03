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
# with bias=True on q/k/v/o projections.
#
# PanguEmbedded uses config.bias=True instead of config.attention_bias=True.
# vLLM's LlamaForCausalLM reads attention_bias, so we sync the field in
# __init__ before delegating everything to the parent class.

from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM


class PanguEmbeddedForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        # PanguEmbedded stores the attention bias flag as `bias`.
        # vLLM's LlamaForCausalLM reads `attention_bias`, so sync it here.
        if not hasattr(config, "attention_bias"):
            config.attention_bias = getattr(config, "bias", False)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
