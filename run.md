```
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
# 可选：用 ModelScope 加速下载
export VLLM_USE_MODELSCOPE=true

vllm serve Qwen/Qwen3-8B --max-model-len 26240 --port 8000
```
fcfs
```
vllm serve Qwen/Qwen3-8B --max-model-len 26240 --port 8000 \
  --additional-config '{"ascend_scheduler_config":{"enabled":true,"policy":"fcfs"}}'
```
aging
```
vllm serve Qwen/Qwen3-8B --max-model-len 26240 --port 8000 \
  --additional-config '{"ascend_scheduler_config":{"enabled":true,"policy":"aging"}}'
```
