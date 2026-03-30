# isort: skip_file
import os
import time
import random
import argparse
import statistics
from dataclasses import dataclass
from typing import List

os.environ["VLLM_USE_MODELSCOPE"] = "true"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# 从 PolicyFactory 动态获取可用策略，避免脚本与 policy.py 不同步
from vllm_ascend.core.policy import PolicyFactory
AVAILABLE_POLICIES = PolicyFactory.get_available_policies()


# ---------------------------------------------------------------------------
# 请求定义
# ---------------------------------------------------------------------------

@dataclass
class BenchRequest:
    idx: int
    arrival_time: float      # 相对于压测开始的偏移（秒）
    prompt_token_ids: List[int]
    max_output_tokens: int


@dataclass
class RequestResult:
    idx: int
    prompt_len: int
    output_len: int
    arrival_time: float      # 绝对时间
    finish_time: float       # 绝对时间

    @property
    def e2e_latency(self) -> float:
        return self.finish_time - self.arrival_time


# ---------------------------------------------------------------------------
# 生成请求（token 长度随机，模拟真实负载差异）
# ---------------------------------------------------------------------------

def generate_requests(
    num_requests: int,
    interval: float,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_output_tokens: int,
    max_output_tokens: int,
    seed: int,
) -> List[BenchRequest]:
    rng = random.Random(seed)
    requests = []
    for i in range(num_requests):
        prompt_len = rng.randint(min_prompt_tokens, max_prompt_tokens)
        output_len = rng.randint(min_output_tokens, max_output_tokens)
        requests.append(BenchRequest(
            idx=i,
            arrival_time=i * interval,
            prompt_token_ids=[1] * prompt_len,   # dummy token ids
            max_output_tokens=output_len,
        ))
    return requests


# ---------------------------------------------------------------------------
# 核心压测逻辑：直接驱动 LLMEngine，无网络开销
# ---------------------------------------------------------------------------

def run_benchmark(
    engine_args: EngineArgs,
    requests: List[BenchRequest],
    policy: str,
) -> List[RequestResult]:
    print(f"\n{'='*60}")
    print(f"策略: {policy}  |  请求数: {len(requests)}")
    print(f"{'='*60}")

    # Clear the global AscendConfig singleton so each policy gets a fresh config.
    from vllm_ascend.ascend_config import clear_ascend_config
    clear_ascend_config()

    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True)

    # 预先计算每个请求的绝对 arrival_time
    # 所有请求一次性加入引擎，调度器凭 arrival_time > now 过滤
    benchmark_start = time.monotonic()
    abs_arrival_times = {
        req.idx: benchmark_start + req.arrival_time
        for req in requests
    }

    for req in requests:
        sp = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=req.max_output_tokens,
        )
        engine.add_request(
            request_id=str(req.idx),
            prompt={"prompt_token_ids": req.prompt_token_ids},
            params=sp,
            arrival_time=abs_arrival_times[req.idx],
        )

    results: dict[str, RequestResult] = {}
    finished = 0
    total = len(requests)
    last_progress = time.monotonic()

    while finished < total:
        step_outputs = engine.step()
        now_t = time.monotonic()
        for output in step_outputs:
            req_id = output.request_id
            if output.finished and req_id not in results:
                idx = int(req_id)
                out_tokens = output.outputs[0].token_ids if output.outputs else []
                results[req_id] = RequestResult(
                    idx=idx,
                    prompt_len=len(requests[idx].prompt_token_ids),
                    output_len=len(out_tokens),
                    arrival_time=abs_arrival_times[idx],
                    finish_time=now_t,
                )
                finished += 1
                last_progress = now_t
                if finished % 100 == 0 or finished == total:
                    print(f"  完成 {finished}/{total} 个请求")
        # 如果超过 30s 没有新请求完成，说明剩余请求的 arrival_time 还未到，
        # 空转等待中——打印提示避免误以为卡死
        if now_t - last_progress > 30:
            remaining = total - finished
            print(f"  等待 {remaining} 个请求的 arrival_time 到达（已等待 {now_t - last_progress:.0f}s）")
            last_progress = now_t

    del engine
    return list(results.values())


# ---------------------------------------------------------------------------
# 统计与输出
# ---------------------------------------------------------------------------

def print_stats(policy: str, results: List[RequestResult]):
    latencies = sorted(r.e2e_latency for r in results)
    n = len(latencies)

    def pct(p):
        idx = min(int(p / 100 * n), n - 1)
        return latencies[idx]

    total_time = max(r.finish_time for r in results) - min(r.arrival_time for r in results)

    print(f"\n[{policy}] 结果统计 ({n} 个请求)")
    print(f"  总耗时:        {total_time:.2f} s")
    print(f"  吞吐量:        {n / total_time:.2f} req/s")
    print(f"  E2E 延迟 avg:  {statistics.mean(latencies):.3f} s")
    print(f"  E2E 延迟 p50:  {pct(50):.3f} s")
    print(f"  E2E 延迟 p80:  {pct(80):.3f} s")
    print(f"  E2E 延迟 p90:  {pct(90):.3f} s")
    print(f"  E2E 延迟 p95:  {pct(95):.3f} s")
    print(f"  E2E 延迟 p99:  {pct(99):.3f} s")
    print(f"  E2E 延迟 p100: {pct(100):.3f} s")


def save_csv(policy: str, results: List[RequestResult], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{policy}_results.csv")
    with open(path, "w") as f:
        f.write("idx,prompt_len,output_len,arrival_time,finish_time,e2e_latency\n")
        for r in sorted(results, key=lambda x: x.idx):
            f.write(f"{r.idx},{r.prompt_len},{r.output_len},"
                    f"{r.arrival_time:.6f},{r.finish_time:.6f},{r.e2e_latency:.6f}\n")
    print(f"  结果已保存: {path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="fcfs vs aging 压测脚本")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-model-len", type=int, default=26240)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--interval", type=float, default=0.1,
                        help="请求到达间隔（秒）")
    parser.add_argument("--min-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=512,
                        help="不开 chunked_prefill 时上限为 max_num_batched_tokens(默认2048)，"
                             "prompt 长度范围越大，aging vs fcfs 差异越明显")
    parser.add_argument("--min-output-tokens", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policies", nargs="+", default=AVAILABLE_POLICIES,
                        choices=AVAILABLE_POLICIES,
                        help=f"要测试的策略列表，可选: {AVAILABLE_POLICIES}，默认全部测试")
    parser.add_argument("--aging-time-weight", type=float, default=588.0 * 0.3,
                        help="aging 策略的时间权重（越大越像 FCFS）")
    parser.add_argument("--aging-token-weight", type=float, default=-1.0,
                        help="aging 策略的 token 权重（越负越倾向于短 prompt）")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results")
    return parser.parse_args()


def main():
    args = parse_args()

    requests = generate_requests(
        num_requests=args.num_requests,
        interval=args.interval,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )

    print(f"共生成 {len(requests)} 个请求")
    print(f"  prompt 长度: {args.min_prompt_tokens}~{args.max_prompt_tokens} tokens")
    print(f"  output 长度: {args.min_output_tokens}~{args.max_output_tokens} tokens")
    print(f"  到达间隔:    {args.interval} s")

    all_results = {}

    for policy in args.policies:
        engine_args = EngineArgs(
            model=args.model,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            additional_config={
                "ascend_scheduler_config": {
                    "enabled": True,
                    "policy": policy,
                    "aging_time_weight": args.aging_time_weight,
                    "aging_token_weight": args.aging_token_weight,
                }
            },
        )
        results = run_benchmark(engine_args, requests, policy)
        all_results[policy] = results
        print_stats(policy, results)
        save_csv(policy, results, args.output_dir)

    # 对比摘要（支持任意数量策略，以第一个策略为基准做差值）
    if len(args.policies) >= 2:
        print(f"\n{'='*60}")
        print(f"对比摘要（基准: {args.policies[0]}）")
        print(f"{'='*60}")

        sorted_results = {
            p: sorted(r.e2e_latency for r in all_results[p])
            for p in args.policies
        }
        n = len(next(iter(sorted_results.values())))

        def pct(lst, p):
            return lst[min(int(p / 100 * n), n - 1)]

        col_w = 12
        header = f"{'指标':<{col_w}}" + "".join(f"{p:>{col_w}}" for p in args.policies)
        for policy in args.policies[1:]:
            diff_label = f"vs {args.policies[0]}"
            header += f"{diff_label:>{col_w}}"
        print(header)

        baseline = args.policies[0]
        for label, p in [("avg", None), ("p50", 50), ("p80", 80), ("p90", 90), ("p99", 99), ("p100", 100)]:
            row = f"{label:<{col_w}}"
            values = {}
            for policy in args.policies:
                lst = sorted_results[policy]
                v = statistics.mean(lst) if p is None else pct(lst, p)
                values[policy] = v
                row += f"{v:>{col_w}.3f}"
            for policy in args.policies[1:]:
                diff = values[policy] - values[baseline]
                row += f"{diff:>+{col_w}.3f}"
            print(row)


if __name__ == "__main__":
    main()
