from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol, Sequence, Type, TypeVar


class SchedulableRequest(Protocol):
    arrival_time: float
    num_tokens: int
    num_prompt_tokens: int
    num_computed_tokens: int


T = TypeVar("T", bound=SchedulableRequest)


class Policy(ABC):

    @abstractmethod
    def get_priority(
        self,
        now: float,
        request: SchedulableRequest,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        requests: Sequence[T],
    ) -> List[T]:
        return sorted(
            list(requests),
            key=lambda request: self.get_priority(now, request),
            reverse=True,
        )


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        request: SchedulableRequest,
    ) -> float:
        return now - request.arrival_time


class AGING(Policy):

    def get_priority(
        self,
        now: float,
        request: SchedulableRequest,
    ) -> float:
        time_weight = 588.0 * 0.3
        token_weight = -1.0

        # Use num_prompt_tokens as the task-length signal so that the
        # short-job penalty stays meaningful during the decode phase.
        # (num_tokens - num_computed_tokens is always 1 during decode,
        # which makes the weight term useless and degrades aging to FCFS.)
        num_prompt_remaining = max(
            request.num_prompt_tokens - request.num_computed_tokens, 0
        )

        return time_weight * (now - request.arrival_time) + token_weight * (
            num_prompt_remaining
        )


class PolicyFactory:

    _POLICY_REGISTRY: dict[str, Type[Policy]] = {
        "fcfs": FCFS,
        "aging": AGING,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        policy_cls = cls._POLICY_REGISTRY.get(policy_name)
        if policy_cls is None:
            raise ValueError(
                f"Unsupported policy {policy_name!r}. "
                f"Available policies: {cls.get_available_policies()}"
            )
        return policy_cls(**kwargs)

    @classmethod
    def get_available_policies(cls) -> List[str]:
        return list(cls._POLICY_REGISTRY.keys())

