from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from configfuzz.experiment import ExperimentMethod, MutationIntent


PRIMARY_SEED = 2026
SENSITIVITY_SEEDS = (17, 42, 101, 2026, 4099)


@dataclass(frozen=True, slots=True)
class ScheduledReplication:
    intent_id: str
    workload_id: str
    method: ExperimentMethod
    seed: int
    role: str

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "workload_id": self.workload_id,
            "method": self.method.value,
            "seed": self.seed,
            "role": self.role,
        }


def is_seed_sensitivity_intent(intent_id: str) -> bool:
    """Select the fixed 20% seed-sensitivity subset without RNG state."""
    digest = hashlib.sha256(intent_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def build_replication_schedule(
    intents: Sequence[MutationIntent],
    *,
    methods: Sequence[ExperimentMethod],
    supported_workloads: Iterable[str] | None = None,
) -> list[ScheduledReplication]:
    allowed = set(supported_workloads) if supported_workloads is not None else None
    schedule: list[ScheduledReplication] = []
    for intent in intents:
        if allowed is not None and intent.workload_id not in allowed:
            continue
        sensitivity = is_seed_sensitivity_intent(intent.intent_id)
        for method in methods:
            schedule.append(
                ScheduledReplication(
                    intent_id=intent.intent_id,
                    workload_id=intent.workload_id,
                    method=method,
                    seed=PRIMARY_SEED,
                    role="primary",
                )
            )
            if sensitivity:
                for seed in SENSITIVITY_SEEDS:
                    if seed == PRIMARY_SEED:
                        continue
                    schedule.append(
                        ScheduledReplication(
                            intent_id=intent.intent_id,
                            workload_id=intent.workload_id,
                            method=method,
                            seed=seed,
                            role="seed_sensitivity",
                        )
                    )
    return schedule


def summarize_schedule(schedule: Sequence[ScheduledReplication]) -> dict[str, object]:
    by_role: dict[str, int] = {}
    by_method: dict[str, int] = {}
    sensitivity_intents: set[str] = set()
    primary_intents: set[str] = set()
    for item in schedule:
        by_role[item.role] = by_role.get(item.role, 0) + 1
        by_method[item.method.value] = by_method.get(item.method.value, 0) + 1
        if item.role == "primary":
            primary_intents.add(item.intent_id)
        else:
            sensitivity_intents.add(item.intent_id)
    return {
        "record_count": len(schedule),
        "primary_intent_count": len(primary_intents),
        "sensitivity_intent_count": len(sensitivity_intents),
        "role_counts": dict(sorted(by_role.items())),
        "method_counts": dict(sorted(by_method.items())),
    }
