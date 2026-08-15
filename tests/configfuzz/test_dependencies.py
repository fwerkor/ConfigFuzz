from __future__ import annotations

import json

from configfuzz.cli import main
from configfuzz.dependencies import (
    DependencyGraph,
    DependencyNodeKind,
    DependencyRelation,
    DependencyStatus,
)
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet, Evidence, EvidenceKind


def constraint_set(
    parameter: str,
    expression: str,
    kind: ConstraintKind,
    parameters: tuple[str, ...],
) -> ConstraintSet:
    result = ConstraintSet(parameter=parameter)
    result.add(
        Constraint(
            expression=expression,
            kind=kind,
            parameters=parameters,
            confidence=0.9,
            evidence=(Evidence(EvidenceKind.STATIC, "framework/config.py", 10),),
        )
    )
    return result


def test_builds_deduplicated_divisibility_hyperedge() -> None:
    expression = "num_attention_heads % tensor_model_parallel_size == 0"
    participants = ("num_attention_heads", "tensor_model_parallel_size")
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "num_attention_heads",
                expression,
                ConstraintKind.RELATION,
                participants,
            ),
            constraint_set(
                "tensor_model_parallel_size",
                expression,
                ConstraintKind.RELATION,
                participants,
            ),
        ],
        scope={"framework": "Megatron-LM", "version": "abc123"},
    )

    assert len(graph.edges) == 1
    edge = next(iter(graph.edges.values()))
    assert edge.relation is DependencyRelation.DIVISIBILITY
    assert edge.drivers == ("tensor_model_parallel_size",)
    assert edge.dependents == ("num_attention_heads",)
    assert edge.scope_dict == {"framework": "Megatron-LM", "version": "abc123"}
    assert graph.related_parameters("tensor_model_parallel_size") == (
        "num_attention_heads",
    )
    assert graph.affected_parameters("tensor_model_parallel_size") == (
        "num_attention_heads",
    )


def test_conditional_dependency_has_direction_and_feature_node() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "sequence_parallel",
                "sequence_parallel => tensor_model_parallel_size > 1",
                ConstraintKind.CONDITIONAL,
                ("sequence_parallel", "tensor_model_parallel_size"),
            )
        ]
    )

    edge = next(iter(graph.edges.values()))
    assert edge.guard == "sequence_parallel"
    assert edge.predicate == "tensor_model_parallel_size > 1"
    assert edge.relation is DependencyRelation.REQUIRES
    assert edge.drivers == ("sequence_parallel",)
    assert edge.dependents == ("tensor_model_parallel_size",)
    assert graph.nodes["sequence_parallel"].kind is DependencyNodeKind.FEATURE


def test_conditional_relation_preserves_predicate_direction() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "pipeline_model_parallel_size",
                "custom_layout is None => num_layers % pipeline_model_parallel_size == 0",
                ConstraintKind.CONDITIONAL,
                (
                    "pipeline_model_parallel_size",
                    "custom_layout",
                    "num_layers",
                ),
            )
        ]
    )

    edge = next(iter(graph.edges.values()))
    assert edge.drivers == (
        "custom_layout",
        "pipeline_model_parallel_size",
    )
    assert edge.dependents == ("num_layers",)
    assert "num_layers" in graph.affected_parameters(
        "pipeline_model_parallel_size"
    )


def test_derived_quantity_nodes_are_explicit_and_not_repairable() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "hidden_size",
                "head_dim == hidden_size / num_attention_heads",
                ConstraintKind.RELATION,
                ("hidden_size", "num_attention_heads", "head_dim"),
            )
        ]
    )

    assert graph.nodes["head_dim"].kind is DependencyNodeKind.DERIVED
    assert graph.nodes["hidden_size"].kind is DependencyNodeKind.PARAMETER
    assert graph.nodes["num_attention_heads"].kind is DependencyNodeKind.PARAMETER
    assert not graph._repairable_node("head_dim")


def test_environment_nodes_and_status_are_explicit() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "tensor_model_parallel_size",
                "args.world_size % tensor_model_parallel_size == 0",
                ConstraintKind.ENVIRONMENT,
                ("tensor_model_parallel_size", "args"),
            )
        ]
    )

    edge = next(iter(graph.edges.values()))
    assert edge.relation is DependencyRelation.ENVIRONMENT
    assert edge.status is DependencyStatus.ENVIRONMENT_SPECIFIC
    assert edge.participants == (
        "tensor_model_parallel_size",
        "args.world_size",
    )
    assert graph.nodes["args.world_size"].kind is DependencyNodeKind.ENVIRONMENT


def test_self_environment_attribute_is_retained_as_context() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "train_batch_size",
                "train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps * self.world_size",
                ConstraintKind.ENVIRONMENT,
                (
                    "train_batch_size",
                    "train_micro_batch_size_per_gpu",
                    "gradient_accumulation_steps",
                ),
            )
        ]
    )

    edge = next(iter(graph.edges.values()))
    assert "self.world_size" in edge.participants
    assert graph.nodes["self.world_size"].kind is DependencyNodeKind.ENVIRONMENT


def test_unscanned_source_symbol_is_derived_context() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "group_size",
                "len(excluded_ranks_set) < group_size",
                ConstraintKind.ENVIRONMENT,
                ("group_size",),
            )
        ],
        configuration_parameters=("group_size",),
    )

    assert graph.nodes["group_size"].kind is DependencyNodeKind.PARAMETER
    assert graph.nodes["excluded_ranks_set"].kind is DependencyNodeKind.DERIVED


def test_active_constraint_evaluation_respects_guard() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "sequence_parallel",
                "sequence_parallel => tensor_model_parallel_size > 1",
                ConstraintKind.CONDITIONAL,
                ("sequence_parallel", "tensor_model_parallel_size"),
            )
        ]
    )
    edge = next(iter(graph.edges.values()))

    inactive = graph.evaluate_edge(
        edge,
        {"sequence_parallel": False, "tensor_model_parallel_size": 1},
    )
    active = graph.evaluate_edge(
        edge,
        {"sequence_parallel": True, "tensor_model_parallel_size": 1},
    )

    assert inactive.active is False
    assert inactive.satisfied is True
    assert active.active is True
    assert active.satisfied is False


def test_type_constraints_are_executable() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "hidden_size",
                "hidden_size: integer",
                ConstraintKind.TYPE,
                ("hidden_size",),
            )
        ]
    )
    edge = next(iter(graph.edges.values()))

    assert graph.evaluate_edge(edge, {"hidden_size": 4096}).satisfied is True
    assert graph.evaluate_edge(edge, {"hidden_size": 4096.0}).satisfied is False


def test_plan_joint_mutation_repairs_divisible_dependent() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "tensor_model_parallel_size",
                "num_attention_heads % tensor_model_parallel_size == 0",
                ConstraintKind.RELATION,
                ("num_attention_heads", "tensor_model_parallel_size"),
            )
        ]
    )

    plan = graph.plan_joint_mutation(
        "tensor_model_parallel_size",
        6,
        {"tensor_model_parallel_size": 4, "num_attention_heads": 32},
    )

    assert plan.changes == {
        "tensor_model_parallel_size": 6,
        "num_attention_heads": 36,
    }
    assert plan.violated_edges == ()
    assert plan.unresolved_edges == ()
    assert plan.validation_order == (
        "tensor_model_parallel_size",
        "num_attention_heads",
    )


def test_plan_joint_mutation_repairs_required_threshold() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "sequence_parallel",
                "sequence_parallel => tensor_model_parallel_size > 1",
                ConstraintKind.CONDITIONAL,
                ("sequence_parallel", "tensor_model_parallel_size"),
            )
        ]
    )

    plan = graph.plan_joint_mutation(
        "sequence_parallel",
        True,
        {"sequence_parallel": False, "tensor_model_parallel_size": 1},
    )

    assert plan.changes == {
        "sequence_parallel": True,
        "tensor_model_parallel_size": 2,
    }
    assert plan.violated_edges == ()


def test_connected_components_and_round_trip() -> None:
    graph = DependencyGraph.from_constraint_sets(
        [
            constraint_set(
                "hidden_size",
                "hidden_size % tensor_model_parallel_size == 0",
                ConstraintKind.RELATION,
                ("hidden_size", "tensor_model_parallel_size"),
            ),
            constraint_set(
                "micro_batch_size",
                "micro_batch_size > 0",
                ConstraintKind.RANGE,
                ("micro_batch_size",),
            ),
        ]
    )

    assert graph.connected_components() == (
        ("hidden_size", "tensor_model_parallel_size"),
        ("micro_batch_size",),
    )

    restored = DependencyGraph.from_dict(graph.to_dict())
    assert restored.to_dict() == graph.to_dict()

    active_artifact = {
        "schema_version": 1,
        "active_validation": {"dependency_graph": graph.to_dict()},
    }
    restored_active = DependencyGraph.from_dict(active_artifact)
    assert restored_active.to_dict() == graph.to_dict()


def test_graph_and_plan_cli_round_trip(tmp_path) -> None:
    expression = "num_attention_heads % tensor_model_parallel_size == 0"
    scan_path = tmp_path / "scan.json"
    graph_path = tmp_path / "graph.json"
    baseline_path = tmp_path / "baseline.json"
    plan_path = tmp_path / "plan.json"
    scan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": [
                    constraint_set(
                        "tensor_model_parallel_size",
                        expression,
                        ConstraintKind.RELATION,
                        ("num_attention_heads", "tensor_model_parallel_size"),
                    ).to_dict()
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "num_attention_heads": 32,
                "tensor_model_parallel_size": 4,
            }
        ),
        encoding="utf-8",
    )

    assert main(["graph", str(scan_path), "--output", str(graph_path)]) == 0
    assert (
        main(
            [
                "plan-mutation",
                str(graph_path),
                str(baseline_path),
                "--parameter",
                "tensor_model_parallel_size",
                "--value",
                "6",
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))["plan"]
    assert plan["proposed_changes"] == {
        "tensor_model_parallel_size": 6,
        "num_attention_heads": 36,
    }
