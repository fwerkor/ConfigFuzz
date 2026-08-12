# ConfigFuzz Research Plan

## 1. Problem statement

Mutation-based testing of large-model systems depends on configuration parameters whose valid domains are constrained by model structure, parallel strategy, implementation choices, hardware, software version, and available resources. In the lm-sv baseline, these constraints were accumulated through repeated manual experiments and are distributed across parameter pools, validators, scripts, comments, and model-specific branches.

ConfigFuzz investigates the following problem:

> Given a previously unsupported configuration parameter and its software context, infer a useful specification of its valid configuration domain and use that specification to guide mutation-based testing.

“Useful” is deliberately weaker than complete semantic recovery. A useful specification should reject most trivially invalid configurations while retaining bug-triggering and boundary configurations.

## 2. Research questions

The evaluation follows a continuous argument chain: RQ1 establishes the problem and its cost, RQ2 evaluates ConfigFuzz's coordinated mutation mechanism, and RQ3 determines whether the additional deep execution translates into defect discovery.

### RQ1: Constraint characteristics, native coverage, and violation cost

What characteristics do MindSpeed configuration constraints exhibit, how extensively are they covered by native validation, and what costs arise when uncovered constraints are violated?

Metrics:

- constraint category, arity, guard, scope, semantic class, and software-layer distributions;
- full explicit, partial, implicit/delayed, and uncovered validation rates;
- first-failure milestone distribution;
- median and P95 time-to-failure, GPU-seconds wasted, timeout rate, and error-message quality.

The reviewed manual corpus is the empirical object of study. It is not used as a self-oracle for evaluating recovery accuracy.

### RQ2: Deep execution efficiency, intent preservation, and diversity

Under identical frozen mutation intents and testing budgets, can ConfigFuzz generate more deep, valid, and diverse executions while preserving the requested target value?

Metrics:

- deep execution yield per GPU-hour and stage reach rates;
- target-value retention rate;
- coordinated-parameter count, modification distance, and solver cost;
- expected rejection and delayed failure rates;
- constraint, boundary, guard-transition, topology, feature-interaction, and backend-path diversity.

The required methods are raw mutation, native-validator guidance, constraint filtering without repair, ConfigFuzz's affected-region repair, and the global-repair ablation.

### RQ3: Historical bug replay and current bug discovery

Can ConfigFuzz replay and discover more real configuration-related framework bugs at lower cost than the comparison methods?

Metrics:

- historical bug replay rate under a buggy/fixed differential oracle;
- tests, wall time, and GPU-hours to the first reproducer;
- independent reproducible, developer-confirmed, and fixed current-version bugs;
- GPU-hours per confirmed bug and false-positive rate.

Historical exact reproducers are reserved for final confirmation and are never supplied as search inputs. Failures are counted only after excluding expected rejection, ordinary resource exhaustion, and infrastructure faults.

## 3. Constraint ontology

The first DSL should support the classes already visible in the baseline:

1. **Type constraints**: integer, float, Boolean, string, list, enumeration.
2. **Range constraints**: lower/upper bounds, open or closed intervals.
3. **Set constraints**: explicit finite choices.
4. **Arithmetic constraints**: divisibility, multiples, powers of two, alignment.
5. **Relational constraints**: `x <= y`, `x * y <= z`, `y % x == 0`.
6. **Conditional constraints**: `condition => constraint`.
7. **Model-specific constraints**: active only for selected architectures.
8. **Environment constraints**: world size, device count, backend capability, software version.
9. **Resource constraints**: memory and runtime feasibility under a measured environment.

Every inferred constraint must carry provenance, confidence, and scope. A numeric observation such as `batch_size <= 32` must not be generalized into a hard semantic constraint when it only reflects memory capacity on one machine.

## 4. Proposed system

### 4.1 Parameter context extraction

Input:

- parameter name or configuration path;
- repository and entry point;
- one known-working baseline configuration;
- optional model and environment metadata.

Extract:

- definition, default, annotation, CLI declaration, and documentation;
- reads, aliases, data-flow uses, branches, assertions, exceptions, and repair logic;
- related parameters appearing in the same expressions or program slice;
- external values such as world size, device count, and memory.

### 4.2 Static candidate mining

Sources of evidence:

- explicit guards and assertions;
- enum choices in argparse, dataclasses, schemas, YAML, and UI declarations;
- array indexing, division, reshape, sharding, and collective operations;
- validator and auto-repair code;
- documentation and error messages;
- LLM-assisted translation only after deterministic extraction has collected the relevant slice.

The LLM should propose normalized candidates and explanations, not serve as the sole oracle.

### 4.3 Active runtime probing

Generate high-information tests around:

- constants found in source (`c-1`, `c`, `c+1`);
- default values and nearby values;
- powers of two and divisors of related structural parameters;
- environment boundaries such as device count;
- configurations that distinguish competing candidate formulas.

Avoid the full Cartesian product. Jointly mutate only parameters connected by static data flow or by currently hypothesized relations.

### 4.4 Outcome classification

Use outcome labels that preserve validity, resource, infrastructure, and defect
semantics:

- `VALID`: reaches the selected execution milestone;
- `INVALID`: explicit configuration validation rejects the input;
- `RESOURCE_FAILURE`: capacity or resource exhaustion prevents execution;
- `INFRASTRUCTURE_FAILURE`: launcher, cluster, network, or dependency failure;
- `UNEXPLAINED_FAILURE`: a failure whose validity meaning is not yet known;
- `POTENTIAL_BUG`: satisfies current constraints but exposes a crash, inconsistency, or incorrect result.
- `UNKNOWN`: any remaining inconclusive observation.

This classifier is central to the research. Treating every failed run as `INVALID` will cause the system to learn away real bugs.

### 4.5 Constraint synthesis

Represent candidates in the bounded DSL and use a solver-backed CEGIS loop:

1. synthesize the simplest constraint set consistent with current labeled samples;
2. generate a configuration that distinguishes the current hypothesis from an alternative;
3. execute and classify the configuration;
4. update the sample set and repeat until the budget or convergence criterion is reached.

A practical objective can minimize:

- violated positive samples;
- accepted explicit-negative samples;
- number and complexity of constraints;
- dependence on environment-specific constants;
- disagreement with high-confidence static evidence.

### 4.6 Dependency constraint hypergraph

Normalize the recovered constraints into a hypergraph whose nodes are
parameters, feature switches, and environment values. Each edge retains the
complete formula, guard, evidence, confidence, scope, and validation status.
Direction is inferred only for bounded patterns such as divisibility, bounds,
and guarded requirements; ambiguous formulas remain symmetric.

The graph supports:

- deduplication of constraints indexed under multiple parameters;
- related/affected-parameter queries;
- connected-component discovery for bounded joint exploration;
- active-edge selection under a concrete configuration;
- status transitions from static candidate to dynamically supported,
  confirmed, environment-specific, scope-disputed, or contradicted.

### 4.7 Constraint-aware mutation

Use inferred constraints in four modes:

- random valid sampling;
- valid boundary sampling;
- joint relational sampling;
- single-constraint violation for testing validation and error handling.

Inputs that satisfy all inferred constraints but trigger system failure should be prioritized as potential defects.

## 5. Experimental datasets

### 5.1 RQ1 constraint audit

`corpus/lmsv/manual_constraints.yaml` is transformed into `experiments/rq1/constraint_audit.yaml`. Each record contains the normalized predicate and guard, participants, arity, scope, category, semantic class, provenance, software layer, first affected milestone, native-validation label, evidence, and review state.

Bootstrap classifications are deterministic annotation aids. Native coverage and first affected milestone remain unknown until supported by source or execution evidence. The current source-review queue is generated from pinned MindSpeed-LLM, MindSpeed, and Megatron-LM trees and must not be interpreted as automatic coverage labels.

### 5.2 RQ2 workloads and mutation intents

The workload registry contains seven training subjects: Qwen2, Llama2, ChatGLM3, Mixtral, DeepSeek-V3, InternVL3, and CogVideoX. Once their stable baseline configurations are bound, the primary RQ2 intent pool is generated independently of ConfigFuzz's recovered constraints: scalar fields exposed by the qualified baseline receive generic numeric/Boolean boundary mutations, and TP/PP/EP/CP fields receive generic topology values. Relation-derived divisibility boundaries, guard transitions, and other constraint-focused cases are emitted into a separate `constraint_challenge` pool. The primary `method_independent` pool selects 150 intents per workload and is frozen with a content hash before any method runs; the challenge pool is reported separately when used.

### 5.3 RQ3 historical bug benchmark

Historical candidates are mined from configuration-related fix commits with identifiable parent revisions. A diverse shortlist is manually reviewed before entries are admitted to `experiments/rq3/historical_bugs.yaml`. Admission requires a workload, observable non-performance oracle, repeatable failure on the buggy commit, passage on the fixed commit, and agreement with the patch root cause. Older verified bugs form the development split; newer verified bugs form the final evaluation split.

## 6. Experimental methods

RQ2 and RQ3 use the same core methods:

1. **Raw Mutation**: apply only the requested target assignment.
2. **Native-Validator Guided**: reject configurations caught by the framework validator without coordinating parameters.
3. **Constraint-Filter Only**: apply the audited constraints only as a filter.
4. **Static-Hard ConfigFuzz**: coordinate the affected region while treating every statically recovered candidate as a hard constraint. This ablates execution validation and uncertainty-aware constraint status.
5. **ConfigFuzz**: fix the target assignment, hard-enforce confirmed/environment-specific relations, retain unresolved candidates as confidence-tiered guidance, and coordinate the affected parameter region.
6. **Global Repair**: use the same status-aware constraint treatment as ConfigFuzz while allowing all parameters to change, serving as the locality ablation.

All methods use identical baselines, frozen intents, hardware/software environments, per-test timeouts, test-count budgets, and GPU-hour budgets. Randomized methods run at least five seeds.

## 7. Initial implementation stages

### Stage A: corpus and DSL

- inventory existing manually encoded constraints;
- finalize normalized representation;
- create unit-test snippets for every constraint class.

### Stage B: static miner

- Python AST and CST extraction;
- scoped alias and data-flow tracking;
- validator, argparse, dataclass, schema, YAML, and shell adapters;
- candidate ranking and provenance.

The current AST implementation now provides strict/broad modes, lexical scope
and exception-flow handling, local symbolic expansion, conditional
normalization, repair-control filtering, bounded cross-file function summaries,
argparse/dataclass/YAML declaration extraction, and file-level parallelism.
Its formal input is framework source. The lm-sv source tree remains a regression
fixture and manual-baseline source, not the method's inference oracle.

### Stage C: execution harness

- reproducible baseline configuration;
- process isolation, timeout, resource recording, and log capture;
- milestone-based validity oracle;
- explicit `UNKNOWN` and `POTENTIAL_BUG` handling.

### Stage D: synthesis

- template library and Z3 encoding;
- query selection and counterexample loop;
- stopping criteria and confidence calibration.

### Stage E: lm-sv integration

- replace fixed pools with generated specifications where available;
- retain manual constraints as fallback;
- compare mutation efficiency and bug discovery.

## 8. Experiment readiness

Completed without accelerator access:

- a 93-record RQ1 audit dataset with evidence-gated review fields;
- pinned static mining over MindSpeed-LLM, MindSpeed, and the required Megatron-LM revision;
- a ranked per-constraint native-validation review queue;
- deterministic RQ1, RQ2, and RQ3 metric aggregation;
- a unified JSONL run schema that records generation, intent preservation, active constraints and provenance, constraint status changes, affected repair region, exact solver modifications, milestones, outcomes, cost, runtime behavior signatures, and bug-oracle/root-cause evidence;
- workload and intent registries for all seven RQ2 training subjects;
- deterministic generation and SHA-256 freezing of RQ2 mutation intentions after baselines are bound;
- full-history mining of configuration-related fix candidates and a balanced 40-item RQ3 source-review shortlist;
- schemas and validators for the final historical bug benchmark;
- repository/environment fingerprinting and focused unit tests.

Remaining source-review work:

1. Manually adjudicate the RQ1 native-validation candidates and record exact code evidence.
2. Review the 23 constraints with no implementation-side static candidate for likely implicit or delayed enforcement sites.
3. Triage the RQ3 shortlist into verified development/evaluation bugs.
4. Bind stable dense, long-sequence/GQA/FlashAttention, and MoE baseline configurations.
5. Regenerate and freeze the method-independent mutation-intent pool after baseline qualification; retain constraint-challenge intents as a separately reported stress subset.

Remaining accelerator work:

1. Execute RQ1 satisfying/violating pairs and record first failure, wall time, GPU-seconds, peak memory, timeout, and message quality.
2. Run the six RQ2 methods under identical intent and GPU-hour budgets for at least five seeds where randomness applies.
3. Replay the verified historical benchmark on buggy/fixed revisions.
4. Run current-version campaigns, minimize independent failures, and seek developer confirmation.

`experiments/protocol.yaml` is the authoritative machine-readable experiment specification.
