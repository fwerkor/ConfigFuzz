# ConfigFuzz Research Plan

## 1. Problem statement

Mutation-based testing of large-model systems depends on configuration parameters whose valid domains are constrained by model structure, parallel strategy, implementation choices, hardware, software version, and available resources. In the lm-sv baseline, these constraints were accumulated through repeated manual experiments and are distributed across parameter pools, validators, scripts, comments, and model-specific branches.

ConfigFuzz investigates the following problem:

> Given a previously unsupported configuration parameter and its software context, infer a useful specification of its valid configuration domain and use that specification to guide mutation-based testing.

“Useful” is deliberately weaker than complete semantic recovery. A useful specification should reject most trivially invalid configurations while retaining bug-triggering and boundary configurations.

## 2. Research questions

### RQ1: Constraint recovery accuracy

How accurately can ConfigFuzz recover manually encoded constraints when those constraints are hidden from the inference component?

Metrics:

- precision and recall over normalized atomic constraints;
- valid-domain agreement on independently generated configurations;
- accuracy by constraint class;
- confidence calibration.

### RQ2: Testing efficiency

Does inferred constraint guidance reduce wasted executions?

Metrics:

- valid configuration rate;
- number of initialization failures;
- time and device-hours per valid test;
- number of unique valid boundary configurations under a fixed budget.

### RQ3: Bug-finding effectiveness

Does ConfigFuzz preserve or improve defect discovery compared with unrestricted mutation and manually constrained mutation?

Metrics:

- unique confirmed defects;
- time to first defect;
- defect-triggering configuration diversity;
- failures incorrectly learned as invalid constraints.

### RQ4: Generalization

Can constraints inferred from one model, framework version, or hardware environment transfer to unseen parameters or environments?

Metrics:

- leave-one-model-out performance;
- cross-version and cross-hardware valid-domain agreement;
- proportion of hard, conditional, environment, and resource constraints correctly separated.

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

## 5. Baseline dataset construction

Use the current lm-sv snapshot as a labeled corpus.

1. Enumerate mutable parameters from YAML pools, validators, mutators, scripts, and model configuration templates.
2. Normalize each manually implemented rule into the DSL.
3. Record source location, model scope, environment scope, and whether the code rejects, repairs, or merely warns.
4. Hide each target rule from ConfigFuzz during evaluation while retaining the remaining program context.
5. Validate inferred rules against generated configurations and, where feasible, actual executions.

The corpus should distinguish:

- rules confidently required by the underlying framework;
- conservative rules chosen by lm-sv;
- empirical resource limits;
- workaround rules for known implementation defects.

The normalized corpus lives at `corpus/lmsv/manual_constraints.yaml`. It uses a
single schema for validator repairs and mutation-pool sampling rules while
retaining enforcement behavior, semantic strength, scope, source location, and
repair strategy. The corpus is evaluation data and a manual baseline; it must
not be supplied to the framework-side inference component during testing.

## 6. Experimental baselines

Compare against:

1. unrestricted random mutation;
2. type-only mutation;
3. manually maintained lm-sv constraints;
4. static extraction only;
5. dynamic probing only;
6. static + active probing + synthesis (ConfigFuzz);
7. optional LLM-only proposal baseline.

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

## 8. Immediate tasks

Completed in the initial prototype:

- static inventory over the lm-sv baseline;
- seven-way runtime outcome classification that separates explicit invalidity,
  resource failure, infrastructure failure, unexplained failure, and potential bugs;
- isolated, timeout-bounded subprocess probing;
- integer, float, Boolean, string, and enum candidate generation;
- first Z3 templates for bounds, enums, divisibility, and contextual relations;
- a real lm-sv validator adapter and hidden-size recovery experiment.
- a normalized corpus of reviewed Task1 validator rules and Task6 mutation-pool rules.
- a strict static scanner that removes known name-matching false positives,
  normalizes guarded relations, and scans multiple parameters in parallel.
- bounded interprocedural propagation for direct helper-function calls.
- argparse, dataclass, `Literal`, field-metadata, and YAML schema extraction.
- a versioned Megatron-LM framework-side scan at commit `42460a7`.
- an explicit dependency hypergraph with qualified configuration paths,
  directional impact queries, condition evaluation, and edge status.
- a bounded joint-mutation planner that repairs simple divisibility,
  alignment, equality, bound, and Boolean dependencies.
- a status-aware Z3 joint solver over impacted graph edges, with confirmed and
  environment-scoped edges enforced as hard constraints and unconfirmed evidence
  treated as weighted soft constraints.
- runtime feedback attribution that distinguishes consistency from necessity,
  requires provenance-matched paired interventions for confirmation, marks valid
  counterexamples as scope disputes, preserves potential-bug inputs, and
  deduplicates repeated feedback batches.
- a solver-backed paired-intervention designer that produces minimally
  different satisfying, violating, and repaired configurations while
  preserving other confirmed edges.

Next tasks:

1. Attach model, backend, hardware, and execution-stage scope to graph edges.
2. Add return-value/object-field propagation and shell/documentation adapters.
3. Use `_apply_fix` arguments and repair strategies to rank candidate semantics.
4. Evaluate scanner and graph precision/recall against the reviewed corpus.
5. Execute designed interventions and match runtime rejection provenance to the
   target edge automatically.
6. Add adaptive edge selection, valid-boundary objectives, and convergence
   criteria.
7. Model memory, device topology, and backend capability as scoped resource edges.
8. Integrate solver-generated plans into the lm-sv mutation and execution path.
