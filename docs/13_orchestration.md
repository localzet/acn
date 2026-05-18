# Experiment Orchestration

ACN orchestration coordinates existing modules without moving their responsibilities.

## Component Flow

```mermaid
flowchart TD
    ExperimentManager --> EvolutionPipeline
    EvolutionPipeline --> StageTransitionManager
    EvolutionPipeline --> TrainingSession
    TrainingSession --> Trainer["Trainer or external runner"]
    EvolutionPipeline --> Versioning["TrainingVersionRepository"]
    EvolutionPipeline --> Controller["AdaptiveController"]
    Controller --> DecisionExecutor
    DecisionExecutor --> Citadel["CitadelSafetyLayer"]
    DecisionExecutor --> RollbackCoordinator
    RollbackCoordinator --> Citadel
    RollbackCoordinator --> Versioning
```

## Event Flow

```mermaid
sequenceDiagram
    participant M as ExperimentManager
    participant P as EvolutionPipeline
    participant S as StageTransitionManager
    participant T as TrainingSession
    participant V as Version Repository
    participant C as Controller
    participant D as DecisionExecutor
    participant Z as Citadel

    M->>P: run experiment
    P->>S: start stage
    P->>T: run stage
    T-->>P: checkpoint + metrics
    P->>V: create checkpoint + commit
    P->>S: complete stage
    P->>C: decide next action
    C-->>P: controller decision
    P->>D: execute decision
    D->>Z: validate critical action
    D->>V: branch or rollback when allowed
```

## Boundaries

- `TrainingSession` adapts a synchronous stage runner to orchestration.
- `EvolutionPipeline` owns ordering and state transitions.
- `DecisionExecutor` validates and routes controller decisions.
- `RollbackCoordinator` is the only rollback executor.
- `Trainer`, evaluator and controller remain decoupled from persistence and orchestration.
