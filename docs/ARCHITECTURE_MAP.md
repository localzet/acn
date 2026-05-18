# ACN Architecture Map

Date: 2026-05-18  
Scope: current ACN modular monolith.

This document is a graph-first map of ACN dependencies and runtime flows. It is intentionally concise and complements the deeper audit in `docs/ARCHITECTURE_AUDIT.md`.

## 1. Package Dependency Graph

```mermaid
flowchart TD
    Root["adaptive-core-network"]

    Root --> Apps["apps"]
    Root --> Packages["packages/acn"]
    Root --> Infra["infra"]
    Root --> Configs["configs"]
    Root --> Scripts["scripts"]
    Root --> Tests["tests"]
    Root --> Docs["docs"]

    Apps --> API["apps/api"]
    Apps --> Worker["apps/worker"]
    Apps --> Web["apps/web"]

    API --> CoreConfig["acn.config"]
    API --> APIRouters["acn_api.dashboard"]

    Worker --> CoreConfig

    Web --> APIContract["FastAPI REST/SSE/WS contract"]

    Packages --> Training["acn.training"]
    Packages --> Continual["acn.continual"]
    Packages --> Controller["acn.controller"]
    Packages --> Versioning["acn.versioning"]
    Packages --> Citadel["acn.citadel"]
    Packages --> Orchestration["acn.orchestration"]
    Packages --> Experiments["acn.experiments"]
    Packages --> CoreConfig

    Infra --> Alembic["infra/db/alembic"]
    Infra --> Docker["infra/docker"]

    Scripts --> TrainingScripts["training/controller/citadel/continual/orchestration/demo/experiment scripts"]
    Tests --> TestSuites["unit + subsystem + integration tests"]
```

Explanation:
- `apps/*` are deployable/process boundaries.
- `packages/acn/src/acn/*` contains the modular monolith core.
- `infra/*` contains local runtime infrastructure and migrations.
- `scripts/*` are executable examples and reproducible workflows.
- `tests/*` validate subsystem boundaries and selected integration paths.

## 2. Module Dependency Graph

```mermaid
flowchart TD
    API["apps/api/acn_api"] --> Config["acn.config"]
    API --> DashboardRouter["acn_api.dashboard"]

    Worker["apps/worker/acn_worker"] --> Config

    Web["apps/web"] --> DashboardREST["GET /api/v1/dashboard/snapshot"]
    Web --> DashboardSSE["GET /api/v1/dashboard/events"]
    Web --> DashboardWS["WS /api/v1/dashboard/ws"]
    Web --> OverrideREST["POST /api/v1/overrides"]

    Experiments["acn.experiments"] --> Continual["acn.continual"]
    Experiments --> Controller["acn.controller"]
    Experiments --> Citadel["acn.citadel"]
    Experiments --> Versioning["acn.versioning"]

    Orchestration["acn.orchestration"] --> Continual
    Orchestration --> Controller
    Orchestration --> Citadel
    Orchestration --> Versioning

    Citadel --> ControllerDomain["acn.controller.domain"]
    Citadel --> VersionRepo["acn.versioning.repository"]
    Citadel --> VersionModels["acn.versioning.models"]

    Controller --> ControllerDomain
    NeuralController["acn.controller.neural"] --> ControllerPolicies["acn.controller.policies"]
    NeuralController --> ControllerDomain

    Continual --> ContinualDataset["acn.continual.dataset"]
    Continual --> ContinualStage["acn.continual.stage"]
    Continual --> Replay["acn.continual.replay"]
    Continual --> Stream["acn.continual.stream"]

    Training["acn.training"] --> TrainingConfig["acn.training.config"]
    Training --> Checkpointing["acn.training.checkpointing"]
    Training --> Optimizers["acn.training.optimizers"]
    Training --> Schedulers["acn.training.schedulers"]
    Training --> Freezing["acn.training.freezing"]

    Versioning --> VersionDomain["acn.versioning.domain"]
    Versioning --> VersionModels

    Orchestration --> OrchModels["acn.orchestration.models"]
    Orchestration --> OrchRepo["acn.orchestration.repository"]

    Config --> Settings["acn.config.settings"]
    Config --> Logging["acn.config.logging"]
```

Explanation:
- `acn.training` is intentionally isolated from `controller`, `citadel`, `versioning`, `orchestration`, API and UI.
- `acn.controller` is decision-only and does not execute mutations.
- `acn.citadel` validates controller actions against version history.
- `acn.orchestration` is the main composition layer across stage training, versioning, controller decisions and safety validation.

## 3. Clean Architecture Boundary Map

```mermaid
flowchart TB
    subgraph Presentation["Presentation / Process Boundary"]
        Web["React dashboard"]
        API["FastAPI API"]
        Worker["Worker entrypoint"]
        Scripts["CLI scripts"]
    end

    subgraph Application["Application Coordination"]
        Orchestration["acn.orchestration"]
        Experiments["acn.experiments"]
    end

    subgraph Domain["Domain Logic"]
        Training["acn.training"]
        Continual["acn.continual"]
        Controller["acn.controller"]
        Citadel["acn.citadel"]
        VersioningContracts["versioning domain + repository protocol"]
    end

    subgraph Infrastructure["Infrastructure Adapters"]
        SQLAModels["SQLAlchemy models/repositories"]
        Alembic["Alembic migrations"]
        Docker["Docker Compose services"]
        MLflow["MLflow server"]
        MinIO["MinIO"]
        Redis["Redis"]
        Postgres["PostgreSQL"]
    end

    Web --> API
    API --> Application
    Worker --> Application
    Scripts --> Application
    Application --> Domain
    Citadel --> VersioningContracts
    Domain --> SQLAModels
    SQLAModels --> Postgres
    Alembic --> Postgres
    Docker --> Postgres
    Docker --> Redis
    Docker --> MLflow
    MLflow --> MinIO
```

Explanation:
- The intended dependency direction is inward from process/adapters to application/domain.
- Current code mostly follows this, with SQLAlchemy repositories located inside domain-specific packages.
- Redis, MLflow and MinIO are provisioned infrastructure but are not fully used by application code yet.

## 4. Runtime Interaction Graph

```mermaid
flowchart LR
    Operator["Operator / researcher"] --> Browser["Browser"]
    Browser --> Web["React dashboard"]

    Web -->|REST| API["FastAPI API"]
    Web -->|SSE| API
    Web -->|WebSocket fallback| API

    API --> Settings["acn.config.Settings"]
    API --> DashboardRouter["dashboard router"]

    Worker["ACN worker"] --> Settings
    Worker -. future .-> Redis["Redis queue/events"]
    Worker -. future .-> Orchestration["EvolutionPipeline"]

    Orchestration --> TrainingSession["TrainingSession"]
    TrainingSession --> StageRunner["StageTrainingRunner"]
    StageRunner --> Trainer["Trainer"]

    Orchestration --> VersionRepo["TrainingVersionRepository"]
    Orchestration --> Controller["AdaptiveController"]
    Orchestration --> DecisionExecutor["DecisionExecutor"]
    DecisionExecutor --> Citadel["CitadelSafetyLayer"]
    DecisionExecutor --> Rollback["RollbackCoordinator"]
    Rollback --> Citadel
    Rollback --> VersionRepo

    VersionRepo --> Postgres["PostgreSQL"]
    Citadel --> AuditRepo["AuditLogRepository"]
    AuditRepo --> Postgres
    Orchestration --> ExperimentRepo["ExperimentStateRepository"]
    ExperimentRepo --> Postgres

    Trainer -. planned .-> MLflow["MLflow"]
    MLflow -. artifacts .-> MinIO["MinIO"]
```

Explanation:
- Solid arrows represent implemented or directly available interactions.
- Dotted arrows represent intended but incomplete runtime integration.
- The worker process currently logs startup only and does not consume Redis jobs.
- The dashboard backend currently emits empty contract data, not repository-backed state.

## 5. Training Flow Graph

```mermaid
flowchart TD
    Model["torch.nn.Module"] --> Trainer["Trainer"]
    Criterion["criterion: nn.Module"] --> Trainer
    OptimizerConfig["OptimizerConfig"] --> OptimizerFactory["build_optimizer"]
    OptimizerFactory --> Optimizer["torch.optim.Optimizer"]
    Optimizer --> Trainer

    SchedulerConfig["SchedulerConfig"] --> SchedulerFactory["build_scheduler"]
    SchedulerFactory --> Scheduler["LRScheduler | None"]
    Scheduler --> Trainer

    TrainerConfig["TrainerConfig"] --> Trainer
    TrainLoader["train DataLoader"] --> TrainEpoch["Trainer.train_epoch"]
    ValidationLoader["validation DataLoader"] --> Validate["Trainer.validate"]

    Trainer --> Device["device resolution: cuda if available else cpu"]
    Trainer --> AMP["autocast + GradScaler when CUDA mixed_precision=true"]

    TrainEpoch --> BatchParse["parse mapping/tuple batch"]
    BatchParse --> Forward["model(inputs)"]
    Forward --> Loss["criterion(outputs, targets)"]
    Loss --> Backward["scaled backward"]
    Backward --> Clip["optional grad clipping"]
    Clip --> Step["optimizer step"]
    Step --> TrainMetrics["EpochMetrics train"]

    Validate --> ValForward["inference forward"]
    ValForward --> ValMetrics["EpochMetrics validation"]

    TrainMetrics --> History["TrainingHistory"]
    ValMetrics --> History
    Trainer --> CheckpointManager["CheckpointManager"]
    CheckpointManager --> CheckpointFile["epoch-XXXX.pt"]
```

Explanation:
- The trainer is reusable and API-agnostic.
- Checkpoints include model, optimizer, scheduler, scaler and trainer state.
- The trainer does not currently emit MLflow metrics or orchestration events.

## 6. Continual Learning Flow Graph

```mermaid
flowchart TD
    SourceMap["Mapping[str, IDataSource]"] --> Scenario["ContinualLearningScenario"]
    StageConfigs["DatasetStageConfig[]"] --> Scenario

    Scenario --> Stage["DatasetStage"]
    Stage --> ClassIds["class_ids"]
    Stage --> Introduced["introduced_class_ids"]
    Stage --> DomainShift["domain_shift source override"]
    Stage --> ReplayRatio["replay_ratio"]

    Scenario --> Source["IDataSource.build_dataset"]
    Source --> ClassFiltered["ClassFilteredDataset"]
    DomainShift --> SyntheticSource["SyntheticDomainShiftSource"]
    SyntheticSource --> Transformed["TransformedImageDataset"]

    ClassFiltered --> StageDataset["stage dataset"]
    Transformed --> StageDataset

    ReplayBuffer["ReplayBuffer"] --> ReplayDataset["ReplayDataset"]
    ReplayRatio --> ReplaySampleCount["max replay samples"]
    ReplayDataset --> Combined["CombinedImageDataset"]
    StageDataset --> Combined

    Combined --> TrainerRunner["trainer/stage runner"]
    TrainerRunner --> Predictions["targets + predictions"]
    Predictions --> ForgettingEvaluator["ForgettingEvaluator"]
    ForgettingEvaluator --> ContinualMetrics["old retention, new adaptation, forgetting, latency"]
```

Explanation:
- `IDataSource` decouples scenario logic from concrete datasets.
- Replay is composed as a normal Dataset, so the trainer remains decoupled.
- Stream sources can produce Dataset snapshots but are not part of real-time training yet.

## 7. Version Store Graph

```mermaid
flowchart TD
    Checkpoint["StableCheckpointRecord"] --> Commit["CommitRecord"]
    Branch["BranchRecord"] --> Commit
    ParentCommit["parent CommitRecord"] --> Commit
    Commit --> CommitGraphNode["CommitGraphNode"]
    ParentCommit --> CommitGraphEdge["CommitGraphEdge"]
    CommitGraphNode --> CommitGraph["CommitGraph"]
    CommitGraphEdge --> CommitGraph

    Repository["SqlAlchemyTrainingVersionRepository"] --> CheckpointModel["StableCheckpointModel"]
    Repository --> BranchModel["BranchModel"]
    Repository --> CommitModel["CommitModel"]

    BranchModel -->|head_commit_id| CommitModel
    BranchModel -->|base_commit_id| CommitModel
    CommitModel -->|parent_commit_id| CommitModel
    CommitModel -->|checkpoint_id| CheckpointModel
```

Explanation:
- Versioning models Git-like lineage, not Git storage.
- Stable checkpoints are immutable through SQLAlchemy update/delete guards.
- Rollback is implemented as moving a branch head to a reachable ancestor commit.

## 8. Orchestration Graph

```mermaid
flowchart TD
    ExperimentManager["ExperimentManager"] --> StateRepo["ExperimentStateRepository"]
    ExperimentManager --> VersionRepo["TrainingVersionRepository"]

    EvolutionPipeline["EvolutionPipeline"] --> StateRepo
    EvolutionPipeline --> VersionRepo
    EvolutionPipeline --> TrainingSession["TrainingSession"]
    EvolutionPipeline --> AdaptiveController["AdaptiveController"]
    EvolutionPipeline --> DecisionExecutor["DecisionExecutor"]
    EvolutionPipeline --> StageTransitionManager["StageTransitionManager"]

    TrainingSession --> StageRunner["StageTrainingRunner protocol"]
    StageRunner --> StageResult["StageTrainingResult"]

    StageTransitionManager --> StageExecution["StageExecutionRecord"]
    StageResult --> CheckpointRegistration["create checkpoint"]
    CheckpointRegistration --> CommitCreation["create commit"]
    CommitCreation --> ExperimentUpdate["update current/best commit"]

    AdaptiveController --> ControllerDecision["ControllerDecision"]
    ControllerDecision --> DecisionExecutor

    DecisionExecutor --> Citadel["CitadelSafetyLayer"]
    DecisionExecutor --> RollbackCoordinator["RollbackCoordinator"]
    DecisionExecutor --> BranchCreate["create experimental branch"]
    RollbackCoordinator --> Citadel
    RollbackCoordinator --> VersionRepo
```

Explanation:
- `EvolutionPipeline` is the central application coordinator.
- `TrainingSession` adapts an async stage runner; it does not know trainer internals.
- `DecisionExecutor` is responsible for routing controller decisions into safe mutations.

## 9. Orchestration Sequence

```mermaid
sequenceDiagram
    participant M as ExperimentManager
    participant P as EvolutionPipeline
    participant S as StageTransitionManager
    participant T as TrainingSession
    participant V as VersionRepository
    participant C as AdaptiveController
    participant D as DecisionExecutor
    participant Z as Citadel

    M->>V: get_branch or create_branch
    M-->>P: ExperimentRecord
    P->>S: start_stage(experiment, stage)
    S-->>P: StageExecutionRecord
    P->>T: await run_stage(stage)
    T-->>P: StageTrainingResult
    P->>V: create_checkpoint(uri, hash, metadata)
    V-->>P: StableCheckpointRecord
    P->>V: create_commit(branch, checkpoint, metrics)
    V-->>P: CommitRecord
    P->>S: complete_stage(execution_id, commit_id, metrics)
    P->>C: decide(metric_history, TrainingContext)
    C-->>P: ControllerDecision
    P->>D: execute(decision)
    D->>Z: validate_action
    Z-->>D: validation result
    D-->>P: DecisionExecutionResult
```

Explanation:
- The sequence is implemented for orchestration tests and synthetic pipelines.
- Real worker-backed execution and persistent dashboard events are not implemented yet.

## 10. Controller Decision Graph

```mermaid
flowchart TD
    Metrics["MetricPoint history"] --> Analyze["RuleBasedAdaptivePolicy.analyze"]
    Context["TrainingContext"] --> Decide["RuleBasedAdaptivePolicy.decide"]
    Analyze --> Signals["ControllerSignals"]

    Signals --> Degradation{"degradation?"}
    Degradation -->|yes| DegradationAction{"degradation_action == rollback?"}
    DegradationAction -->|yes| RollbackDecision["ROLLBACK target_commit_id=best_commit_id"]
    DegradationAction -->|no| DecreaseLR["DECREASE_LEARNING_RATE"]

    Degradation -->|no| Overfitting{"overfitting?"}
    Overfitting -->|yes| Frozen{"already frozen or configured LR decrease?"}
    Frozen -->|yes| DecreaseLR
    Frozen -->|no| Freeze["FREEZE_LAYERS layer_selector=features"]

    Overfitting -->|no| StableFrozen{"stable improvement and frozen?"}
    StableFrozen -->|yes| Unfreeze["UNFREEZE_LAYERS layer_selector=all"]

    StableFrozen -->|no| Underfitting{"underfitting?"}
    Underfitting -->|yes| IncreaseLR["INCREASE_LEARNING_RATE"]

    Underfitting -->|no| Plateau{"plateau?"}
    Plateau -->|yes| PlateauAction{"plateau_action == decrease LR?"}
    PlateauAction -->|yes| DecreaseLR
    PlateauAction -->|no| Branch["CREATE_EXPERIMENTAL_BRANCH source_commit_id=current_commit_id"]

    Plateau -->|no| Continue["CONTINUE_TRAINING"]

    RollbackDecision --> Decision["ControllerDecision"]
    DecreaseLR --> Decision
    Freeze --> Decision
    Unfreeze --> Decision
    IncreaseLR --> Decision
    Branch --> Decision
    Continue --> Decision
```

Explanation:
- Rule-based decisions are explainable and priority-ordered.
- Decisions are not executed by the controller.
- Citadel and `DecisionExecutor` handle mutation safety and execution routing.

## 11. Neural Controller Graph

```mermaid
flowchart TD
    Metrics["MetricPoint history"] --> FeatureBuilder["build_policy_features"]
    Context["TrainingContext"] --> FeatureBuilder
    State["NeuralControllerState"] --> FeatureBuilder

    FeatureBuilder --> Features["10 scalar features"]
    Features --> PolicyNetwork["PolicyNetwork MLP"]
    PolicyNetwork --> Probabilities["softmax probabilities"]
    Probabilities --> Confidence{"confidence >= threshold?"}

    Confidence -->|yes| NeuralDecision["ControllerDecision from neural action"]
    Confidence -->|no| Fallback["RuleBasedAdaptivePolicy.decide"]
    Fallback --> FallbackDecision["ControllerDecision with fallback reason"]

    NeuralDecision --> ActionParams["action-specific parameters"]
    FallbackDecision --> Output["final decision"]
    ActionParams --> Output
```

Explanation:
- Neural policy is a small MLP optimized for lightweight offline training/inference.
- It falls back to rule-based decisions when confidence is low.
- It is not currently the default `AdaptiveController` implementation.

## 12. Citadel Validation Graph

```mermaid
flowchart TD
    Request["CitadelActionRequest"] --> Critical{"action in critical_actions?"}
    Critical -->|no| AllowNonCritical["ALLOW non-critical"]

    Critical -->|yes| ActionType{"action type"}
    ActionType -->|ROLLBACK| ValidateRollback["validate target_commit_id and reachability"]
    ActionType -->|LR change| ValidateLR["validate numeric LR within min/max"]
    ActionType -->|freeze/unfreeze| ValidateLayer["validate non-empty layer_selector"]
    ActionType -->|create branch| ValidateBranch["validate source_commit_id reachability"]

    ValidateRollback --> Reasons{"validation reasons?"}
    ValidateLR --> Reasons
    ValidateLayer --> Reasons
    ValidateBranch --> Reasons

    Reasons -->|none| AllowCritical["ALLOW critical action"]
    Reasons -->|has reasons| Override{"valid override and action overrideable?"}
    Override -->|yes| OverrideApproved["ALLOW override_approved"]
    Override -->|no| Deny["DENY, may require override"]

    AllowNonCritical --> Audit["record audit"]
    AllowCritical --> Audit
    OverrideApproved --> Audit
    Deny --> Audit
```

Explanation:
- Citadel is the safety gate for critical controller actions.
- Current enforcement depends on callers routing actions through Citadel.
- Audit logs are supported in memory and via SQLAlchemy.

## 13. Dashboard Interaction Graph

```mermaid
flowchart TD
    App["App.tsx"] --> UseDashboardData["useDashboardData"]
    UseDashboardData --> FetchSnapshot["fetchDashboardSnapshot"]
    UseDashboardData --> LiveUpdates["connectLiveUpdates"]
    UseDashboardData --> SubmitOverride["submitOverride"]

    FetchSnapshot --> SnapshotEndpoint["GET /api/v1/dashboard/snapshot"]
    LiveUpdates --> EventSource{"EventSource available?"}
    EventSource -->|yes| SSE["GET /api/v1/dashboard/events"]
    EventSource -->|no| WS["WS /api/v1/dashboard/ws"]
    SubmitOverride --> OverrideEndpoint["POST /api/v1/overrides"]

    UseDashboardData --> SnapshotState["DashboardSnapshot state"]
    SnapshotState --> CommitGraph["CommitGraphView"]
    SnapshotState --> BranchGraph["BranchGraphView"]
    SnapshotState --> Metrics["MetricsTimelineView"]
    SnapshotState --> Inspector["ExperimentInspectorView"]
    SnapshotState --> Decisions["ControllerDecisionsView"]
    SnapshotState --> Rollback["RollbackHistoryView"]
    SnapshotState --> Logs["LiveLogsView"]
    SnapshotState --> OverrideConsole["OverrideConsole"]
```

Explanation:
- Frontend is contract-driven and typed with `DashboardSnapshot`.
- Current backend provides empty contract data.
- Demo mode bypasses live API data and uses deterministic preset playback.

## 14. Database Map

```mermaid
erDiagram
    stable_checkpoints {
        string id PK
        text uri UK
        string content_hash UK
        int size_bytes
        json metadata
        datetime created_at
    }

    training_branches {
        string id PK
        string name UK
        string head_commit_id FK
        string base_commit_id FK
        json metadata
        datetime created_at
    }

    training_commits {
        string id PK
        string branch_id FK
        string checkpoint_id FK
        string parent_commit_id FK
        text message
        string authored_by
        json metrics
        json metadata
        datetime created_at
    }

    citadel_audit_logs {
        string id PK
        string action
        string actor
        string branch_name
        string decision
        json reasons
        json parameters
        string override_by
        text override_reason
        string override_ticket_id
        datetime created_at
    }

    experiments {
        string id PK
        string name UK
        string branch_name
        string status
        string current_stage_id
        string current_commit_id
        string best_commit_id
        json metadata
        datetime created_at
        datetime updated_at
    }

    experiment_stage_executions {
        string id PK
        string experiment_id
        string stage_id
        string status
        string commit_id
        json metrics
        datetime started_at
        datetime completed_at
    }

    stable_checkpoints ||--o{ training_commits : checkpoint_id
    training_branches ||--o{ training_commits : branch_id
    training_commits ||--o{ training_commits : parent_commit_id
    training_commits ||--o{ training_branches : head_or_base
    experiments ||--o{ experiment_stage_executions : experiment_id
```

Explanation:
- Versioning schema has real relational constraints.
- Citadel audit and experiment state are less relationally strict.
- Experiment commit/branch references are string fields, not enforced FKs.

## 15. Infrastructure Map

```mermaid
flowchart TD
    Compose["docker-compose.yml"] --> APIService["api"]
    Compose --> WorkerService["worker"]
    Compose --> WebService["web"]
    Compose --> Postgres["postgres:16-alpine"]
    Compose --> Redis["redis:7-alpine"]
    Compose --> MinIO["minio"]
    Compose --> BucketInit["create-minio-bucket"]
    Compose --> MLflow["mlflow"]

    APIService --> PythonDocker["infra/docker/python.Dockerfile"]
    WorkerService --> PythonDocker
    WebService --> WebDocker["infra/docker/web.Dockerfile"]
    MLflow --> MLflowDocker["infra/docker/mlflow.Dockerfile"]

    APIService --> Postgres
    APIService --> Redis
    WorkerService --> Postgres
    WorkerService --> Redis
    BucketInit --> MinIO
    MLflow --> Postgres
    MLflow --> MinIO
```

Explanation:
- Compose provides the full intended local stack.
- Redis, MLflow and MinIO are infrastructure-ready but not materially used by core ACN flows.
- The web service currently serves Vite, not a production static build.

## 16. Testing Map

```mermaid
flowchart TD
    Tests["tests"] --> Unit["unit"]
    Tests --> TrainingTests["training"]
    Tests --> VersioningTests["versioning"]
    Tests --> CitadelTests["citadel"]
    Tests --> ControllerTests["controller"]
    Tests --> ContinualTests["continual"]
    Tests --> OrchestrationTests["orchestration"]
    Tests --> APITests["api"]
    Tests --> WorkerTests["worker"]
    Tests --> ExperimentTests["experiments"]
    Tests --> DemoTests["demo"]
    Tests --> IntegrationTests["integration"]

    TrainingTests --> Training["acn.training"]
    VersioningTests --> Versioning["acn.versioning"]
    CitadelTests --> Citadel["acn.citadel"]
    ControllerTests --> Controller["acn.controller"]
    ContinualTests --> Continual["acn.continual"]
    OrchestrationTests --> Orchestration["acn.orchestration"]
    APITests --> API["apps/api"]
    WorkerTests --> Worker["apps/worker"]
    ExperimentTests --> Experiments["acn.experiments"]
    IntegrationTests --> CrossModule["versioning + rollback + branch consistency"]
```

Explanation:
- Tests are organized by subsystem and mostly isolated.
- Current coverage is high for implemented code, but not proof of production runtime readiness.
- Missing areas: frontend tests, PostgreSQL integration tests, Redis/worker queue tests, MLflow/MinIO tests, GPU tests.

## 17. Critical Architectural Edges

These edges matter most for future changes:

```text
Trainer must not depend on:
  API, UI, controller, Citadel, versioning, orchestration.

Controller must not execute:
  rollback, branch creation, checkpoint mutation, optimizer mutation directly.

Citadel should guard:
  rollback, branch creation, freeze/unfreeze, LR changes, checkpoint registration.

Orchestration may coordinate:
  training session, version repository, controller, Citadel, rollback coordinator.

Dashboard should depend on:
  HTTP/event contracts only.

Worker should eventually own:
  long-running training/orchestration execution.
```

## 18. Current Reality vs Intended Map

```mermaid
flowchart LR
    subgraph Implemented["Implemented"]
        Trainer["Trainer"]
        Replay["ReplayBuffer"]
        VersionStore["Version store"]
        Citadel["Citadel validation"]
        RuleController["Rule controller"]
        NeuralController["Neural policy"]
        OrchestrationCore["Orchestration core"]
        DashboardUI["Dashboard UI"]
        DockerStack["Docker stack"]
    end

    subgraph Partial["Partial / stubbed"]
        DashboardAPI["Dashboard API data source"]
        WorkerLoop["Worker execution loop"]
        LiveEvents["Persistent live events"]
        E2ERealTraining["Real E2E continual training"]
        ArtifactStore["Checkpoint artifact store"]
    end

    subgraph Planned["Provisioned / planned"]
        RedisQueue["Redis queue/events"]
        MLflowLogging["MLflow logging"]
        MinIOArtifacts["MinIO artifacts"]
        Auth["Authentication/authorization"]
        DistributedExecution["Distributed execution"]
    end
```

Explanation:
- The codebase has strong internal boundaries.
- Runtime integration is the main unfinished area.
- External reviewers should distinguish implemented contracts from operationally wired behavior.

