# Transaction and Integrity Model

Date: 2026-05-18

ACN uses synchronous SQLAlchemy repositories and explicit unit-of-work boundaries for Stage 1.

## Goals

- Keep experiment state referentially linked to training commits.
- Keep checkpoint registration, commit creation, branch updates and experiment updates atomic.
- Prevent failed rollback or commit flows from leaving orphaned experiment state.
- Preserve repository boundaries: repositories flush changes but do not own commits.

## Database Integrity

```mermaid
erDiagram
    training_commits ||--o{ experiments : current_commit_id
    training_commits ||--o{ experiments : best_commit_id
    training_commits ||--o{ experiment_stage_executions : commit_id
    experiments ||--o{ experiment_stage_executions : experiment_id
```

Foreign keys:

- `experiments.current_commit_id -> training_commits.id`
- `experiments.best_commit_id -> training_commits.id`
- `experiment_stage_executions.commit_id -> training_commits.id`
- `experiment_stage_executions.experiment_id -> experiments.id`

Indexes:

- `training_commits.branch_id`
- `training_commits.parent_commit_id`
- `experiment_stage_executions.experiment_id`
- `experiment_stage_executions.commit_id`
- `experiments.current_commit_id`
- `experiments.best_commit_id`

## Unit Of Work

The unit-of-work abstraction lives in `packages/acn/src/acn/infrastructure/uow.py`.

```mermaid
flowchart TD
    App["Application service / pipeline"] --> UOW["UnitOfWork.transaction"]
    UOW --> Repos["Repositories sharing one Session"]
    Repos --> Flush["flush changes"]
    UOW --> Commit["commit on success"]
    UOW --> Rollback["rollback on exception"]
```

Repositories cooperate with shared transactions by flushing only. They do not call `commit`.

## Orchestration Boundary

`EvolutionPipeline` can receive a `UnitOfWork`. When configured, each stage's mutation block is atomic:

```mermaid
sequenceDiagram
    participant P as EvolutionPipeline
    participant U as UnitOfWork
    participant V as VersionRepository
    participant S as StateRepository
    participant D as DecisionExecutor

    P->>U: transaction
    U->>V: create checkpoint
    U->>V: create commit and update branch head
    U->>S: complete stage execution
    U->>S: update experiment commit pointers
    U->>D: execute branch or rollback decision
    U-->>P: commit on success / rollback on failure
```

Training itself remains outside the DB transaction. The transaction starts after a stage returns checkpoint metadata and metrics.

## Rollback Safety

Rollback now supports guarded branch-head updates:

- callers pass `current_commit_id`;
- repository verifies the branch head still matches the expected current commit;
- stale rollback attempts raise `BranchHeadConflictError`;
- failed rollback validation or artifact restoration happens before branch head movement.

```mermaid
sequenceDiagram
    participant R as RollbackCoordinator
    participant C as Citadel
    participant A as ArtifactStore
    participant V as VersionRepository

    R->>C: validate rollback
    C-->>R: allowed
    R->>A: load and checksum-verify artifact
    A-->>R: payload
    R->>V: rollback_branch(expected_head_commit_id=current_commit_id)
    V-->>R: branch updated or BranchHeadConflictError
```

## Migration

Migration file:

- `infra/db/alembic/versions/20260518_0004_add_experiment_commit_integrity.py`

This migration adds foreign keys and indexes without changing external API contracts.

## Non-Goals

- No distributed transactions.
- No event sourcing.
- No async database layer.
- No queue framework introduced in this step.
