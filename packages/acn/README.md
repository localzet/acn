# ACN Shared Package

Internal Python package for shared configuration, domain contracts, services, infrastructure adapters and reusable training primitives.

`acn.training` is intentionally runtime-agnostic: it does not depend on FastAPI, worker internals or frontend code.

`acn.versioning` stores Git-like training evolution metadata through a repository interface and SQLAlchemy persistence.

`acn.citadel` validates critical actions and records audit logs before orchestration code mutates training or versioning state.

`acn.continual` provides dataset-stage orchestration, replay and continual evaluation primitives for adaptive image classification under domain shift.
