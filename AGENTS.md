# AGENTS.md

## Project
Adaptive Core Network (ACN)

## Goal
Build a production-grade adaptive neural training framework with:
- version-controlled training
- rollback
- branching
- continual learning
- adaptive policies
- human override
- experiment tracking

## Architecture Rules
- Use modular monolith architecture
- Do NOT introduce Kubernetes
- Do NOT introduce microservices unless explicitly requested
- Prefer clean architecture
- Prefer composition over inheritance
- Strict typing required
- Production-grade code only

## Tech Stack
- Python 3.12
- PyTorch
- FastAPI
- PostgreSQL
- Redis
- React
- TypeScript
- Docker Compose

## Code Style
- Fully typed Python
- Ruff-compatible
- Black-compatible
- Small focused modules
- No god classes
- No magic constants
- Use dataclasses or Pydantic models

## Constraints
- Target hardware:
  - RTX 3060 Laptop
  - 40GB RAM
- Optimize for iterative experimentation
- Avoid extremely heavy models

## Important
- Never rewrite unrelated files
- Never remove existing architecture without approval
- Always explain architecture decisions
- Always provide tests
- Always update README/docs if architecture changes

## Development Workflow
- One subsystem per task
- Minimal diff per commit
- Build incrementally