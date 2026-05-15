# System Architecture

Frontend (React)
↓
FastAPI Gateway
↓
Citadel Decision Layer
↓
Redis Queue
↓
ACN Worker
↓
PostgreSQL + MinIO + MLflow

Architecture style:
Modular Monolith + Worker.