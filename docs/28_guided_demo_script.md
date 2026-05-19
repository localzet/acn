# Guided ACN Demo Script

Audience: non-ML specialists, managers and reviewers.

Target duration: 5 minutes.

Core message:

> ACN is an adaptive training control system. It helps a model learn, detects when training goes
> wrong, restores a stable version and leaves a clear history of what happened.

## Setup

Open:

```text
http://localhost:5173
```

Choose `Live Demo`.

For the full infrastructure-backed version, also open:

- MLflow: <http://localhost:5000>
- Runtime health: <http://localhost:8000/api/v1/runtime/health>

## Demo Flow

### 1. What ACN Is

What the viewer sees:

- A live training dashboard.
- A simple visual task: distinguish `airplane` vs `ship`.
- Big labels: epoch, accuracy, loss, rollback count.

What is happening technically:

- A small CNN is trained locally.
- ACN watches metrics and records model versions.

Why it matters:

- The value is not one model. The value is controlled model evolution.

Presenter line:

> ACN is like a control room for model training. It watches the model learn and keeps recoverable
> versions as it goes.

### 2. Initial Poor Predictions

What the viewer sees:

- Early validation images.
- Some predictions are wrong or low-confidence.

What is happening technically:

- The model starts with random weights.
- It has not learned the image patterns yet.

Why it matters:

- The audience can see learning begin from an imperfect state.

Presenter line:

> At the start, the model is guessing. That is normal. The point is that ACN shows us whether the
> learning is improving or getting worse.

### 3. Learning Progress

What the viewer sees:

- Accuracy rises.
- Loss falls.
- Checkpoints appear in the timeline.

What is happening technically:

- Training updates model weights.
- ACN records checkpoints.

Visual explanation:

```text
Checkpoint = save point of model state
```

Why it matters:

- If training later breaks, we have a known-good model state to restore.

### 4. Degradation Event

What the viewer sees:

- A bad training stage appears.
- Validation loss worsens.
- Event feed says degradation was detected.

What is happening technically:

- The demo intentionally applies a learning-rate spike and corrupted labels.
- ACN detects that validation quality degraded.

Why it matters:

- In manual ML work, a person often notices this later by inspecting logs.
- ACN notices during the run.

Presenter line:

> This is the moment where ordinary training would keep damaging the model unless someone is
> watching carefully.

### 5. Rollback

What the viewer sees:

- Rollback count increments.
- Event feed says rollback was initiated and checkpoint restored.
- Training resumes from a stable model.

What is happening technically:

- ACN restores the last stable checkpoint.
- The learning rate is reduced for recovery.

Visual explanation:

```text
Rollback = restore the last stable model
```

Why it matters:

- Recovery is faster and safer than manually restarting from scratch.

### 6. Branch / Alternative Path

What the viewer sees:

- Current branch is shown as `main` or `visual-demo`.
- Checkpoint history shows model versions.

What is happening technically:

- A branch is an experiment line.
- Future versions can explore alternative settings without losing the stable path.

Visual explanation:

```text
Branch = alternative experiment path
```

Why it matters:

- Teams can compare training directions while keeping a traceable history.

### 7. Final Inference

What the viewer sees:

- Upload image panel.
- Prediction class.
- Confidence.
- Checkpoint ID and latency.
- Early model vs selected model comparison.

What is happening technically:

- The trained checkpoint is loaded for inference.
- The same image can be tested against different model versions.

Why it matters:

- This proves adaptive training produced a usable model, not just charts.

## Manual ML Workflow vs ACN Workflow

| Manual ML Workflow | ACN Workflow |
| --- | --- |
| Engineer watches logs manually. | ACN watches metrics continuously. |
| Bad run may be noticed late. | Degradation is detected during training. |
| Recovery often means restarting. | Rollback restores the last stable checkpoint. |
| Experiment history is scattered. | Commits, checkpoints, decisions and events are traceable. |
| Comparing versions is manual. | Model versions can be selected and tested. |

## Final Showcase Checklist

Show these before ending:

- Best model version.
- Best accuracy.
- Rollback count.
- Total checkpoints.
- Training timeline.
- Final prediction examples.
- MLflow run ID if full stack is enabled.
- MinIO artifact URI if full stack is enabled.

## Why This Matters

ACN reduces manual experimentation by automating observation and recovery decisions.

ACN makes training safer because unstable states can be rolled back.

ACN improves traceability because every important model version has a checkpoint, event and
decision history.

ACN makes adaptive recovery visible to non-specialists: the audience can see the model fail,
recover and improve.

## Closing Line

> ACN turns model training from a fragile manual process into a controlled, observable and
> recoverable workflow.
