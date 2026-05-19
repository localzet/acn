# Inference Showcase

Goal:

> Prove that adaptive training produced a usable model.

The inference showcase is part of `Live Demo`.

## What It Shows

- Drag-and-drop or upload an image.
- Select a checkpoint/model version.
- Run prediction.
- See predicted class, confidence, checkpoint ID, model version and latency.
- Compare early model vs selected/final model.
- See average confidence, average latency and prediction distribution.
- Review prediction history.

## Local Flow

1. Start API and frontend.
2. Open `Live Demo`.
3. Press `Start`.
4. Wait for checkpoints to appear.
5. Upload an image in `Final inference test`.
6. Compare `Early model` against `Latest model`.

## API

Prediction:

```text
POST /api/v1/demo/predict
```

Payload:

```json
{
  "image_data_url": "data:image/png;base64,...",
  "checkpoint_id": "latest"
}
```

Response:

```json
{
  "predictedClass": "ship",
  "confidence": 0.94,
  "checkpointId": "cmt_epoch_09",
  "modelVersion": "selected",
  "latencyMs": 3.5
}
```

Comparison:

```text
POST /api/v1/demo/compare
```

This compares an early checkpoint against the selected checkpoint.

## Service Layer

The reusable inference code lives under:

```text
packages/acn/src/acn/inference/
```

It includes:

- `ImagePreprocessor`
- `InferenceService`
- typed inference result records

The visual demo uses this service with the trained `TinyVisualClassifier` checkpoints.

## Why This Matters

Charts alone do not prove a model is useful. The inference panel shows the result of training:

```text
input image -> selected model version -> prediction + confidence + latency
```

The version selector makes the value visible:

- early model: weaker and less confident;
- final selected model: improved after adaptive training and rollback.

## Export

The demo export writes:

- markdown summary;
- metrics JSON;
- timeline JSON;
- final model info;
- prediction results;
- model metadata;
- screenshot-ready SVG.

Endpoint:

```text
POST /api/v1/demo/export
```
