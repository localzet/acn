# Future Video Support

Implemented abstraction:

IDataSource
├── ImageDatasetSource
├── VideoFileSource
└── CameraStreamSource

Core modules:
- FrameSampler
- TemporalBuffer
- StreamMetadata

Planned modules:
- DriftDetector
- StreamEvaluator

Design notes:
- Video and camera sources use async frame readers.
- Frame decoding backends are injected, so OpenCV, ffmpeg or camera SDKs can be added later.
- `TemporalBuffer` exposes snapshots as regular PyTorch datasets to preserve trainer compatibility.
- Sampling rate and maximum frame count are configurable per source.
- Unlabeled streams remain valid for ingestion, but training dataset access requires frame labels or a default target.
