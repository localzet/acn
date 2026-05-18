import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from acn.continual import (
    CameraStreamSource,
    FrameSampler,
    StreamFrame,
    StreamMetadata,
    TemporalBuffer,
    VideoFileSource,
)
from acn.continual.stage import DatasetSplit
from acn.training.trainer import Batch


async def _frame_reader(metadata: StreamMetadata) -> AsyncIterator[dict[str, object]]:
    fps = metadata.fps or 4.0
    for index in range(6):
        yield {
            "image": torch.full((1, 2, 2), float(index)),
            "target": index % 2,
            "timestamp_seconds": index / fps,
            "frame_index": index,
        }


def test_video_file_source_ingests_sampled_frames_for_trainer_dataset() -> None:
    source = VideoFileSource(
        name="clip-a",
        path=Path("clip-a.mp4"),
        frame_reader=_frame_reader,
        sampler=FrameSampler(sample_rate_hz=2.0),
        stream_metadata=StreamMetadata(source_id="clip-a", source_type="video_file", fps=4.0),
    )

    ingested = asyncio.run(source.ingest())
    dataset = source.build_dataset(split=DatasetSplit.TRAIN)

    assert ingested == 3
    assert len(dataset) == 3
    assert [dataset[index][1] for index in range(len(dataset))] == [0, 0, 0]

    loader = DataLoader(dataset, batch_size=2)
    inputs, targets = next(iter(loader))
    batch = Batch(inputs=inputs, targets=targets)

    assert batch.inputs.shape == (2, 1, 2, 2)
    assert batch.targets.tolist() == [0, 0]


def test_camera_stream_source_uses_temporal_buffer_capacity() -> None:
    buffer = TemporalBuffer(capacity=2)
    source = CameraStreamSource(
        name="camera-a",
        camera_id="0",
        frame_reader=_frame_reader,
        sampler=FrameSampler(max_frames=4),
        buffer=buffer,
        stream_metadata=StreamMetadata(source_id="camera-a", source_type="camera", fps=10.0),
    )

    ingested = asyncio.run(source.ingest())
    snapshot = asyncio.run(buffer.snapshot())

    assert ingested == 4
    assert len(snapshot) == 2
    assert [frame.frame_index for frame in snapshot] == [2, 3]


def test_temporal_buffer_can_create_labeled_dataset_snapshot() -> None:
    metadata = StreamMetadata(source_id="stream", source_type="camera")
    buffer = TemporalBuffer(capacity=3)
    asyncio.run(
        buffer.extend(
            (
                _frame(0, metadata, target=1),
                _frame(1, metadata, target=2),
            )
        )
    )

    dataset = asyncio.run(buffer.as_dataset())

    assert len(dataset) == 2
    assert dataset[0][1] == 1
    assert dataset[1][1] == 2


def test_stream_dataset_rejects_unlabeled_frames_without_default_target() -> None:
    source = VideoFileSource(
        name="unlabeled",
        path=Path("unlabeled.mp4"),
        frame_reader=_unlabeled_reader,
    )

    asyncio.run(source.ingest())
    dataset = source.build_dataset(split=DatasetSplit.TRAIN)

    with pytest.raises(ValueError, match="unlabeled"):
        _ = dataset[0]


def test_stream_metadata_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="fps"):
        StreamMetadata(source_id="bad", source_type="camera", fps=0.0)


async def _unlabeled_reader(metadata: StreamMetadata) -> AsyncIterator[Tensor]:
    _ = metadata
    yield torch.zeros(1, 2, 2)


def _frame(index: int, metadata: StreamMetadata, *, target: int) -> StreamFrame:
    return StreamFrame(
        image=torch.full((1, 2, 2), float(index)),
        timestamp_seconds=float(index),
        frame_index=index,
        metadata=metadata,
        target=target,
    )
