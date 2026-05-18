import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from torch import Tensor
from torch.utils.data import Dataset

from acn.continual.dataset import ImageSample
from acn.continual.stage import DatasetSplit

type StreamSourceType = Literal["video_file", "camera"]
type RawFrame = object
type FrameReader = Callable[["StreamMetadata"], AsyncIterator[RawFrame]]


@dataclass(frozen=True, slots=True)
class StreamMetadata:
    source_id: str
    source_type: StreamSourceType
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    uri: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fps is not None and self.fps <= 0.0:
            msg = "fps must be positive when provided."
            raise ValueError(msg)
        if self.width is not None and self.width <= 0:
            msg = "width must be positive when provided."
            raise ValueError(msg)
        if self.height is not None and self.height <= 0:
            msg = "height must be positive when provided."
            raise ValueError(msg)
        if self.duration_seconds is not None and self.duration_seconds < 0.0:
            msg = "duration_seconds cannot be negative."
            raise ValueError(msg)
        if self.frame_count is not None and self.frame_count < 0:
            msg = "frame_count cannot be negative."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StreamFrame:
    image: Tensor
    timestamp_seconds: float
    frame_index: int
    metadata: StreamMetadata
    target: int | None = None


@dataclass(frozen=True, slots=True)
class FrameSampler:
    sample_rate_hz: float | None = None
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0.0:
            msg = "sample_rate_hz must be positive when provided."
            raise ValueError(msg)
        if self.max_frames is not None and self.max_frames <= 0:
            msg = "max_frames must be positive when provided."
            raise ValueError(msg)

    def should_keep(
        self,
        *,
        frame: StreamFrame,
        emitted_count: int,
        last_timestamp_seconds: float | None,
    ) -> bool:
        if self.max_frames is not None and emitted_count >= self.max_frames:
            return False
        if self.sample_rate_hz is None:
            return True
        if last_timestamp_seconds is None:
            return True
        return frame.timestamp_seconds - last_timestamp_seconds >= 1.0 / self.sample_rate_hz


class IStreamSource(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def class_ids(self) -> tuple[int, ...]: ...

    @property
    def metadata(self) -> StreamMetadata: ...

    async def frames(self) -> AsyncIterator[StreamFrame]: ...

    async def ingest(self, buffer: "TemporalBuffer | None" = None) -> int: ...

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[Any]: ...


class TemporalBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            msg = "capacity must be positive."
            raise ValueError(msg)
        self._frames: deque[StreamFrame] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        maxlen = self._frames.maxlen
        if maxlen is None:
            msg = "TemporalBuffer is expected to have a fixed capacity."
            raise RuntimeError(msg)
        return maxlen

    def __len__(self) -> int:
        return len(self._frames)

    async def append(self, frame: StreamFrame) -> None:
        async with self._lock:
            self._frames.append(frame)

    async def extend(self, frames: Sequence[StreamFrame]) -> None:
        async with self._lock:
            self._frames.extend(frames)

    async def snapshot(self) -> tuple[StreamFrame, ...]:
        async with self._lock:
            return tuple(self._frames)

    def snapshot_now(self) -> tuple[StreamFrame, ...]:
        return tuple(self._frames)

    async def clear(self) -> None:
        async with self._lock:
            self._frames.clear()

    async def as_dataset(self, *, default_target: int | None = None) -> Dataset[ImageSample]:
        return StreamFrameDataset(await self.snapshot(), default_target=default_target)


class StreamFrameDataset(Dataset[ImageSample]):
    def __init__(
        self,
        frames: Sequence[StreamFrame],
        *,
        default_target: int | None = None,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self._default_target = default_target
        allowed_classes = frozenset(class_ids) if class_ids is not None else None
        self._frames = tuple(
            frame
            for frame in frames
            if allowed_classes is None or _resolve_target(frame, default_target) in allowed_classes
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> ImageSample:
        frame = self._frames[index]
        return frame.image, _resolve_target(frame, self._default_target)


@dataclass(slots=True)
class VideoFileSource:
    name: str
    path: Path
    frame_reader: FrameReader
    class_ids: tuple[int, ...] = ()
    sampler: FrameSampler = field(default_factory=FrameSampler)
    buffer: TemporalBuffer = field(default_factory=lambda: TemporalBuffer(capacity=1024))
    default_target: int | None = None
    stream_metadata: StreamMetadata | None = None

    @property
    def metadata(self) -> StreamMetadata:
        return self.stream_metadata or StreamMetadata(
            source_id=self.name,
            source_type="video_file",
            uri=str(self.path),
        )

    async def frames(self) -> AsyncIterator[StreamFrame]:
        async for frame in _sampled_frames(
            frame_reader=self.frame_reader,
            metadata=self.metadata,
            sampler=self.sampler,
            default_target=self.default_target,
        ):
            yield frame

    async def ingest(self, buffer: TemporalBuffer | None = None) -> int:
        target_buffer = buffer or self.buffer
        ingested = 0
        async for frame in self.frames():
            await target_buffer.append(frame)
            ingested += 1
        return ingested

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[Any]:
        _ = split
        return StreamFrameDataset(
            self.buffer.snapshot_now(),
            default_target=self.default_target,
            class_ids=class_ids,
        )


@dataclass(slots=True)
class CameraStreamSource:
    name: str
    camera_id: str
    frame_reader: FrameReader
    class_ids: tuple[int, ...] = ()
    sampler: FrameSampler = field(default_factory=FrameSampler)
    buffer: TemporalBuffer = field(default_factory=lambda: TemporalBuffer(capacity=256))
    default_target: int | None = None
    stream_metadata: StreamMetadata | None = None

    @property
    def metadata(self) -> StreamMetadata:
        return self.stream_metadata or StreamMetadata(
            source_id=self.name,
            source_type="camera",
            uri=self.camera_id,
        )

    async def frames(self) -> AsyncIterator[StreamFrame]:
        async for frame in _sampled_frames(
            frame_reader=self.frame_reader,
            metadata=self.metadata,
            sampler=self.sampler,
            default_target=self.default_target,
        ):
            yield frame

    async def ingest(self, buffer: TemporalBuffer | None = None) -> int:
        target_buffer = buffer or self.buffer
        ingested = 0
        async for frame in self.frames():
            await target_buffer.append(frame)
            ingested += 1
        return ingested

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[Any]:
        _ = split
        return StreamFrameDataset(
            self.buffer.snapshot_now(),
            default_target=self.default_target,
            class_ids=class_ids,
        )


def _coerce_stream_frame(
    raw_frame: RawFrame,
    *,
    metadata: StreamMetadata,
    fallback_index: int,
    fallback_target: int | None,
) -> StreamFrame:
    if isinstance(raw_frame, StreamFrame):
        return raw_frame
    if isinstance(raw_frame, Tensor):
        return _build_frame(
            image=raw_frame,
            metadata=metadata,
            frame_index=fallback_index,
            target=fallback_target,
        )
    if isinstance(raw_frame, Mapping):
        return _coerce_mapping_frame(
            raw_frame,
            metadata=metadata,
            fallback_index=fallback_index,
            fallback_target=fallback_target,
        )
    if isinstance(raw_frame, tuple):
        return _coerce_tuple_frame(
            raw_frame,
            metadata=metadata,
            fallback_index=fallback_index,
            fallback_target=fallback_target,
        )

    msg = f"Unsupported frame type: {type(raw_frame).__name__}."
    raise TypeError(msg)


async def _sampled_frames(
    *,
    frame_reader: FrameReader,
    metadata: StreamMetadata,
    sampler: FrameSampler,
    default_target: int | None,
) -> AsyncIterator[StreamFrame]:
    emitted_count = 0
    last_timestamp_seconds: float | None = None
    index = 0
    async for raw_frame in frame_reader(metadata):
        frame = _coerce_stream_frame(
            raw_frame,
            metadata=metadata,
            fallback_index=index,
            fallback_target=default_target,
        )
        index += 1
        if not sampler.should_keep(
            frame=frame,
            emitted_count=emitted_count,
            last_timestamp_seconds=last_timestamp_seconds,
        ):
            continue
        emitted_count += 1
        last_timestamp_seconds = frame.timestamp_seconds
        yield frame


def _coerce_mapping_frame(
    raw_frame: Mapping[str, object],
    *,
    metadata: StreamMetadata,
    fallback_index: int,
    fallback_target: int | None,
) -> StreamFrame:
    image = raw_frame.get("image")
    if not isinstance(image, Tensor):
        msg = "Frame mapping must include an image tensor."
        raise TypeError(msg)
    target = _optional_int(raw_frame.get("target"), fallback_target=fallback_target)
    timestamp = _optional_float(raw_frame.get("timestamp_seconds"))
    frame_index = _optional_int(raw_frame.get("frame_index"), fallback_target=fallback_index)
    if frame_index is None:
        msg = "Frame mapping must include a frame index when no fallback is available."
        raise TypeError(msg)
    return _build_frame(
        image=image,
        metadata=metadata,
        frame_index=frame_index,
        target=target,
        timestamp_seconds=timestamp,
    )


def _coerce_tuple_frame(
    raw_frame: tuple[object, ...],
    *,
    metadata: StreamMetadata,
    fallback_index: int,
    fallback_target: int | None,
) -> StreamFrame:
    if not raw_frame:
        msg = "Frame tuple cannot be empty."
        raise TypeError(msg)
    image = raw_frame[0]
    if not isinstance(image, Tensor):
        msg = "Frame tuple first element must be an image tensor."
        raise TypeError(msg)
    target = (
        _optional_int(raw_frame[1], fallback_target=fallback_target)
        if len(raw_frame) > 1
        else fallback_target
    )
    timestamp = _optional_float(raw_frame[2]) if len(raw_frame) > 2 else None
    return _build_frame(
        image=image,
        metadata=metadata,
        frame_index=fallback_index,
        target=target,
        timestamp_seconds=timestamp,
    )


def _build_frame(
    *,
    image: Tensor,
    metadata: StreamMetadata,
    frame_index: int,
    target: int | None,
    timestamp_seconds: float | None = None,
) -> StreamFrame:
    resolved_timestamp = timestamp_seconds
    if resolved_timestamp is None:
        fps = metadata.fps or 1.0
        resolved_timestamp = frame_index / fps
    return StreamFrame(
        image=image,
        timestamp_seconds=resolved_timestamp,
        frame_index=frame_index,
        metadata=metadata,
        target=target,
    )


def _resolve_target(frame: StreamFrame, default_target: int | None) -> int:
    target = frame.target if frame.target is not None else default_target
    if target is None:
        msg = "Stream frame is unlabeled and no default target was configured."
        raise ValueError(msg)
    return target


def _optional_int(value: object, *, fallback_target: int | None) -> int | None:
    if value is None:
        return fallback_target
    if isinstance(value, bool):
        msg = "Boolean values are not valid integer targets."
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    msg = f"Expected integer value, got {type(value).__name__}."
    raise TypeError(msg)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    msg = f"Expected numeric timestamp, got {type(value).__name__}."
    raise TypeError(msg)
