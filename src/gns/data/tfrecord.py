"""A minimal reader for the released ``*.tfrecord`` datasets.

The datasets are ``tf.SequenceExample`` records: one record per trajectory, with
the particle types in the context and one length-delimited float32 blob per
frame in the feature lists.  Decoding that needs a few dozen lines of protobuf
wire-format parsing, which is cheaper than depending on TensorFlow for a
one-time conversion.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np

_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH = 2
_WIRE_32BIT = 5


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes) -> Iterator[tuple[int, bytes | int]]:
    """Yield ``(field_number, payload)`` for every field in a message."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == _WIRE_VARINT:
            value, pos = _read_varint(buf, pos)
            yield field, value
        elif wire == _WIRE_LENGTH:
            size, pos = _read_varint(buf, pos)
            yield field, buf[pos : pos + size]
            pos += size
        elif wire == _WIRE_64BIT:
            yield field, buf[pos : pos + 8]
            pos += 8
        elif wire == _WIRE_32BIT:
            yield field, buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")


def _map_entry(buf: bytes) -> tuple[str, bytes]:
    key = b""
    value = b""
    for field, payload in _fields(buf):
        if field == 1:
            key = payload
        elif field == 2:
            value = payload
    return key.decode(), value


def _bytes_list(feature: bytes) -> list[bytes]:
    """Unwrap ``Feature -> bytes_list -> value``, the shape every field uses.

    Positions and particle types are both stored as raw little-endian buffers
    inside a ``bytes_list``, so the caller gets back one blob per entry.
    """
    values = []
    for field, payload in _fields(feature):
        if field != 1:  # 1 is bytes_list; 2 and 3 are float and int64 lists
            continue
        values += [value for sub, value in _fields(payload) if sub == 1]
    return values


def read_records(path: str | Path) -> Iterator[bytes]:
    """Yield the raw payload of every record in a TFRecord file."""
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            handle.read(4)  # CRC of the header; the files are trusted.
            payload = handle.read(length)
            handle.read(4)  # CRC of the payload.
            yield payload


def read_trajectories(
    path: str | Path, dim: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(positions [T, N, dim], particle_types [N])`` per trajectory."""
    for record in read_records(path):
        context: dict[str, bytes] = {}
        sequence: dict[str, list[bytes]] = {}
        for field, payload in _fields(record):
            if field == 1:  # context Features
                for sub, entry in _fields(payload):
                    if sub != 1:
                        continue
                    name, feature = _map_entry(entry)
                    # ``key`` is an int64 trajectory id we do not need, and it
                    # is the only context feature that is not a bytes blob.
                    if name != "particle_type":
                        continue
                    context[name] = _bytes_list(feature)[0]
            elif field == 2:  # FeatureLists
                for sub, entry in _fields(payload):
                    if sub == 1:
                        name, feature_list = _map_entry(entry)
                        if name != "position":
                            continue
                        sequence[name] = [
                            _bytes_list(feature)[0]
                            for f, feature in _fields(feature_list)
                            if f == 1
                        ]

        types = np.frombuffer(context["particle_type"], dtype=np.int64)
        frames = [
            np.frombuffer(blob, dtype=np.float32).reshape(-1, dim)
            for blob in sequence["position"]
        ]
        yield np.stack(frames, axis=0), types.copy()
