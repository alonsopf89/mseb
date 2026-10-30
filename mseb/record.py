# Copyright 2026 The MSEB Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bagz and MsgPack serialization for MSEB multimodal objects."""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import os
from typing import Any

from absl import logging
import bagz
from etils import epath
from mseb import types
import msgpack
import numpy as np
import tqdm

tqdm = tqdm.tqdm

_SERIALIZABLE_TYPES = {
    cls.__name__: cls
    for cls in (
        types.EncodingStats,
        types.TextContextParams,
        types.SoundContextParams,
        types.ImageContextParams,
        types.PredictionContextParams,
        types.Text,
        types.TextWithTitleAndContext,
        types.Sound,
        types.SoundWithTitleAndContext,
        types.Image,
        types.SoundEmbedding,
        types.SoundEmbeddingWithTitleAndContext,
        types.TextEmbedding,
        types.ImageEmbedding,
        types.TextPrediction,
        types.SoundEmbeddingCollection,
        types.StreamingSpeechToTextEvent,
        types.StreamingStats,
    )
}


def _msgpack_default(obj: Any) -> Any:
  """Encodes NumPy arrays, scalars, and MSEB dataclasses for msgpack."""
  if isinstance(obj, np.ndarray):
    if obj.dtype.kind == 'O':
      return {'__ndarray_obj__': obj.tolist(), 'shape': obj.shape}
    return {
        '__ndarray__': obj.tobytes(),
        'dtype': obj.dtype.str,
        'shape': obj.shape,
    }
  if isinstance(obj, np.generic):
    return obj.item()
  if (
      dataclasses.is_dataclass(obj)
      and not isinstance(obj, type)
      and type(obj).__name__ in _SERIALIZABLE_TYPES
  ):
    return {
        '__type__': type(obj).__name__,
        'data': {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)},
    }
  raise TypeError(f'Unsupported type for msgpack serialization: {type(obj)}')


def _msgpack_object_hook(d: dict[str, Any]) -> Any:
  """Decodes NumPy arrays and MSEB dataclasses from msgpack."""
  if '__ndarray__' in d:
    return (
        np.frombuffer(d['__ndarray__'], dtype=np.dtype(d['dtype']))
        .reshape(d['shape'])
        .copy()
    )
  if '__ndarray_obj__' in d:
    return np.array(d['__ndarray_obj__'], dtype=object).reshape(d['shape'])
  if '__type__' in d:
    if d['__type__'] not in _SERIALIZABLE_TYPES:
      raise ValueError(f"Unknown serialized type: {d['__type__']}")
    return _SERIALIZABLE_TYPES[d['__type__']](**d['data'])
  return d


def encode(obj: types.MultiModalObject) -> bytes:
  """Serializes an MSEB MultiModalObject to msgpack bytes."""
  return msgpack.packb(obj, default=_msgpack_default, use_bin_type=True)


def decode(data: bytes) -> types.MultiModalObject:
  """Deserializes an MSEB MultiModalObject from msgpack bytes."""
  return msgpack.unpackb(data, object_hook=_msgpack_object_hook, raw=False)


def get_num_shards(num_examples: int, examples_per_shard: int = 25_000) -> int:
  """Returns the number of shards for the given number of examples."""
  return math.ceil(num_examples / examples_per_shard)


def etch(
    output_prefix: str,
    objects: (
        Mapping[str, types.MultiModalObject] | Sequence[types.MultiModalObject]
    ),
    examples_per_shard: int = 25_000,
) -> None:
  """Serializes MSEB objects into sharded Bagz files on disk."""
  values = objects.values() if isinstance(objects, Mapping) else objects
  num_objects = len(objects)
  num_shards = get_num_shards(
      num_objects, examples_per_shard=examples_per_shard
  )
  logging.info(
      'Etching %d records to %s@%d.bagz', num_objects, output_prefix, num_shards
  )
  epath.Path(os.path.dirname(output_prefix)).mkdir(parents=True, exist_ok=True)
  obj_it = iter(values)
  for shard_id in range(num_shards):
    with bagz.Writer(
        f'{output_prefix}-{shard_id:05d}-of-{num_shards:05d}.bagz'
    ) as writer:
      for _ in range(
          shard_id * examples_per_shard,
          min(num_objects, (shard_id + 1) * examples_per_shard),
      ):
        writer.write(encode(next(obj_it)))


def play(output_prefix: str) -> dict[str, Any]:
  """Reads serialized MSEB objects from Bagz shards back into a dict."""
  output_pattern = output_prefix + '*'
  logging.info('Playing records from %s', output_pattern)
  bagz_files = sorted(
      f
      for f in epath.Path(os.path.dirname(output_pattern)).glob(
          os.path.basename(output_pattern)
      )
      if f.name.endswith(('.bagz', '.bag'))
  )
  if not bagz_files:
    raise FileNotFoundError(f'No Bagz files found matching {output_pattern}')
  objects = {}
  for filename in tqdm(  # pyrefly: ignore[not-callable]
      bagz_files, total=len(bagz_files), desc='Loading records'
  ):
    for raw in bagz.Reader(os.fspath(filename)):
      obj = decode(raw)
      objects[obj.context.id] = obj
  logging.info('Loaded %d records from %s', len(objects), output_pattern)
  if not objects:
    raise FileNotFoundError(
        f'Found Bagz files matching {output_pattern} but they contained zero'
        ' records'
    )
  return objects
