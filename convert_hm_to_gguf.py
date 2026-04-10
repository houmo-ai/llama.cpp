#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import logging
import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    cast,
)
from itertools import chain
from transformers import AutoConfig
import glob

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import re
import datetime
import gguf
import struct


logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2

class HmmInfo:
    """HMM模型信息类，包含模型的目标平台、硬件配置、特性等信息"""
    target: str
    batch: int
    core_num: int
    device_num: int
    context_length: int
    name: str
    size: str
    flash_attention: bool
    version: str
    convert_time: str

    def __init__(self):
        # 初始化所有属性为默认值
        self.target = ""
        self.batch = 0
        self.core_num = 0
        self.device_num = 1
        self.context_length = 0
        self.name = ""
        self.size = ""
        self.flash_attention = False
        self.version = ""
        self.convert_time = ""

    def __str__(self) -> str:
        """友好的打印格式，便于阅读"""
        return (
            f"📊 HMM Model Information\n"
            f"├───────────────────────────────────────\n"
            f"│ Basic Info:\n"
            f"│   Name:         {self.name or 'Unknown'}\n"
            f"│   Size:         {self.size or 'Unknown'}\n"
            f"│   Version:      {self.version or 'Unknown'}\n"
            f"│   Target:       {self.target or 'Unknown'}\n"
            f"│   Convert Time: {self.convert_time or 'Unknown'}\n"
            f"│   Batch Size:       {self.batch}\n"
            f"│   Core Number:      {self.core_num}\n"
            f"│   Device Number:    {self.device_num}\n"
            f"│   Context Length:   {self.context_length}\n"
            f"│\n"
            f"│ Features:\n"
            f"│   Flash Attention:  {'✅ Enabled' if self.flash_attention else '❌ Disabled'}\n"
            f"└───────────────────────────────────────"
        )

    def __repr__(self) -> str:
        """调试用格式，包含所有属性"""
        return (
            f"HmmInfo(target='{self.target}', batch={self.batch}, core_num={self.core_num}, "
            f"device_num={self.device_num}, context_length={self.context_length}, name='{self.name}', "
            f"size='{self.size}', flash_attention={self.flash_attention}, version='{self.version}', "
            f"convert_time='{self.convert_time}')"
        )

import subprocess

def read_hmm_info(hmm_file: str) -> HmmInfo:
    """
    Read model info from hmm file using readhmm command

    Args:
        hmm_file: Path to hmm or hmms file

    Returns:
        HmmInfo object with model information
    """
    hmm_info = HmmInfo()
    # Check if the file exists
    if not os.path.exists(hmm_file):
        raise FileNotFoundError(f"File not found: {hmm_file}")

    # Determine file type and construct command
    file_ext = os.path.splitext(hmm_file)[1].lower()
    readhmm_exists = False
    # Execute command and get output
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    # 拼接readhmm的完整路径（脚本目录下的readhmm）
    readhmm_path = os.path.join(script_dir, "readhmm")
    if os.path.exists(readhmm_path):
        readhmm_exists = True
    else:
        readhmm_exists = False

    if readhmm_exists:
        # Construct command
        if file_ext == '.hmm':
            cmd = [readhmm_path, '--info', hmm_file]
        else:
            cmd = [readhmm_path, '--info', '--models', hmm_file]

        # Execute command and get output
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute readhmm command: {e.stderr}")

        # Parse JSON output
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse readhmm output as JSON: {output}")

        # Extract info based on file type
        if file_ext == '.hmm':
            # For .hmm files, info is directly under 'info' field
            info_data = data.get('info', {})
            build_option = ast.literal_eval(info_data.get('build_option', '{}'))
        else:
            # For .hmms files, info is under 'models' -> first model -> 'info'
            models = data.get('models', {})
            if not models:
                raise ValueError(f"No models found in {hmm_file}")

            # Get the first model's info
            first_model_key = next(iter(models.keys()))
            info_data = models[first_model_key].get('info', {})
            build_option = ast.literal_eval(info_data.get('build_option', '{}'))

        # Extract basic info
        hmm_info.target = info_data.get('target', '')
        hmm_info.name = info_data.get('name', '')
        hmm_info.version = info_data.get('version', '')
        hmm_info.core_num = info_data.get('core_num', 0)

        # Extract batch from build_option
        hmm_info.batch = build_option.get('batch', 1)

        # Extract device_num from build_option's ndevice, default to 1 if 0 or not present
        hmm_info.device_num = build_option.get('ndevice', 0)
        if hmm_info.device_num == 0:
            hmm_info.device_num = 1

        # Extract context-length from build_option's modify_llm
        modify_llm = build_option.get('modify_llm', {})
        hmm_info.context_length = modify_llm.get('context-length', 0)

        # Extract flash_attention from build_option
        hmm_info.flash_attention = bool(build_option.get('flash_attention', 0))
    else:
        # readhmm not found, fallback to embedding.pt
        logger.info("readhmm command not found, falling back to embedding.pt detection")

        # Find embedding.pt files in current directory
        embedding_files = list(Path('.').glob('*embedding.pt'))
        if embedding_files:
            # Use the first embedding.pt file
            embedding_file = embedding_files[0]
            logger.info(f"Found embedding file: {embedding_file}")

            # Load embedding file and check dtype
            try:
                embedding = torch.load(embedding_file, map_location='cpu', weights_only=False)
                if isinstance(embedding, dict):
                    # If it's a dict, check the first tensor
                    first_key = next(iter(embedding.keys()))
                    dtype = embedding[first_key].dtype
                else:
                    # If it's a tensor directly
                    dtype = embedding.dtype

                # Set target based on dtype
                if dtype == torch.int16:
                    hmm_info.target = 'xh1'
                elif dtype == torch.float16:
                    hmm_info.target = 'xh2'
                else:
                    logger.warning(f"Unknown dtype {dtype} in embedding.pt, defaulting to xh2")
                    hmm_info.target = 'xh2'
            except Exception as e:
                logger.error(f"Failed to load embedding.pt: {e}")
                # Default to xh2 if we can't determine
                hmm_info.target = 'xh2'
        else:
            hmm_info.target = 'xh2'
        # Set default values
        hmm_info.core_num = 2
        hmm_info.batch = 1

    # Check if it's a hmms file to set device_num=2
    if file_ext == '.hmms':
        hmm_info.device_num = 2
    else:
        hmm_info.device_num = 1

    hmm_info.convert_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return hmm_info

AnyModel = TypeVar("AnyModel", bound="type[ModelBase]")


class ModelBase:
    _model_classes: dict[ModelType, dict[str, type[ModelBase]]] = {
        ModelType.TEXT: {},
        ModelType.MMPROJ: {},
    }

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    part_names: list[str]
    is_safetensors: bool
    hparams: dict[str, Any]
    tensor_names: set[str] | None
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path
    remote_hf_model_id: str | None
    is_mistral_format: bool = False

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH
    model_description: str | None = None

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    def __init__(
        self,
        dir_model: Path,
        ftype: gguf.LlamaFileType,
        fname_out: Path,
        *,
        is_big_endian: bool = False,
        use_temp_file: bool = False,
        eager: bool = False,
        metadata_override: Path | None = None,
        model_name: str | None = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        hparams: dict[str, Any] | None = None,
        remote_hf_model_id: str | None = None,
        model_description: str | None = None,
        hmm_vit_height: int = 0,
        hmm_vit_width: int = 0,
    ):
        if (
            type(self) is ModelBase
            or type(self) is TextModel
            or type(self) is MmprojModel
        ):
            raise TypeError(
                f"{type(self).__name__!r} should not be directly instantiated"
            )

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.remote_hf_model_id = remote_hf_model_id
        if remote_hf_model_id is not None:
            self.is_safetensors = True

            def get_remote_tensors() -> Iterator[tuple[str, Tensor]]:
                logger.info(
                    f"Using remote model with HuggingFace id: {remote_hf_model_id}"
                )
                remote_tensors = (
                    gguf.utility.SafetensorRemote.get_list_tensors_hf_model(
                        remote_hf_model_id
                    )
                )
                self.tensor_names = set(name for name in remote_tensors.keys())
                for (
                    name,
                    remote_tensor,
                ) in gguf.utility.SafetensorRemote.get_list_tensors_hf_model(
                    remote_hf_model_id
                ).items():
                    yield (name, LazyTorchTensor.from_remote_tensor(remote_tensor))

            self.get_tensors = get_remote_tensors
        else:
            self.part_names = ModelBase.get_model_part_names(
                self.dir_model, "model", ".safetensors"
            )
            self.is_safetensors = len(self.part_names) > 0
            if not self.is_safetensors:
                self.part_names = ModelBase.get_model_part_names(
                    self.dir_model, "pytorch_model", ".bin"
                )
        self.hparams = (
            ModelBase.load_hparams(self.dir_model) if hparams is None else hparams
        )
        self.tensor_names = None
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py
        self.hmm_info = None
        self.hmm_vit_height = hmm_vit_height
        self.hmm_vit_width = hmm_vit_width

        # Apply heuristics to figure out typical tensor encoding based on first layer tensor encoding type
        if self.ftype == gguf.LlamaFileType.GUESSED:
            # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(
                    f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(
                    f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch=gguf.MODEL_ARCH_NAMES[self.model_arch],
            endianess=self.endianess,
            use_temp_file=self.use_temp_file,
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
        )

    @classmethod
    def add_prefix_to_filename(cls, path: Path, prefix: str) -> Path:
        stem, suffix = path.stem, path.suffix
        new_name = f"{prefix}{stem}{suffix}"
        return path.with_name(new_name)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        tensor_names_from_parts: set[str] = set()

        index_name = "model.safetensors" if self.is_safetensors else "pytorch_model.bin"
        index_name += ".index.json"
        index_file = self.dir_model / index_name

        if index_file.is_file():
            self.tensor_names = set()
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(index_file, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                self.tensor_names.update(weight_map.keys())
        else:
            self.tensor_names = tensor_names_from_parts
            weight_map = {}

        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open

                ctx = cast(
                    ContextManager[Any],
                    safe_open(self.dir_model / part_name, framework="pt", device="cpu"),
                )
            else:
                ctx = contextlib.nullcontext(
                    torch.load(
                        str(self.dir_model / part_name),
                        map_location="cpu",
                        mmap=True,
                        weights_only=True,
                    )
                )

            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    if self.is_safetensors:
                        if self.lazy:
                            data = model_part.get_slice(name)
                            data = LazyTorchTensor.from_safetensors_slice(data)
                        else:
                            data = model_part.get_tensor(name)
                    else:
                        data = model_part[name]
                        if self.lazy:
                            data = LazyTorchTensor.from_eager(data)
                    yield name, data

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_parts.symmetric_difference(self.tensor_names)) > 0:
            missing = sorted(self.tensor_names.difference(tensor_names_from_parts))
            extra = sorted(tensor_names_from_parts.difference(self.tensor_names))
            missing_files = sorted(
                set(weight_map[n] for n in missing if n in weight_map)
            )
            if len(extra) == 0 and len(missing_files) > 0:
                raise ValueError(
                    f"Missing or incomplete model files: {missing_files}\n"
                    f"Missing tensors: {missing}"
                )
            else:
                raise ValueError(
                    "Mismatch between weight map and model parts for tensor names:\n"
                    f"Missing tensors: {missing}\n"
                    f"Extra tensors: {extra}"
                )

    def format_tensor_name(
        self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight"
    ) -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(
                f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}"
            )
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(
        self,
        name: str,
        key: gguf.MODEL_TENSOR,
        bid: int | None,
        suffix: str = ".weight",
    ) -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(
        self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")
    ) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        raise NotImplementedError(
            "set_gguf_parameters() must be implemented in subclasses"
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(
        self, name: str, new_name: str, bid: int | None, n_dims: int
    ) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid, n_dims  # unused

        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(
            ".weight,"
        )

        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            # we don't need these
            if name.endswith(
                (".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")
            ):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
                # TODO: why do we squeeze here?
                # data = data_torch.squeeze().numpy()
                data = data_torch.numpy()

                # if data ends up empty, it means data_torch was a scalar tensor -> restore
                if len(data.shape) == 0:
                    data = data_torch.numpy()

                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(
                    name, new_name, bid, n_dims
                )

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                # Some tensor types are always in float32
                if data_qtype is False and (
                    any(
                        self.match_model_tensor_name(new_name, key, bid)
                        for key in (
                            gguf.MODEL_TENSOR.FFN_GATE_INP,
                            gguf.MODEL_TENSOR.POS_EMBD,
                            gguf.MODEL_TENSOR.TOKEN_TYPES,
                            gguf.MODEL_TENSOR.SSM_CONV1D,
                            gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                            gguf.MODEL_TENSOR.TIME_MIX_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_LERP_FUSED,
                            gguf.MODEL_TENSOR.POSNET_NORM1,
                            gguf.MODEL_TENSOR.POSNET_NORM2,
                            gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                            gguf.MODEL_TENSOR.A_ENC_EMBD_POS,
                            gguf.MODEL_TENSOR.ALTUP_CORRECT_COEF,
                            gguf.MODEL_TENSOR.ALTUP_PREDICT_COEF,
                        )
                    )
                    or not new_name.endswith(".weight")
                ):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                        gguf.MODEL_TENSOR.ALTUP_ROUTER,
                        gguf.MODEL_TENSOR.LAUREL_L,
                        gguf.MODEL_TENSOR.LAUREL_R,
                    )
                ):
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_TQ1_0,
                        gguf.LlamaFileType.MOSTLY_TQ2_0,
                    ):
                        # TODO: use Q4_K and Q6_K
                        data_qtype = gguf.GGMLQuantizationType.F16

                # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if isinstance(data_qtype, bool):
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                        data_qtype = gguf.GGMLQuantizationType.Q8_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ1_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ1_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ2_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ2_0
                    else:
                        raise ValueError(f"Unknown file type: {self.ftype.name}")

                try:
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)

                shape = (
                    gguf.quant_shape_from_byte_shape(data.shape, data_qtype)
                    if data.dtype == np.uint8
                    else data.shape
                )

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(
                    f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
                )

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):

        total_params, shared_params, expert_params, expert_count = (
            self.gguf_writer.get_total_parameter_count()
        )

        self.metadata = gguf.Metadata.load(
            self.metadata_override, self.dir_model_card, self.model_name, total_params
        )

        # If we are using HF model id, set the metadata name to the model id
        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None or self.metadata.name == "":
            self.metadata.name = self.dir_model.name

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(
                total_params, shared_params, expert_params, expert_count
            )

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def write_vocab(self):
        raise NotImplementedError("write_vocab() must be implemented in subclasses")

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams(dir_model: Path):
        try:
            # for security reason, we don't allow loading remote code by default
            # if a model need remote code, we will fallback to config.json
            config = AutoConfig.from_pretrained(
                dir_model, trust_remote_code=False
            ).to_dict()
        except Exception as e:
            logger.warning(f"Failed to load model config from {dir_model}: {e}")
            logger.warning("Trying to load config.json instead")
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        if "llm_config" in config:
            # rename for InternVL
            config["text_config"] = config["llm_config"]
        if "thinker_config" in config:
            # rename for Qwen2.5-Omni
            config["text_config"] = config["thinker_config"]["text_config"]
        return config

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            model_type = (
                ModelType.MMPROJ
                if modelcls.model_arch == gguf.MODEL_ARCH.MMPROJ
                else ModelType.TEXT
            )
            for name in names:
                cls._model_classes[model_type][name] = modelcls
            return modelcls

        return func

    @classmethod
    def print_registered_models(cls):
        for model_type, model_classes in cls._model_classes.items():
            logger.error(f"{model_type.name} models:")
            for name in sorted(model_classes.keys()):
                logger.error(f"  - {name}")

    @classmethod
    def from_model_architecture(
        cls, arch: str, model_type=ModelType.TEXT
    ) -> type[ModelBase]:
        try:
            return cls._model_classes[model_type][arch]
        except KeyError:
            raise NotImplementedError(f"Architecture {arch!r} not supported!") from None


class TextModel(ModelBase):
    model_type = ModelType.TEXT
    hf_arch: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_mistral_format:
            self.hf_arch = get_model_architecture(self.hparams, self.model_type)
        else:
            self.hf_arch = ""

        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}

        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.rope_parameters = self.hparams.get("rope_parameters", self.hparams.get("rope_scaling")) or {}

        rope_theta = self.find_hparam(["global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base"], optional=True)
        local_rope_theta = self.find_hparam(["local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq"], optional=True)

        # Ensure "rope_theta" and "rope_type" is mirrored in rope_parameters
        if "full_attention" not in self.rope_parameters and "sliding_attention" not in self.rope_parameters:
            if local_rope_theta is not None:
                self.rope_parameters["sliding_attention"] = {"rope_theta": local_rope_theta}
            if "rope_theta" not in self.rope_parameters and rope_theta is not None:
                self.rope_parameters["rope_theta"] = rope_theta
            if "rope_type" not in self.rope_parameters and (rope_type := self.rope_parameters.get("type")) is not None:
                self.rope_parameters["rope_type"] = rope_type
    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    def conver_embedding(self, embedding_path: Path):
        # ===================== 1. 保留原权重加载+类型转换逻辑 =====================
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=False
        )
        # 提取核心权重张量
        if isinstance(embedding_weight, dict) and "weight" in embedding_weight:
            embedding_weight = embedding_weight["weight"]
        if isinstance(embedding_weight, torch.nn.Embedding):
            embedding_weight = embedding_weight.weight
        print(f"embedding_weight.dtype: {embedding_weight.dtype}")
        # 数据类型转换（保留int16，仅调整日志注释）
        dtype_mapping = {torch.int16: 0, torch.float16: 1}  # 类型标识与C++端对齐
        final_dtype = None
        if embedding_weight.dtype == torch.int16 or embedding_weight.dtype == torch.int64:
            print(
                "M30：keep weight as int16 for minimal memory usage"  # 移除转float32注释
            )
            embedding_weight = embedding_weight.to(torch.int16)
            final_dtype = torch.int16
        elif embedding_weight.dtype == torch.float16:
            embedding_weight = embedding_weight.to(torch.float16)
            print("M50: keep embedding weight as float16")
            final_dtype = torch.float16
        elif embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.to(torch.float16)
            print("M50: convert embedding weight to float16")
            final_dtype = torch.float16
        else:
            print(f"both M30 & M50 not suuport ")
            sys.exit(1)
        # ===================== 2. 导出为通用二进制格式（保留原类型） =====================
        out_path = self.dir_model / "cpp_quant_embedding.bin"
        embedding_weight = embedding_weight.contiguous()  # 确保内存连续
        with open(out_path, "wb") as f:
            # 写入维度数
            dims = embedding_weight.shape
            f.write(struct.pack("I", len(dims)))  # I=uint32
            # 写入各维度值
            for d in dims:
                f.write(struct.pack("I", d))
            # 写入数据类型标识
            f.write(struct.pack("I", dtype_mapping[final_dtype]))
            # 写入原始数据（保留原类型，不转换）
            if final_dtype == torch.int16:
                f.write(embedding_weight.numpy().astype(np.int16).tobytes())
            elif final_dtype == torch.float16:
                f.write(embedding_weight.numpy().astype(np.float16).tobytes())
        print(f"Embedding weight exported to binary file: {out_path}")
        print(f"Binary info - shape: {dims}, dtype: {final_dtype}")
        return out_path

    def get_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        logger.info(".....Getting tensors")

        # 定义需要匹配的键名与模式映射
        # 分为"必须存在"和"可选存在"两组
        required_keys = {
            "quant_embedding.bin": re.compile(r".*embedding\.pt$"),
            "prefill.hmm": re.compile(r".*prefill\.hmm(s)?$"),
            #"decoder.hmm": re.compile(r".*decode(r)?\.hmm(s)?$"),
        }
        optional_keys = {
            "decoder.hmm": re.compile(r".*decode(r)?\.hmm(s)?$"),
            "prefill_0.hmm": re.compile(r".*prefill_0\.hmm(s)?$"),
            "prefill_1.hmm": re.compile(r".*prefill_1\.hmm(s)?$"),
            "decoder_0.hmm": re.compile(r".*decoder_0\.hmm(s)?$"),
            "decoder_1.hmm": re.compile(r".*decoder_1\.hmm(s)?$"),
        }

        # 检查模型目录是否存在
        if not self.dir_model.exists():
            logger.error(f"Model directory {self.dir_model} does not exist")
            raise FileNotFoundError(f"Model directory {self.dir_model} not found")

        # 获取目录下所有文件
        all_files = [
            f
            for f in self.dir_model.iterdir()
            if f.is_file() and (f.suffix == ".hmm" or f.suffix == ".hmms")
        ]
        print(f"all_files: {all_files}")
        for f in self.dir_model.rglob("*embedding.pt"):  # 递归查找所有 embedding.pt
            if f.is_file():
                all_files.append(f)
                break
        print(f"all_files: {all_files}")

        # 处理必须存在的文件（不存在则报错）
        matched_required = {}
        for key, pattern in required_keys.items():
            matched = [file for file in all_files if pattern.match(file.name)]
            if not matched:
                logger.error(f"Missing required file for key: {key}")
                raise FileNotFoundError(f"No file found for required key {key}")
            matched_required[key] = matched[0]  # 取第一个匹配文件

        # 处理可选文件（不存在则跳过）
        matched_optional = {}
        for key, pattern in optional_keys.items():
            matched = [file for file in all_files if pattern.match(file.name)]
            if matched:
                matched_optional[key] = matched[0]  # 存在则记录
            else:
                logger.warning(f"Optional file for key {key} not found, skipping")

        # 合并所有匹配的文件（必须存在的 + 可选存在的）
        all_matched = {**matched_required, **matched_optional}

        # 按顺序处理所有匹配到的文件
        # 先处理必须存在的，再处理可选的（保持键名顺序）
        for key in list(required_keys.keys()) + list(optional_keys.keys()):
            if key not in all_matched:
                continue  # 跳过不存在的可选文件

            file_path = all_matched[key].absolute()
            logger.info(f".....Loading tensor from {file_path} (mapped to key: {key})")

            if not file_path.is_file():
                logger.error(f".....{file_path} not found")
                raise FileNotFoundError(f"File {file_path} missing")

            # 处理量化嵌入文件的特殊逻辑
            if key == "quant_embedding.bin":
                file_path = self.conver_embedding(file_path)
            if key == "decoder.hmm":
                print(file_path)
                hmm_info = read_hmm_info(file_path)
                print(hmm_info)
                self.hmm_info = json.dumps(hmm_info.__dict__)
            # 读取二进制数据并转换为Tensor
            with open(file_path, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size for {key}: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())

            # 移除转换后的临时文件
            if key == "quant_embedding.bin":
                os.remove(file_path)

            # 以固定键名返回
            yield key, data
        # 遍历目录下所有的json文件，只返回不带路径的文件名
        for json_file in glob.glob(os.path.join(self.dir_model, "*.json")):
            with open(json_file, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield os.path.basename(json_file), data
        for json_file in glob.glob(os.path.join(self.dir_model, "*.txt")):
            with open(json_file, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield os.path.basename(json_file), data
        for json_file in glob.glob(os.path.join(self.dir_model, "*.jinja")):
            with open(json_file, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield os.path.basename(json_file), data

    def set_vocab(self):
        self._set_vocab_gpt2()

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        total_params = self.gguf_writer.get_total_parameter_count()[0]
        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            if not vocab_only:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, self.metadata.size_label, output_type, model_type="LoRA" if total_params < 0 else None)
            else:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=None, model_type="vocab")

            # Use the default filename
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

        logger.info("Set model tokenizer")
        self.set_vocab()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        if (n_embd := self.find_hparam(["hidden_size", "n_embd", "dim"], optional=True)) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
            logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner", "hidden_dim"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        if (n_head := self.find_hparam(["num_attention_heads", "n_head", "n_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count(n_head)
            logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.find_hparam(["num_key_value_heads", "n_kv_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        # TODO: Handle "sliding_attention" similarly when models start implementing it
        rope_params = self.rope_parameters.get("full_attention", self.rope_parameters)
        if (rope_type := rope_params.get("rope_type")) is not None:
            rope_factor = rope_params.get("factor")
            rope_gguf_type = gguf.RopeScalingType.NONE
            if rope_type == "linear" and rope_factor is not None:
                rope_gguf_type = gguf.RopeScalingType.LINEAR
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
                self.gguf_writer.add_rope_scaling_factor(rope_factor)
            elif rope_type == "yarn" and rope_factor is not None:
                rope_gguf_type = gguf.RopeScalingType.YARN
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
                self.gguf_writer.add_rope_scaling_factor(rope_factor)
                self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_params["original_max_position_embeddings"])
                if (yarn_ext_factor := rope_params.get("extrapolation_factor")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_ext_factor(yarn_ext_factor)
                if (yarn_attn_factor := rope_params.get("attention_factor", rope_params.get("attn_factor"))) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_attn_factor(yarn_attn_factor)
                if (yarn_beta_fast := rope_params.get("beta_fast")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_beta_fast(yarn_beta_fast)
                if (yarn_beta_slow := rope_params.get("beta_slow")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_beta_slow(yarn_beta_slow)
                # self.gguf_writer.add_rope_scaling_yarn_log_mul(rope_params["mscale_all_dim"])
            elif rope_type == "su" or rope_type == "longrope":
                rope_gguf_type = gguf.RopeScalingType.LONGROPE
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
            elif rope_type == "dynamic":
                # HunYuan, handled in model class
                pass
            elif rope_type.lower() == "llama3":
                # Handled in generate_extra_tensors
                pass
            else:
                logger.warning(f"Unknown RoPE type: {rope_type}")
            logger.info(f"gguf: rope scaling type = {rope_gguf_type.name}")

        if "mrope_section" in self.rope_parameters:
            mrope_section = self.rope_parameters["mrope_section"]
            # Pad to 4 dimensions [time, height, width, extra]
            while len(mrope_section) < 4:
                mrope_section.append(0)
            self.gguf_writer.add_rope_dimension_sections(mrope_section[:4])
            logger.info(f"gguf: mrope sections: {mrope_section[:4]}")

        if (rope_theta := rope_params.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (local_rope_theta := self.rope_parameters.get("sliding_attention", {}).get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base_swa(local_rope_theta)
            logger.info(f"gguf: rope theta swa = {local_rope_theta}")
        if (f_rms_eps := self.find_hparam(["rms_norm_eps", "norm_eps"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.find_hparam(["num_local_experts", "num_experts"], optional=True)) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.find_hparam(["num_experts_per_tok", "num_experts_per_token"], optional=True)) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")
        if (n_expert_groups := self.hparams.get("n_group")) is not None:
            self.gguf_writer.add_expert_group_count(n_expert_groups)
            logger.info(f"gguf: expert groups count = {n_expert_groups}")
        if (n_group_used := self.hparams.get("topk_group")) is not None:
            self.gguf_writer.add_expert_group_used_count(n_group_used)
            logger.info(f"gguf: expert groups used count = {n_group_used}")

        if (score_func := self.find_hparam(["score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func"], optional=True)) is not None:
            if score_func == "sigmoid":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
            elif score_func == "softmax":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
            else:
                raise ValueError(f"Unsupported expert score gating function value: {score_func}")
            logger.info(f"gguf: expert score gating function = {score_func}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")
        self.gguf_writer.add_bool("is_hmm", True)
        if self.hmm_info:
            self.gguf_writer.add_string("hmm.info", self.hmm_info)

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError('Splitting the vocabulary is not supported')

        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        # Some models mark some added tokens which ought to be control tokens as not special.
        # (e.g. command-r, command-r-plus, deepseek-coder, gemma{,-2})
        seems_special = token_text in (
            "<pad>",  # deepseek-coder
            "<mask>", "<2mass>", "[@BOS@]",  # gemma{,-2}
        )

        seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
        seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))  # deepseek-coder

        # TODO: should these be marked as UNUSED instead? (maybe not)
        seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))  # gemma{,-2}

        return seems_special

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    # The tokenizer in llama.cpp assumes the CONTROL and USER_DEFINED tokens are pre-normalized.
                    # To avoid unexpected issues - we make sure to normalize non-normalized tokens
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                        if previous_token != token:
                            logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer")

                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        # NOTE: this was added for Gemma.
                        # Encoding and decoding the tokens above isn't sufficient for this case.
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert_hf_to_gguf_update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggml-org/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2":
            # ref: https://huggingface.co/THUDM/glm-4-9b-hf
            res = "glm4"
        if chkhsh == "9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902":
            # ref: https://huggingface.co/zai-org/GLM-4.5-Air
            res = "glm4"
        if chkhsh == "cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267":
            # ref: https://huggingface.co/zai-org/GLM-4.7-Flash
            res = "glm4"
        if chkhsh == "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35":
            # ref: https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0
            res = "minerva-7b"
        if chkhsh == "7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664":
            # ref: https://huggingface.co/tencent/Hunyuan-A13B-Instruct
            res = "hunyuan"
        if chkhsh == "bba3b3366b646dbdded5dbc42d59598b849371afc42f7beafa914afaa5b70aa6":
            # ref: https://huggingface.co/tencent/Hunyuan-4B-Instruct
            res = "hunyuan-dense"
        if chkhsh == "a6b57017d60e6edb4d88ecc2845188e0eb333a70357e45dcc9b53964a73bbae6":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base
            res = "falcon-h1"
        if chkhsh == "60476e1243776c4fb1b993dbd7a5f15ac22f83c80afdf425fa5ae01c8d44ef86":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-1B-Base
            res = "falcon-h1"
        if chkhsh == "3eda48b4c4dc7de733d1a8b3e3b4a85243dbbf704da2ee9d42c6beced8897896":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-7B-Base
            res = "falcon-h1"
        if chkhsh == "48f8e02c0359c0bbdd82f26909171fac1c18a457bb47573ed1fe3bbb2c1cfd4b":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-34B-Base
            res = "falcon-h1"
        if chkhsh == "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890":
            # ref: https://huggingface.co/moonshotai/Kimi-K2-Base
            res = "kimi-k2"
        if chkhsh == "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c":
            # ref: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
            res = "qwen2"
        if chkhsh == "66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273":
            # ref: https://huggingface.co/alvarobartt/grok-2-tokenizer
            res = "grok-2"
        if chkhsh == "b3d1dd861f1d4c5c0d2569ce36baf3f90fe8a102db3de50dd71ff860d91be3df":
            # ref: https://huggingface.co/aari1995/German_Semantic_V3
            res = "jina-v2-de"
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e":
            # ref: https://huggingface.co/tiiuae/Falcon3-7B-Base
            res = "falcon3"
        if chkhsh == "8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7":
            # ref: https://huggingface.co/BAAI/bge-large-zh-v1.5
            res = "bert-bge-large"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "d772b220ace2baec124bed8cfafce0ead7d6c38a4b65ef11261cf9d5d62246d1":
            # ref: https://huggingface.co/CohereLabs/tiny-aya-base
            res = "tiny_aya"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448":
            # ref: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
            res = "jina-v1-en"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "a023e9fdc5a11f034d3ef515b92350e56fb2af1f66c6b6811a4444ea9bf8763d":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano
            res = "jina-v5-nano"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"
        if chkhsh == "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee":
            # ref: https://huggingface.co/LumiOpen/Viking-7B
            res = "viking"
        if chkhsh == "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901":
            # ref: https://huggingface.co/core42/jais-13b
            res = "jais"
        if chkhsh == "bc5108ee1eb6a3d600cadd065f63190fbd0554dbc9e4bbd6a0d977970afc8d2a":
            # ref: https://huggingface.co/inceptionai/Jais-2-8B-Chat
            res = "jais-2"
        if chkhsh == "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f":
            # ref: https://huggingface.co/WisdomShell/CodeShell-7B
            res = "codeshell"
        if chkhsh == "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e":
            # ref: https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
            res = "tekken"
        if chkhsh == "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249":
            # ref: https://huggingface.co/HuggingFaceTB/SmolLM-135M
            res = "smollm"
        if chkhsh == "3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7":
            # ref: https://huggingface.co/bigscience/bloom
            res = "bloom"
        if chkhsh == "bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21":
            # ref: https://huggingface.co/TurkuNLP/gpt3-finnish-small
            res = "gpt3-finnish"
        if chkhsh == "4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae":
            # ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
            res = "exaone"
        if chkhsh == "fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085":
            # ref: https://huggingface.co/microsoft/phi-2
            res = "phi-2"
        if chkhsh == "60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450":
            # ref: https://huggingface.co/facebook/chameleon-7b
            res = "chameleon"
        if chkhsh == "8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65":
            # ref: https://huggingface.co/sentence-transformers/stsb-roberta-base
            res = "roberta-bpe"
        if chkhsh == "ad851be1dba641f2e3711822f816db2c265f788b37c63b4e1aeacb9ee92de8eb":
            # ref: https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct
            res = "gigachat"
        if chkhsh == "d4c8f286ea6b520b3d495c4455483cfa2302c0cfcd4be05d781b6a8a0a7cdaf1":
            # ref: https://huggingface.co/Infinigence/Megrez-3B-Instruct
            res = "megrez"
        if chkhsh == "877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-V3
            res = "deepseek-v3"
        if chkhsh == "b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
            res = "deepseek-r1-qwen"
        if chkhsh == "ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e":
            # ref: https://huggingface.co/Xenova/gpt-4o
            res = "gpt-4o"
        if chkhsh == "7dec86086fcc38b66b7bc1575a160ae21cf705be7718b9d5598190d7c12db76f":
            # ref: https://huggingface.co/UW/OLMo2-8B-SuperBPE-t180k
            res = "superbpe"
        if chkhsh == "1994ffd01900cfb37395608534236ecd63f2bd5995d6cb1004dda1af50240f15":
            # ref: https://huggingface.co/trillionlabs/Trillion-7B-preview
            res = "trillion"
        if chkhsh == "96a5f08be6259352137b512d4157e333e21df7edd3fcd152990608735a65b224":
            # ref: https://huggingface.co/inclusionAI/Ling-lite
            res = "bailingmoe"
        if chkhsh == "d353350c764d8c3b39c763113960e4fb4919bea5fbf208a0e3b22e8469dc7406":
            # ref: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
            res = "llama4"
        if chkhsh == "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3":
            # ref: https://huggingface.co/mistral-community/pixtral-12b
            res = "pixtral"
        if chkhsh == "d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec":
            # ref: https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base
            res = "seed-coder"
        if chkhsh == "b0a6b1c0bd5998ebd9df08611efde34a4ff03faed45ae09c43e6b31ebd4b94cf":
            # ref: https://huggingface.co/skt/A.X-4.0
            res = "a.x-4.0"
        if chkhsh == "f6791d196f87ce6b56a7d234be618e0d58f8cda3549416635b2bebcd22cd95c4":
            # ref: https://huggingface.co/K-intelligence/Midm-2.0-Base-Instruct
            res = "midm-2.0"
        if chkhsh == "169bf0296a13c4d9b7672313f749eb36501d931022de052aad6e36f2bf34dd51":
            # ref: https://huggingface.co/LiquidAI/LFM2-Tokenizer
            res = "lfm2"
        if chkhsh == "2085e1638f6c377a0aa4ead21b27bb4cb941bf800df86ed391011769c1758dfb":
            # ref: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
            res = "exaone4"
        if chkhsh == "a1e163ecab2e718a4c829d1148b6e86824ec36163bb71941c3dca9cd5ac25756":
            # ref: https://huggingface.co/JetBrains/Mellum-4b-base
            res = "mellum"
        if chkhsh == "a0b64b4385f123663873756336c085744376d015ff328bb1d901598f63c44152":
            # ref: https://huggingface.co/answerdotai/ModernBERT-base
            res = "modern-bert"
        if chkhsh == "49fc0303c9e0d2c2c565c510f64b2d9b271276acdcdadff733249eda9f7d59df":
            # ref: https://huggingface.co/arcee-ai/Trinity-Tokenizer
            res = "afmoe"
        if chkhsh == "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206":
            # ref: https://huggingface.co/inclusionAI/Ling-mini-base-2.0
            res = "bailingmoe2"
        if chkhsh == "53e325976a6e142379c19b09afcae354f2f496f147afa8f9e189a33fe4e3024e":
            # ref: https://huggingface.co/ibm-granite/granite-docling-258M
            res = "granite-docling"
        if chkhsh == "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95":
            # ref: https://huggingface.co/MiniMaxAI/MiniMax-M2
            res = "minimax-m2"
        if chkhsh == "4a2e2abae11ca2b86d570fc5b44be4d5eb5e72cc8f22dd136a94b37da83ab665":
            # ref: https://huggingface.co/KORMo-Team/KORMo-tokenizer
            res = "kormo"
        if chkhsh == "9d70134b369a70e5735009b6de918f7581b5211f7c074d1f89f753aea8248af1":
            # ref: https://huggingface.co/tencent/Youtu-LLM-2B
            res = "youtu"
        if chkhsh == "16389f0a1f51ee53e562ffd51c371dc508639ab0e4261502071836e50e223e91":
            # ref: https://huggingface.co/upstage/Solar-Open-100B
            res = "solar-open"
        if chkhsh == "6c81ce329e0802883b22eabab0d3fa48357337ef1ecb45443828bf1f6254833f":
            # ref: https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
            res = "exaone-moe"
        if chkhsh == "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4":
            # ref: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct
            res = "qwen35"
        if chkhsh == "b4b8ca1f9769494fbd956ebc4c249de6131fb277a4a3345a7a92c7dd7a55808d":
            # ref: https://huggingface.co/jdopensource/JoyAI-LLM-Flash
            res = "joyai-llm"
        if chkhsh == "e4d54df1ebc1f2b91acd986c5b51aa50837d5faf7c7398e73c1f9e9ee5d19869":
            # ref: https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct-2601
            res = "kanana2"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_none(self) -> None:
        self.gguf_writer.add_tokenizer_model("none")

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self, add_to_gguf=True):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _create_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.find_hparam([
            "vocab_size_per_layer_input", # gemma3n
            "vocab_size",
        ], optional=True) or tokenizer.vocab_size()

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            if token_id >= vocab_size:
                logger.warning(f'ignore tokens from {token_id}: id is out of range, max={vocab_size - 1}')
                break

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, token_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token: str = token_data["content"]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token.encode("utf-8"):
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token!r}')
                    if token_data.get("special") or self.does_token_look_special(token):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

                    scores[token_id] = -1000.0
                    tokens[token_id] = token.encode("utf-8")

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        return tokens, scores, toktypes

    def _set_vocab_llama_hf(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_rwkv_world(self):
        assert (self.dir_model / "rwkv_vocab_v20230424.txt").is_file()
        vocab_size = self.hparams.get("vocab_size", 65536)

        tokens: list[bytes] = ['<s>'.encode("utf-8")]
        toktypes: list[int] = [gguf.TokenType.CONTROL]

        with open(self.dir_model / "rwkv_vocab_v20230424.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(' ')
                assert len(parts) >= 3
                token, token_len = ast.literal_eval(' '.join(parts[1:-1])), int(parts[-1])
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == token_len
                token_text: str = repr(token)[2:-1]  # "b'\xff'" -> "\xff"
                tokens.append(token_text.encode("utf-8"))
                toktypes.append(gguf.TokenType.NORMAL)
        remainder = vocab_size - len(tokens)
        assert remainder >= 0
        for i in range(len(tokens), vocab_size):
            tokens.append(f"[PAD{i}]".encode("utf-8"))
            toktypes.append(gguf.TokenType.UNUSED)

        self.gguf_writer.add_tokenizer_model("rwkv")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        if special_vocab.chat_template is None:
            template_path = Path(__file__).parent / "models" / "templates" / "llama-cpp-rwkv-world.jinja"
            if template_path.is_file():
                with open(template_path, "r", encoding="utf-8") as f:
                    template = f.read()
            else:
                template = "rwkv-world"
            special_vocab.chat_template = template
        # hack: Add '\n\n' as the EOT token to make it chat normally
        special_vocab._set_special_token("eot", 261)
        # hack: Override these as they have already been set (incorrectly)
        special_vocab.special_token_ids["bos"] = 0
        special_vocab.special_token_ids["eos"] = 0

        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_builtin(self, model_name: Literal["gpt-neox", "llama-spm"], vocab_size: int):
        tokenizer_path = Path(sys.path[0]) / "models" / f"ggml-vocab-{model_name}.gguf"
        logger.warning(f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'")
        vocab_reader = gguf.GGUFReader(tokenizer_path, "r")

        default_pre = "mpt" if model_name == "gpt-neox" else "default"

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.MODEL)
        assert field  # tokenizer model
        self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]).decode("utf-8"))

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.PRE)
        self.gguf_writer.add_tokenizer_pre(bytes(field.parts[-1]).decode("utf-8") if field else default_pre)

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.LIST)
        assert field  # token list
        self.gguf_writer.add_token_list([bytes(field.parts[i]) for i in field.data][:vocab_size])

        if model_name == "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.SCORES)
            assert field  # token scores
            self.gguf_writer.add_token_scores([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
        assert field  # token types
        self.gguf_writer.add_token_types([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

        if model_name != "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            assert field  # token merges
            self.gguf_writer.add_token_merges([bytes(field.parts[i]) for i in field.data])

        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)) is not None:
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)) is not None:
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)) is not None:
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.PAD_ID)) is not None:
            self.gguf_writer.add_pad_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)) is not None:
            self.gguf_writer.add_add_bos_token(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_EOS)) is not None:
            self.gguf_writer.add_add_eos_token(field.parts[-1].tolist()[0])

    def _try_set_pooling_type(self) -> None:
        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break

        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            elif pooling["pooling_mode_lasttoken"]:
                pooling_type = gguf.PoolingType.LAST
            else:
                raise NotImplementedError("Only MEAN, CLS, and LAST pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)

    def _set_vocab_glmedge(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_glm(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        # Special tokens
        # Note: Using <|endoftext|> (151329) for eot causes endless generation
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["[gMASK]"])  # 151331
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # 151336
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"]) # 151329
        special_vocab._set_special_token("eom", tokenizer.get_added_vocab()["<|observation|>"])  # 151338
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_interns1(self):
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        vocab = getattr(tokenizer, 'vocab', tokenizer.get_vocab())
        vocab_size = self.hparams.get("vocab_size", len(vocab))
        assert max(vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    # The tokenizer in llama.cpp assumes the CONTROL and USER_DEFINED tokens are pre-normalized.
                    # To avoid unexpected issues - we make sure to normalize non-normalized tokens
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                        if previous_token != token:
                            logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer")

                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab._set_special_token("bos", 151643)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_mistral(self):
        if not _mistral_common_installed:
            raise ImportError(_mistral_import_error_msg)

        vocab = MistralVocab(self.dir_model)
        logger.info(
            f"Converting tokenizer {vocab.tokenizer_type} of size {vocab.vocab_size}."
        )

        self.gguf_writer.add_tokenizer_model(vocab.gguf_tokenizer_model)

        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size, (
            f"token count ({len(tokens)}) != vocab size ({vocab.vocab_size})"
        )

        if vocab.tokenizer_type == MistralTokenizerType.tekken:
            self.gguf_writer.add_tokenizer_pre("tekken")
            self.gguf_writer.add_token_merges(
                vocab.extract_vocab_merges_from_model()
            )

        logger.info(
            f"Setting bos, eos, unk and pad token IDs to {vocab.bos_id}, {vocab.eos_id}, {vocab.unk_id}, {vocab.pad_id}."
        )

        self.gguf_writer.add_bos_token_id(vocab.bos_id)
        self.gguf_writer.add_eos_token_id(vocab.eos_id)
        self.gguf_writer.add_unk_token_id(vocab.unk_id)
        self.gguf_writer.add_pad_token_id(vocab.pad_id)

        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_vocab_size(vocab.vocab_size)

        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(False)

        local_template_file_path = self.dir_model / "chat_template.jinja"

        if self.is_mistral_format and local_template_file_path.is_file():
            # Ministral-3 and other new Mistral models come with chat templates.
            # ref: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/tree/main
            logger.info("Using an existing Mistral local chat template.")

            with open(local_template_file_path, "r", encoding="utf-8") as f:
                template = f.read()
        elif not self.is_mistral_format or not self.disable_mistral_community_chat_template:
            template_dir = Path(__file__).parent / "models/templates/"

            # Log only for Mistral format that the official tokenization and detokenization is via `mistral-common`.
            if self.is_mistral_format:
                logger.info(
                    "Using a Mistral community chat template. These templates can be subject to errors in early days or weeks after a release. "
                    "Mistral recommends to use `mistral-common` to perform tokenization and detokenization."
                )
            template = MistralModel.get_community_chat_template(vocab, template_dir, self.is_mistral_format)
        else:
            logger.info("Not using a Mistral local or community chat template. Ensure to perform the tokenization and detokenization via `mistral-common`.")
            template = None

        if template is not None:
            self.gguf_writer.add_chat_template(template)

    def _set_vocab_plamo(self):
        # PLaMo models use a custom tokenizer with a .jsonl file
        tokenizer_jsonl_path = self.dir_model / "tokenizer.jsonl"
        tokenizer_config_path = self.dir_model / "tokenizer_config.json"

        if not tokenizer_jsonl_path.is_file():
            raise FileNotFoundError(f"PLaMo tokenizer file not found: {tokenizer_jsonl_path}")

        # Load tokenizer config
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        # Load tokens from JSONL file (actually a list format)
        tokens = []
        scores = []
        toktypes = []

        with open(tokenizer_jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    token_data = json.loads(line)
                    # Format: [token, score, type, ?, ?, ?, ?]
                    token = token_data[0].encode("utf-8")
                    score = float(token_data[1])
                    token_type_str = token_data[2] if len(token_data) > 2 else "NORMAL"

                    tokens.append(token)
                    scores.append(score)

                    if token_type_str == "UNKNOWN":
                        toktypes.append(gguf.TokenType.UNKNOWN)
                    elif token_type_str == "CONTROL":
                        toktypes.append(gguf.TokenType.CONTROL)
                    elif token_type_str == "BYTE":
                        toktypes.append(gguf.TokenType.BYTE)
                    else:
                        token_str = token_data[0]
                        if token_str.startswith("<|plamo:") and token_str.endswith("|>"):
                            toktypes.append(gguf.TokenType.CONTROL)
                        else:
                            toktypes.append(gguf.TokenType.NORMAL)

        vocab_size = self.hparams["vocab_size"]
        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(gguf.TokenType.UNUSED)

        self.gguf_writer.add_tokenizer_model("plamo2")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        if "bos_token" in tokenizer_config and tokenizer_config["bos_token"] is not None:
            token_id = tokens.index(tokenizer_config["bos_token"].encode("utf-8"))
            self.gguf_writer.add_bos_token_id(token_id)
        if "eos_token" in tokenizer_config and tokenizer_config["eos_token"] is not None:
            token_id = tokens.index(tokenizer_config["eos_token"].encode("utf-8"))
            self.gguf_writer.add_eos_token_id(token_id)
        if "pad_token" in tokenizer_config and tokenizer_config["pad_token"] is not None:
            token_id = tokens.index(tokenizer_config["pad_token"].encode("utf-8"))
            self.gguf_writer.add_pad_token_id(token_id)
        if "sep_token" in tokenizer_config and tokenizer_config["sep_token"] is not None:
            token_id = tokens.index(tokenizer_config["sep_token"].encode("utf-8"))
            self.gguf_writer.add_sep_token_id(token_id)
        if "unk_token" in tokenizer_config and tokenizer_config["unk_token"] is not None:
            token_id = tokens.index(tokenizer_config["unk_token"].encode("utf-8"))
            self.gguf_writer.add_unk_token_id(token_id)

        # Add <|plamo:op|> as EOT to ensure appropriate end of generation
        self.gguf_writer.add_eot_token_id(4)

        self.gguf_writer.add_add_space_prefix(False)



@ModelBase.register("MiniCPMForCausalLM")
class MiniCPMModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MINICPM

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        embedding_scale = float(self.hparams["scale_emb"])
        self.gguf_writer.add_embedding_scale(embedding_scale)
        logger.info(f"gguf: (minicpm) embedding_scale = {embedding_scale}")
        residual_scale = (
            self.hparams["scale_depth"] / self.hparams["num_hidden_layers"] ** 0.5
        )
        self.gguf_writer.add_residual_scale(residual_scale)
        logger.info(f"gguf: (minicpm) residual_scale = {residual_scale}")
        logit_scale = self.hparams["hidden_size"] / self.hparams["dim_model_base"]
        self.gguf_writer.add_logit_scale(logit_scale)
        logger.info(f"gguf: (minicpm) logit_scale = {logit_scale}")
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "longrope":
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LONGROPE)
            logger.info(
                f"gguf: (minicpm) rope_scaling_type = {gguf.RopeScalingType.LONGROPE}"
            )

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )

    def set_vocab(self):
        self._set_vocab_sentencepiece()


@ModelBase.register("MiniCPM3ForCausalLM")
class MiniCPM3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINICPM3

    def set_gguf_parameters(self):
        hparams = self.hparams

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(
            hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def _reverse_hf_permute(
        self, weights: Tensor, n_head: int, n_kv_head: int | None = None
    ) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(
                n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )


@ModelBase.register("GptOssForCausalLM")
class GptOssModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT_OSS

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_expert_feed_forward_length(
            self.hparams["intermediate_size"]
        )

        rope_scaling = self.hparams.get("rope_scaling") or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        assert (
            rope_type == "yarn"
        ), f"GPT-OSS only supports yarn rope scaling, got {rope_type}"
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
        self.gguf_writer.add_rope_scaling_orig_ctx_len(
            rope_scaling.get("original_max_position_embeddings", 4096)
        )

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]

    n_block_keys = ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth"]

    has_vision_encoder: bool = True  # by default
    has_audio_encoder: bool = False

    # for models having multiple encoders, we need to separate their hparams
    hparams_vision: dict[str, Any] | None = None
    hparams_audio: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model_arch != gguf.MODEL_ARCH.MMPROJ:
            raise TypeError(
                "MmprojModel must be subclassed with model_arch = gguf.MODEL_ARCH.MMPROJ"
            )

        # get n_embd of the text model
        if "text_config" not in self.hparams:
            self.hparams["text_config"] = {}
        if "audio_config" not in self.hparams:
            self.hparams["audio_config"] = {}
        text_config = {**self.hparams, **self.hparams["text_config"]}
        self.n_embd_text = text_config.get("hidden_size", text_config.get("n_embd", 0))
        assert self.n_embd_text > 0, "n_embd not found in hparams"

        # move vision config to the top level, while preserving the original hparams in global_config
        import copy

        self.global_config = copy.deepcopy(self.hparams)
        self.hparams_vision = self.get_vision_config()
        self.hparams_audio = self.get_audio_config()

        if self.hparams_vision is None and self.hparams_audio is None:
            raise ValueError("vision_config / audio_config not found in hparams")

        # for compat with vision-only models
        self.hparams = self.hparams_vision or self.hparams_audio or self.hparams

        # TODO @ngxson : this is a hack to support both vision and audio encoders
        have_multiple_encoders = self.has_audio_encoder and self.has_vision_encoder
        self.block_count = (
            128 if have_multiple_encoders else self.find_hparam(self.n_block_keys, True)
        )
        self.tensor_map = gguf.get_tensor_name_map(
            gguf.MODEL_ARCH.MMPROJ, self.block_count
        )

        # load preprocessor config
        with open(
            self.dir_model / "preprocessor_config.json", "r", encoding="utf-8"
        ) as f:
            self.preprocessor_config = json.load(f)

    def get_vision_config(self) -> dict[str, Any] | None:
        return self.global_config.get("vision_config")

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MMPROJ)

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        if self.has_vision_encoder:
            self.gguf_writer.add_clip_has_vision_encoder(True)
            self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

            # vision config
            self.gguf_writer.add_vision_image_size(self.find_vparam(["image_size"]))
            self.gguf_writer.add_vision_patch_size(self.find_vparam(["patch_size"]))
            self.gguf_writer.add_vision_embedding_length(
                self.find_vparam(["hidden_size"])
            )
            self.gguf_writer.add_vision_feed_forward_length(
                self.find_vparam(["intermediate_size"])
            )
            self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
            self.gguf_writer.add_vision_head_count(
                self.find_vparam(["num_attention_heads"])
            )

            # preprocessor config
            self.gguf_writer.add_vision_image_mean(
                self.preprocessor_config["image_mean"]
            )
            self.gguf_writer.add_vision_image_std(self.preprocessor_config["image_std"])

        if self.has_audio_encoder:
            self.gguf_writer.add_clip_has_audio_encoder(True)
            self.gguf_writer.add_audio_projection_dim(self.n_embd_text)

            # audio config
            self.gguf_writer.add_audio_embedding_length(
                self.find_aparam(["hidden_size"])
            )
            self.gguf_writer.add_audio_feed_forward_length(
                self.find_aparam(["intermediate_size"])
            )
            self.gguf_writer.add_audio_block_count(self.find_aparam(self.n_block_keys))
            self.gguf_writer.add_audio_head_count(
                self.find_aparam(["num_attention_heads"])
            )

        if not self.has_vision_encoder and not self.has_audio_encoder:
            raise ValueError("MmprojModel must have either vision or audio encoder")
        self.gguf_writer.add_bool("is_hmm", True)
        if self.hmm_info:
            self.gguf_writer.add_string("hmm.info", self.hmm_info)

    def write_vocab(self):
        raise ValueError("MmprojModel does not support vocab writing")

    def find_vparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_vision is not None
        return self._find_param(self.hparams_vision, keys, optional)

    def find_aparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_audio is not None
        return self._find_param(self.hparams_audio, keys, optional)

    def _find_param(
        self, obj: dict[str, Any], keys: Iterable[str], optional: bool = False
    ) -> Any:
        key = next((k for k in keys if k in obj), None)
        if key is not None:
            return obj[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        # TODO: @guoxing.xu check & test this block
        logger.info(".....Getting tensors")
        pattern = "*visual.hmm"
        # 遍历目录下匹配的所有文件（仅文件，不包含子目录）
        for file_path in self.dir_model.glob(pattern):
            logger.info(f".....Loading tensor from {file_path}")

            if not file_path.is_file():
                logger.warning(f".....{file_path} is not a file, skipping")
                continue
            hmm_info = read_hmm_info(file_path)
            self.hmm_info = json.dumps(hmm_info.__dict__)
            try:
                # 读取二进制数据并转换为 torch.Tensor
                with open(file_path, "rb") as file:
                    binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                    logger.info(f".....Tensor size: {binary_data.shape}")
                    data = torch.from_numpy(binary_data.copy())
                    yield "visual.hmm", data
            except Exception as e:
                logger.error(
                    f".....Failed to load {file_path}: {str(e)}", exc_info=True
                )
                continue
                # 遍历目录下所有的json文件，只返回不带路径的文件名
        for json_file in glob.glob(os.path.join(self.dir_model, "*.json")):
            with open(json_file, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield os.path.basename(json_file), data
        for json_file in glob.glob(os.path.join(self.dir_model, "*.txt")):
            with open(json_file, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield os.path.basename(json_file), data
    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register("QWenLMHeadModel")
class QwenModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        byte_encoder = bytes_to_unicode()
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    @staticmethod
    def bpe(
        mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None
    ) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = (
                parts[:min_idx]
                + [parts[min_idx] + parts[min_idx + 1]]
                + parts[min_idx + 2 :]
            )
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(
            self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        )
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@ModelBase.register(
    "Qwen2Model", "Qwen2ForCausalLM", "Qwen2AudioForConditionalGeneration"
)
class Qwen2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (
            self.hparams.get("rope_scaling") is not None
            and "factor" in self.hparams["rope_scaling"]
        ):
            if self.hparams["rope_scaling"].get("type") == "yarn":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
                self.gguf_writer.add_rope_scaling_factor(
                    self.hparams["rope_scaling"]["factor"]
                )
                self.gguf_writer.add_rope_scaling_orig_ctx_len(
                    self.hparams["rope_scaling"]["original_max_position_embeddings"]
                )

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register(
    "Qwen2VLModel",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5OmniModel",
)
class Qwen2VLModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        mrope_section = self.hparams["rope_scaling"]["mrope_section"]
        mrope_section += [0] * max(0, 4 - len(mrope_section))
        self.gguf_writer.add_rope_dimension_sections(mrope_section)
        self.gguf_writer.add_uint32("hmm.vit.height", self.hmm_vit_height)
        self.gguf_writer.add_uint32("hmm.vit.width", self.hmm_vit_width)

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register(
    "Qwen2VLModel",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
)
class Qwen2VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = self.hparams_vision.get("image_size", 560)
        # rename config.json values
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get(
            "num_heads"
        )
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")
        if "embed_dim" in self.hparams_vision:  # qwen2vl
            self.hparams_vision["intermediate_size"] = self.hparams_vision.get(
                "hidden_size"
            )
            self.hparams_vision["hidden_size"] = self.hparams_vision.get("embed_dim")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        hparams = self.hparams_vision
        model_type = self.global_config["model_type"]
        if model_type == "qwen2_vl":
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2VL)
        elif model_type == "qwen2_5_vl" or model_type == "qwen2_5_omni":
            if model_type == "qwen2_5_omni":
                self.gguf_writer.add_clip_projector_type(
                    gguf.VisionProjectorType.QWEN25O
                )
            else:
                self.gguf_writer.add_clip_projector_type(
                    gguf.VisionProjectorType.QWEN25VL
                )
            self.gguf_writer.add_vision_use_silu(True)
            # find n_wa_pattern (window attention pattern)
            fullatt_block_indexes = hparams.get("fullatt_block_indexes")
            assert (
                fullatt_block_indexes is not None
            ), "fullatt_block_indexes is required for qwen2_5_vl"
            n_wa_pattern = fullatt_block_indexes[0] + 1
            # validate n_wa_pattern
            for i in range(1, len(fullatt_block_indexes)):
                if (
                    fullatt_block_indexes[i] - fullatt_block_indexes[i - 1]
                    != n_wa_pattern
                ):
                    raise ValueError(
                        f"Invalid fullatt_block_indexes: {fullatt_block_indexes}"
                    )
            self.gguf_writer.add_vision_n_wa_pattern(n_wa_pattern)
        else:
            raise ValueError(
                f"Unknown QwenVL model type: {self.global_config['model_type']}"
            )
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.global_config.get("rms_norm_eps", 1e-6)
        )


@ModelBase.register("Qwen2_5OmniModel")
class Qwen25OmniModel(Qwen2VLVisionModel):
    has_vision_encoder = True
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_audio is not None
        self.hparams_audio["hidden_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio["encoder_ffn_dim"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio[
            "encoder_attention_heads"
        ]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_audio is not None
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(
            self.hparams_audio.get("layer_norm_eps", 1e-5)
        )

    def get_vision_config(self) -> dict[str, Any] | None:
        return self.global_config["thinker_config"].get("vision_config")

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config["thinker_config"].get("audio_config")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # SinusoidsPositionEmbedding
        assert self.hparams_audio is not None
        max_timescale = 10000
        length = 1500
        channels = self.hparams_audio["hidden_size"]
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        pos_embd = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        ).to(dtype=torch.float32)
        yield ("audio_tower.embed_positions.weight", pos_embd)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del bid, new_name, n_dims  # unused
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return False

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("thinker."):
            name = name.replace("thinker.", "")

        if name.startswith("audio_tower"):
            # process audio tensors
            if "conv1.bias" in name or "conv2.bias" in name:
                # transpose conv1 and conv2 bias
                data_torch = data_torch.unsqueeze(-1)
            if "audio_bos_eos_token" in name:
                # this tensor is left unused in transformers code
                # https://github.com/huggingface/transformers/blob/6e3063422c4b1c014aa60c32b9254fd2902f0f28/src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py#L1809
                return []
            return [(self.map_tensor_name(name), data_torch)]

        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        name = name.replace("language_model.", "") # InternVL

        # handle aggregated expert tensors
        # GGUF stores dimensions reversed from PyTorch, so:
        # PyTorch (A,B,C) -> GGUF writes [C,B,A] -> GGML reads ne={C,B,A}
        # Input shapes from HF: (n_expert, n_ff_exp, n_embd) or (n_expert, n_embd, n_ff_exp)
        # Expected GGML ne: {n_embd, n_ff_exp, n_expert} for gate/up, {n_ff_exp, n_embd, n_expert} for down
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            mapped = f"{name}.weight" if not name.endswith(".weight") else name
            # HF: [n_expert, n_embd, n_ff] -> GGML: {n_ff, n_embd, n_expert}
            yield from super().modify_tensors(data_torch, mapped, bid)
            return

        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            if data_torch.ndim < 3 or data_torch.shape[-2] % 2 != 0:
                raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
            # HF: [n_expert, 2*n_ff, n_embd] -> split on dim=-2
            n_ff = data_torch.shape[-2] // 2
            gate = data_torch[..., :n_ff, :].contiguous()
            up = data_torch[..., n_ff:, :].contiguous()
            # gate/up: [n_expert, n_ff, n_embd] -> GGML: {n_embd, n_ff, n_expert}
            base_name = name.removesuffix(".weight").removesuffix(".gate_up_proj")
            mapped_gate = f"{base_name}.gate_proj.weight"
            mapped_up = f"{base_name}.up_proj.weight"
            yield from super().modify_tensors(gate, mapped_gate, bid)
            yield from super().modify_tensors(up, mapped_up, bid)
            return

        if name.startswith("mlp") or name.startswith("vision_model") or name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") or name.startswith("model.visual"):
            # skip visual tensors
            return

        if name.find("experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)


    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )


@ModelBase.register("Qwen3ForCausalLM")
class Qwen3Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN3


@ModelBase.register("Qwen3MoeForCausalLM")
class Qwen3MoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = ModelBase.load_hparams(self.dir_model)
        self.origin_hf_arch = hparams.get("architectures", [None])[0]

    def set_vocab(self):
        # deal with intern-s1
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return

        super().set_vocab()

@ModelBase.register("Qwen3NextForCausalLM")
class Qwen3NextModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3NEXT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_ssm_conv_kernel(self.hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(self.hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(self.hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(self.hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(self.hparams["linear_value_head_dim"] * self.hparams["linear_num_value_heads"])
        self.gguf_writer.add_full_attention_interval(self.hparams.get("full_attention_interval", 4))
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.25)))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("mtp"):
            return  # ignore MTP layers for now
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif "conv1d" in name:
            data_torch = data_torch.squeeze()
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            data_torch = data_torch + 1

        if "in_proj_qkvz.weight" in name:
            # original order:  [q, k, v, z] * head_count
            # corrected order: [q * head_count, k * head_count, v * head_count, z * head_count]
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            num_v_heads = self.hparams["linear_num_value_heads"]
            num_k_heads = self.hparams["linear_num_key_heads"]
            hidden_size = self.hparams["hidden_size"]
            split_arg_list_qkvz = [
                head_k_dim, # q partition
                head_k_dim, # k partition
                (num_v_heads // num_k_heads * head_v_dim), # v partition
                (num_v_heads // num_k_heads * head_v_dim), # z partition
            ]
            # view as (n_embd, head_count, [q+k+v+z])
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(-1, num_k_heads, sum(split_arg_list_qkvz))
            # split into q, k, v, z
            q, k, v, z = torch.split(data_torch, split_arg_list_qkvz, dim=-1)
            # flatten dim + head_count
            q = q.contiguous().view(hidden_size, -1)
            k = k.contiguous().view(hidden_size, -1)
            v = v.contiguous().view(hidden_size, -1)
            z = z.contiguous().view(hidden_size, -1)
            # stack back
            qkv = torch.cat([q, k, v], dim=-1).permute(1, 0).contiguous()
            z = z.permute(1, 0).contiguous()
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV,  bid, ".weight"), qkv)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"), z)
        else:
            yield from super().modify_tensors(data_torch, name, bid)



@ModelBase.register("Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration")

class Qwen3VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # Compute image_size if not present
        if "image_size" not in self.hparams_vision:
            # For Qwen3VL/Qwen3VLMoe, compute from num_position_embeddings
            num_pos = self.hparams_vision.get("num_position_embeddings", 2304)
            patch_size = self.hparams_vision.get("patch_size", 16)
            # num_position_embeddings = (image_size / patch_size) ** 2
            # So image_size = sqrt(num_position_embeddings) * patch_size
            image_size = int(num_pos**0.5 * patch_size)
            self.hparams_vision["image_size"] = image_size

        # Rename config values for compatibility
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get(
            "num_heads"
        )
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")

        self.is_deepstack_layers = [False] * int(
            self.hparams_vision["num_hidden_layers"] or 0
        )
        for idx in self.hparams_vision.get("deepstack_visual_indexes", []):
            self.is_deepstack_layers[idx] = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN3VL)
        self.gguf_writer.add_vision_use_gelu(True)

        if self.hparams_vision is not None:
            merge_size = self.hparams_vision.get("spatial_merge_size")
            if merge_size is not None:
                self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

        # Use text config's rms_norm_eps for vision attention layernorm eps
        rms_norm_eps = self.global_config.get("text_config", {}).get(
            "rms_norm_eps", 1e-6
        )
        self.gguf_writer.add_vision_attention_layernorm_eps(rms_norm_eps)

        if self.is_deepstack_layers:
            self.gguf_writer.add_vision_is_deepstack_layers(self.is_deepstack_layers)


@ModelBase.register("Qwen3VLForConditionalGeneration")
class Qwen3VLTextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # Handle MRoPE (Multi-axis Rotary Position Embedding) for Qwen3-VL
        text_config = self.hparams.get("text_config", {})
        # rope_scaling is deprecated in V5, use rope_parameters instead
        rope_scaling = (
            text_config.get("rope_scaling") or text_config.get("rope_parameters") or {}
        )

        if rope_scaling.get("mrope_section"):
            # mrope_section contains [time, height, width] dimensions
            mrope_section = rope_scaling["mrope_section"]
            # Pad to 4 dimensions [time, height, width, extra]
            while len(mrope_section) < 4:
                mrope_section.append(0)
            self.gguf_writer.add_rope_dimension_sections(mrope_section[:4])

            logger.info(f"MRoPE sections: {mrope_section[:4]}")

        vision_config = self.hparams.get("vision_config", {})
        deepstack_layer_num = len(vision_config.get("deepstack_visual_indexes", []))
        self.gguf_writer.add_num_deepstack_layers(deepstack_layer_num)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # Skip vision tensors - they go in the mmproj file
        if name.startswith("model.visual."):
            return []

        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3VLMoeForConditionalGeneration")
class Qwen3VLMoeTextModel(Qwen3MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3VLMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # Handle MRoPE (Multi-axis Rotary Position Embedding) for Qwen3-VL
        text_config = self.hparams.get("text_config", {})
        # rope_scaling is deprecated in V5, use rope_parameters instead
        rope_scaling = (
            text_config.get("rope_scaling") or text_config.get("rope_parameters") or {}
        )

        if rope_scaling.get("mrope_section"):
            # mrope_section contains [time, height, width] dimensions
            mrope_section = rope_scaling["mrope_section"]
            # Pad to 4 dimensions [time, height, width, extra]
            while len(mrope_section) < 4:
                mrope_section.append(0)
            self.gguf_writer.add_rope_dimension_sections(mrope_section[:4])

            logger.info(f"MRoPE sections: {mrope_section[:4]}")

        vision_config = self.hparams.get("vision_config", {})
        deepstack_layer_num = len(vision_config.get("deepstack_visual_indexes", []))
        self.gguf_writer.add_num_deepstack_layers(deepstack_layer_num)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # Skip vision tensors - they go in the mmproj file
        if name.startswith("model.visual."):
            return []

        return super().modify_tensors(data_torch, name, bid)
class _LinearAttentionVReorderBase(Qwen3NextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3NEXT  # overridden by subclasses
    """reorders V heads from grouped to tiled order for ggml broadcast

    see https://github.com/ggml-org/llama.cpp/pull/19468#discussion_r2786394306

    Linear attention may has num_k_heads < num_v_heads. The HF weights store
    V heads grouped by K head: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...].
    ggml binary ops use tiled broadcast: [K0, K1, ..., K0, K1, ...].
    We reorder V heads to tiled order so ggml_repeat can replace the expensive
    interleaved repeat: [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...].
    """

    @staticmethod
    def _reorder_v_heads(tensor: Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> Tensor:
        """Reorder V heads from grouped (by K head) to tiled order along the given dimension."""
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_k_heads = self.hparams.get("linear_num_key_heads", 0)
        num_v_heads = self.hparams.get("linear_num_value_heads", 0)

        if num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads and "linear_attn." in name:
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            num_v_per_k = num_v_heads // num_k_heads

            if ".in_proj_qkv." in name:
                # QKV weight: reorder only the V rows
                q_dim = head_k_dim * num_k_heads
                k_dim = head_k_dim * num_k_heads
                q = data_torch[:q_dim]
                k = data_torch[q_dim:q_dim + k_dim]
                v = data_torch[q_dim + k_dim:]
                v = self._reorder_v_heads(v, 0, num_k_heads, num_v_per_k, head_v_dim)
                data_torch = torch.cat([q, k, v], dim=0)

            elif ".in_proj_z." in name:
                # Z gate weight: reorder rows (num_v_heads * head_v_dim)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, head_v_dim)

            elif ".in_proj_b." in name or ".in_proj_a." in name:
                # Beta/Alpha weight: reorder rows (num_v_heads, head_dim=1)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, 1)

            elif ".A_log" in name or ".dt_bias" in name or ".dt_proj" in name:
                # A_log / dt_bias: 1D parameters with num_v_heads elements
                if data_torch.ndim == 1:
                    data_torch = self._reorder_v_heads(
                        data_torch.unsqueeze(-1), 0, num_k_heads, num_v_per_k, 1
                    ).squeeze(-1)
                else:
                    data_torch = self._reorder_v_heads(data_torch, -1, num_k_heads, num_v_per_k, 1)

            elif ".conv1d" in name:
                # Conv1d kernel: reorder only the V channel portion
                data = data_torch.squeeze()
                qk_channels = head_k_dim * num_k_heads * 2
                qk_part = data[:qk_channels]
                v_part = data[qk_channels:]
                v_part = self._reorder_v_heads(v_part, 0, num_k_heads, num_v_per_k, head_v_dim)
                data_torch = torch.cat([qk_part, v_part], dim=0)

            elif ".out_proj." in name:
                # Out projection weight: reorder columns (input dimension)
                data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, num_v_per_k, head_v_dim)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
class Qwen3_5TextModel(_LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35


@ModelBase.register("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
class Qwen3_5MoeTextModel(_LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE

@ModelBase.register("GPT2LMHeadModel")
class GPT2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        tensors: list[tuple[str, Tensor]] = []

        # we don't need these
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            return tensors

        if name.endswith(
            (".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")
        ):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        tensors.append((new_name, data_torch))

        return tensors


@ModelBase.register("DeepseekForCausalLM")
class DeepseekModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(
            hparams["moe_intermediate_size"]
        )
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (
            weights.reshape(
                n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("DeepseekV2ForCausalLM")
@ModelBase.register("DeepseekV3ForCausalLM")
class DeepseekV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):

        # note: deepseek2 using MLA converts into MQA (ie: GQA with 1 group)
        self.hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])

        # note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
        self.gguf_writer.add_key_length(
            hparams["kv_lora_rank"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(
            hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"]
        )
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(
            hparams["moe_intermediate_size"]
        )
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        if hparams["scoring_func"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif hparams["scoring_func"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(
                f"Unsupported scoring_func value: {hparams['scoring_func']}"
            )

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        rope_scaling = self.hparams.get("rope_scaling") or {}
        if (
            rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn"
            and "factor" in rope_scaling
        ):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(
                rope_scaling["original_max_position_embeddings"]
            )
            self.gguf_writer.add_rope_scaling_yarn_log_mul(
                0.1 * rope_scaling["mscale_all_dim"]
            )

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # rename e_score_correction_bias tensors
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # skip Multi-Token Prediction (MTP) layers
        block_count = self.hparams["num_hidden_layers"]
        match = re.match(r"model.layers.(\d+)", name)
        if match and int(match.group(1)) >= block_count:
            return []

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        # note: MLA with the absorption optimization, needs these two split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(
                n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1]
            )
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            return [
                (self.map_tensor_name(name_kb), k_b),
                (self.map_tensor_name(name_vb), v_b),
            ]

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Qwen2AudioForConditionalGeneration")
class WhisperEncoderModel(MmprojModel):
    has_vision_encoder = False  # no vision encoder
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["hidden_size"] = self.hparams["d_model"]
        self.hparams["intermediate_size"] = self.hparams["encoder_ffn_dim"]
        self.hparams["num_attention_heads"] = self.hparams["encoder_attention_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(
            self.hparams.get("layer_norm_eps", 1e-5)
        )

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del bid, new_name, n_dims  # unused
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return False

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        if name.startswith("language_model."):
            # skip language model tensors
            return []

        # prevent clash naming with vision tensors
        if name.startswith("multi_modal_projector"):
            name = "audio." + name

        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1 and conv2 bias
            data_torch = data_torch.unsqueeze(-1)

        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("BertModel", "BertForMaskedLM", "CamembertModel", "BertForSequenceClassification")
class BertModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BERT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = None

        if cls_out_labels := self.hparams.get("id2label"):
            if len(cls_out_labels) == 2 and cls_out_labels[0] == "LABEL_0":
                # Remove dummy labels added by AutoConfig
                cls_out_labels = None
        self.cls_out_labels = cls_out_labels

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self._try_set_pooling_type()

        if self.cls_out_labels:
            self.gguf_writer.add_classifier_output_labels([v for k, v in sorted(self.cls_out_labels.items())])
        self.gguf_writer.add_bool("is_hmm", True)
        if self.hmm_info:
            self.gguf_writer.add_string("hmm.info", self.hmm_info)

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        # TODO: @guoxing.xu check & test this block
        logger.info(".....Getting tensors")
        for name in [
            "embedding.hmm",
        ]:
            file_path = self.dir_model / name
            logger.info(f".....Loading tensor from {file_path}")
            if not file_path.is_file():
                logger.warning(f".....{file_path} not found")
                continue
            hmm_info = read_hmm_info(file_path)
            self.hmm_info = json.dumps(hmm_info.__dict__)
            with open(file_path, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield "embedding", data
        # 遍历目录下所有的json文件，只返回不带路径的文件名
        for json_file_path in self.dir_model.glob("**/*.json"):
            relative_json_path = json_file_path.relative_to(self.dir_model)
            with open(json_file_path, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                
                yield str(relative_json_path), data
        for json_file in glob.glob(os.path.join(self.dir_model, "*.txt")):
            with open(json_file, "rb") as file:
                binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                logger.info(f".....Tensor size: {binary_data.shape}")
                data = torch.from_numpy(binary_data.copy())
                yield os.path.basename(json_file), data

    def prepare_tensors(self):
        # TODO: @guoxing.xu check & test this block
        # 自定义的tensor，不需要该函数
        # super().prepare_tensors()
        logger.info(".....Preparing tensors")
        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            logger.info(
                f".....Adding tensor {name}, {data_torch.shape}, {data_torch.nbytes}"
            )
            self.gguf_writer.add_tensor(
                name, data_torch.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8
            )
    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.vocab_size = len(tokens)

        # we need this to validate the size of the token_type embeddings
        # though currently we are passing all zeros to the token_type embeddings
        # "Sequence A" or "Sequence B"
        self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))

        # convert to phantom space vocab
        def phantom(tok):
            if tok.startswith("[") and tok.endswith("]"):
                return tok
            if tok.startswith("##"):
                return tok[2:]
            return "\u2581" + tok
        tokens = list(map(phantom, tokens))

        # add vocab to gguf
        self.gguf_writer.add_tokenizer_model("bert")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # handle special tokens
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _xlmroberta_tokenizer_init(self) -> None:
        # we need the pad_token_id to know how to chop down position_embd matrix
        if (pad_token_id := self.hparams.get("pad_token_id")) is not None:
            self._position_offset = 1 + pad_token_id
            if "max_position_embeddings" in self.hparams:
                self.hparams["max_position_embeddings"] -= self._position_offset
        else:
            self._position_offset = None

    def _xlmroberta_set_vocab(self) -> None:
        # to avoid TypeError: Descriptors cannot be created directly
        # exception when importing sentencepiece_model_pb2
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from sentencepiece import SentencePieceProcessor
        from sentencepiece import sentencepiece_model_pb2 as model

        tokenizer_path = self.dir_model / 'sentencepiece.bpe.model'

        tokenizer_json = {}
        tokenizer_config_json = {}
        if not tokenizer_path.is_file():
            tokenizer_path = self.dir_model / 'tokenizer.json'
            tokenizer_config_path = self.dir_model / 'tokenizer_config.json'

            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"File not found: {tokenizer_path}")

            from base64 import b64decode
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model)

            with open(tokenizer_path, "r", encoding="utf-8") as fp:
                tokenizer_json = json.load(fp)

            if tokenizer_config_path.is_file():
                with open(tokenizer_config_path, "r", encoding="utf-8") as fp:
                    tokenizer_config_json = json.load(fp)

            add_prefix = tokenizer.add_prefix_space
            remove_whitespaces = tokenizer.clean_up_tokenization_spaces
            precompiled_charsmap = b64decode(tokenizer_json["normalizer"]["precompiled_charsmap"])

            vocab_size = max(self.hparams.get("vocab_size", 0), tokenizer.vocab_size)
        else:
            sentencepiece_model = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue]
            sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())
            assert sentencepiece_model.trainer_spec.model_type == 1  # UNIGRAM

            add_prefix = sentencepiece_model.normalizer_spec.add_dummy_prefix
            remove_whitespaces = sentencepiece_model.normalizer_spec.remove_extra_whitespaces
            precompiled_charsmap = sentencepiece_model.normalizer_spec.precompiled_charsmap

            tokenizer = SentencePieceProcessor()
            tokenizer.LoadFromFile(str(tokenizer_path))

            vocab_size = max(self.hparams.get("vocab_size", 0), tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        if isinstance(tokenizer, SentencePieceProcessor):
            for token_id in range(tokenizer.vocab_size()):
                piece = tokenizer.IdToPiece(token_id)
                text = piece.encode("utf-8")
                score = tokenizer.GetScore(token_id)

                toktype = SentencePieceTokenTypes.NORMAL
                if tokenizer.IsUnknown(token_id):
                    toktype = SentencePieceTokenTypes.UNKNOWN
                elif tokenizer.IsControl(token_id):
                    toktype = SentencePieceTokenTypes.CONTROL
                elif tokenizer.IsUnused(token_id):
                    toktype = SentencePieceTokenTypes.UNUSED
                elif tokenizer.IsByte(token_id):
                    toktype = SentencePieceTokenTypes.BYTE

                tokens[token_id] = text
                scores[token_id] = score
                toktypes[token_id] = toktype
        else:
            added_vocab = tokenizer.get_added_vocab()
            unk_token = tokenizer_config_json.get("unk_token")
            unk_token_id = added_vocab.get(unk_token, tokenizer_json["model"].get("unk_id", 3))

            for token_id in range(tokenizer.vocab_size):
                piece = tokenizer._convert_id_to_token(token_id)
                if (piece := tokenizer._convert_id_to_token(token_id)) is not None:
                    text = piece.encode("utf-8")
                    score = tokenizer_json["model"]["vocab"][token_id][1]

                    toktype = SentencePieceTokenTypes.NORMAL
                    if token_id == unk_token_id:
                        toktype = SentencePieceTokenTypes.UNKNOWN
                    elif token_id in tokenizer.all_special_ids:
                        toktype = SentencePieceTokenTypes.CONTROL
                    elif token_id in added_vocab.values():
                        toktype = SentencePieceTokenTypes.USER_DEFINED
                    # No reliable way to detect this, but jina doesn't have any
                    # elif tokenizer.IsByte(token_id):
                    #     toktype = SentencePieceTokenTypes.BYTE

                    tokens[token_id] = text
                    scores[token_id] = score
                    toktypes[token_id] = toktype

        if isinstance(tokenizer, SentencePieceProcessor):
            # realign tokens (see HF tokenizer code)
            tokens = [b'<s>', b'<pad>', b'</s>', b'<unk>'] + tokens[3:-1]
            scores = [0.0, 0.0, 0.0, 0.0] + scores[3:-1]
            toktypes = [
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.CONTROL,
                SentencePieceTokenTypes.UNKNOWN,
            ] + toktypes[3:-1]

            if self.model_arch == gguf.MODEL_ARCH.NOMIC_BERT_MOE:
                # Add mask token missing from sentencepiece.bpe.model
                tokens[250001] = b'<mask>'
                scores[250001] = 0.0
                toktypes[250001] = SentencePieceTokenTypes.CONTROL

        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(add_prefix)
        self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))
        self.gguf_writer.add_remove_extra_whitespaces(remove_whitespaces)
        if precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(precompiled_charsmap)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelBase.register("XLMRobertaModel", "XLMRobertaForSequenceClassification")
class XLMRobertaModel(BertModel):
    model_arch = gguf.MODEL_ARCH.BERT
    _lora_files = {}
    _lora_names = []

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, **kwargs: Any):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model)

        if lora_names := hparams.get("lora_adaptations"):
            self._lora_names = lora_names
            self.model_arch = gguf.MODEL_ARCH.JINA_BERT_V3

        super().__init__(dir_model, ftype, fname_out, hparams=hparams, **kwargs)
        self._xlmroberta_tokenizer_init()

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._lora_names:
            for name in self._lora_names:
                fname = self.add_prefix_to_filename(self.fname_out, f"lora-{name}-")
                self._lora_files[name] = gguf.GGUFWriter(fname, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file, dry_run=self.dry_run)

        return super().generate_extra_tensors()

    def set_type(self):
        for lora_writer in self._lora_files.values():
            lora_writer.add_type(gguf.GGUFType.ADAPTER)
            lora_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
        super().set_type()

    def set_vocab(self):
        self._xlmroberta_set_vocab()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # if name starts with "roberta.", remove the prefix
        # e.g. https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main
        if name.startswith("roberta."):
            name = name[8:]

        # jina-embeddings-v3
        if ".parametrizations." in name:
            name = name.replace(".parametrizations.", ".")
            if name.endswith(".original"):
                name = name[:-9]

        # position embeddings start at pad_token_id + 1, so just chop down the weight tensor
        if name == "embeddings.position_embeddings.weight":
            if self._position_offset is not None:
                data_torch = data_torch[self._position_offset:,:]

        if name.endswith(".0.lora_A") or name.endswith(".0.lora_B"):
            if name.startswith("pooler.dense"):
                return []

            num_loras = data_torch.size(0)
            assert num_loras == len(self._lora_names)

            # Split out each LoRA in their own GGUF
            for i, lora_writer in enumerate(self._lora_files.values()):
                new_name = self.map_tensor_name(name[:-9]) + name[-7:].lower()
                data = data_torch[i, :, :]
                # Transpose/flip token_embd/types into correct shape
                if new_name == "token_embd.weight.lora_b":
                    data = data.T
                elif new_name.startswith("token_types.weight."):
                    new_name = new_name[:-1] + ("a" if new_name[-1:] == "b" else "b")
                lora_writer.add_tensor(new_name, data.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)

            return []

        return super().modify_tensors(data_torch, name, bid)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # jina-embeddings-v3
        if rotary_emb_base := self.hparams.get("rotary_emb_base"):
            self.gguf_writer.add_rope_freq_base(rotary_emb_base)
        lora_alpha = self.hparams.get("lora_alpha")
        if lora_prompt_prefixes := self.hparams.get("task_instructions"):
            assert self._lora_files and all(lora_name in lora_prompt_prefixes for lora_name in self._lora_files.keys())
        for lora_name, lora_writer in self._lora_files.items():
            lora_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha if lora_alpha is not None else 1.0)
            lora_writer.add_string(gguf.Keys.Adapter.LORA_TASK_NAME, lora_name)
            if lora_prompt_prefixes:
                lora_writer.add_string(gguf.Keys.Adapter.LORA_PROMPT_PREFIX, lora_prompt_prefixes[lora_name])

    def write(self):
        super().write()
        for lora_writer in self._lora_files.values():
            lora_writer.write_header_to_file()
            lora_writer.write_kv_data_to_file()
            lora_writer.write_tensors_to_file(progress=True)
            lora_writer.close()



###### CONVERSION LOGIC ######


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy()),
        )

    @classmethod
    def meta_with_dtype_and_shape(
        cls, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(
            meta=cls.meta_with_dtype_and_shape(dtype, shape),
            args=(st_slice,),
            func=lambda s: s[:],
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(
            meta=meta,
            args=(remote_tensor,),
            func=lambda r: torch.frombuffer(r.data(), dtype=dtype).reshape(shape),
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file"
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "q4_0", "auto"],
        default="q4_0",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian",
        action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model",
        type=str,
        help="directory containing model file or huggingface repository ID (if --remote)",
        nargs="?",
    )
    parser.add_argument(
        "--use-temp-file",
        action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy",
        action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors",
        type=int,
        default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size",
        type=str,
        default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split",
        action="store_true",
        help="do not add tensors to the first split (disabled by default)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Specify the path for an authorship metadata override file",
    )
    parser.add_argument(
        "--hmm-info",
        type=str,
        default="dadao",
        help="Specify Description of the HMM model to be used for conversion",
    )
    parser.add_argument(
        "--hmm-vit-width",
        type=int,
        default=644,
        help="Specify the visual width of the HMM model to be used for qwenvl",
    )
    parser.add_argument(
        "--hmm-vit-height",
        type=int,
        default=364,
        help="Specify the visual height of the HMM model to be used for qwenvl",
    )
    parser.add_argument(
        "--print-supported-models",
        action="store_true",
        help="Print the supported models",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj",
        action="store_true",
        help="(Experimental) Export multimodal projector (mmproj) for vision models. This will only work on some vision models. A prefix 'mmproj-' will be added to the output file name.",
    )

    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(
            f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G"
        )

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def get_model_architecture(hparams: dict[str, Any], model_type: ModelType) -> str:
    # TODO @ngxson : this won't work correctly if the model has both audio & vision encoders
    # maybe we should fallback to text model's arch in that case, since not many models have both
    text_config = hparams.get("text_config", {})
    vision_config = hparams.get("vision_config", {})
    arch = None
    if (arches := hparams.get("architectures")) is not None and len(arches) > 0:
        arch = arches[0]
    elif "ssm_cfg" in hparams:
        # For non-hf Mamba and Mamba2 models
        arch = hparams["ssm_cfg"].get("layer", "Mamba") + "ForCausalLM"

    # if "architectures" is found in the sub-config, use that instead
    if model_type == ModelType.TEXT and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    elif (
        model_type == ModelType.MMPROJ
        and vision_config.get("architectures") is not None
    ):
        arch = vision_config["architectures"][0]
    if arch is None:
        raise ValueError("Failed to detect model architecture")
    return arch


def main() -> None:
    args = parse_args()

    if args.print_supported_models:
        logger.error("Supported models:")
        ModelBase.print_registered_models()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"],
        )
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)

    if not dir_model.is_dir():
        logger.error(f"Error: {dir_model} is not a directory")
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    if args.mmproj:
        if "mmproj" not in fname_out.name:
            fname_out = ModelBase.add_prefix_to_filename(fname_out, "mmproj-")

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        hparams = ModelBase.load_hparams(dir_model)
        model_architecture = get_model_architecture(hparams, model_type)
        logger.info(f"Model architecture: {model_architecture}")
        try:
            model_class = ModelBase.from_model_architecture(
                model_architecture, model_type=model_type
            )
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)

        model_instance = model_class(
            dir_model,
            output_type,
            fname_out,
            is_big_endian=args.bigendian,
            use_temp_file=args.use_temp_file,
            eager=args.no_lazy,
            metadata_override=args.metadata,
            model_name=args.model_name,
            split_max_tensors=args.split_max_tensors,
            split_max_size=split_str_to_n_bytes(args.split_max_size),
            dry_run=args.dry_run,
            small_first_shard=args.no_tensor_first_split,
            remote_hf_model_id=hf_repo_id,
            model_description=args.hmm_info,
            hmm_vit_height=args.hmm_vit_height,
            hmm_vit_width=args.hmm_vit_width,
        )

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(
                f"Model vocab successfully exported to {model_instance.fname_out}"
            )
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = (
                f"{model_instance.fname_out.parent}{os.sep}"
                if is_split
                else model_instance.fname_out
            )
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == "__main__":
    main()
