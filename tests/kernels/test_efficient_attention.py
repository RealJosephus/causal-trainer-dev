from __future__ import annotations

import ast
import functools
import importlib
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from causal_trainer.kernels.efficient_attention import (
    AttentionKernelProfile,
    AttentionMetadata,
    attention_partition_specs,
    default_attention_scale,
    efficient_attention,
    prepare_attention_metadata,
    resolve_attention_profile,
    validate_attention_profile,
)
from causal_trainer.modeling.architecture import decoder_forward, forward
from causal_trainer.modeling.attention import (
    _attention_precanonicalized,
    attention,
    make_causal_segment_mask,
    vanilla_attention,
)

efficient_module = importlib.import_module("causal_trainer.kernels.efficient_attention")


def _mesh_shape(*, dp: int = 1, fsdp: int = 1, tp: int = 1, sp: int = 1) -> dict[str, int]:
    return {"dp": dp, "fsdp": fsdp, "ep": 1, "tp": tp, "sp": sp}


def _single_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1, 1, 1, 1)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def test_efficient_is_the_model_api_default() -> None:
    for function in (
        _attention_precanonicalized,
        attention,
        decoder_forward,
        forward,
    ):
        assert inspect.signature(function).parameters["implementation"].default == "efficient"


def test_compact_statistics_is_explicitly_opt_in() -> None:
    q_shape, kv_shape = (1, 8192, 4, 128), (1, 8192, 1, 128)
    mesh = _mesh_shape(tp=2, sp=2)
    ordinary = resolve_attention_profile(q_shape, kv_shape, mesh)
    compact = resolve_attention_profile(q_shape, kv_shape, mesh, compact_statistics=True)

    assert ordinary.compact_statistics is False
    assert compact.compact_statistics is True
    assert (compact.block_h, compact.block_q, compact.block_kv_outer, compact.block_kv_compute) == (
        ordinary.block_h, ordinary.block_q, ordinary.block_kv_outer, ordinary.block_kv_compute
    )
    with pytest.raises(TypeError, match="compact_statistics must be a static bool"):
        resolve_attention_profile(q_shape, kv_shape, mesh, compact_statistics=1)


@pytest.mark.parametrize("heads", [1, 2, 8])
def test_compact_statistics_preserves_token_head_order_when_loading_tiles(heads: int) -> None:
    stats = jnp.arange(2 * 512 * heads, dtype=jnp.float32).reshape(2, 512, heads)
    compact = efficient_module._statistics_kernel_input(stats, compact=True)
    expanded = efficient_module._statistics_kernel_input(stats, compact=False)
    start, stop = 128 * heads, 256 * heads

    assert compact.shape == (2 * 512 * heads,)
    assert compact.size * 128 == expanded.size
    compact_tile = efficient_module._load_row_stat(compact[512 * heads + start:512 * heads + stop], 128 * heads)
    ordinary_tile = efficient_module._load_row_stat(expanded[1:2, 128:256], 128 * heads)
    np.testing.assert_array_equal(compact_tile, ordinary_tile)
    np.testing.assert_array_equal(jnp.max(compact_tile, axis=-1), compact[512 * heads + start:512 * heads + stop])


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("sequence", [4096, 8192])
def test_statistics_hbm_shapes_in_forward_and_both_backward_kernels(
    monkeypatch: pytest.MonkeyPatch,
    compact: bool,
    sequence: int,
) -> None:
    """Trace real wrappers at training shapes without allocating their buffers.

    Intercept only the hardware call: this catches accidental full-sequence
    lane expansion on any of the three Pallas boundaries on CPU CI as well.
    """

    calls = []

    def capture_pallas_call(_kernel, **kwargs):
        def call(*args):
            calls.append((kwargs, tuple(value.shape for value in args)))
            return jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), kwargs["out_shape"])

        return call

    monkeypatch.setattr(efficient_module.pl, "pallas_call", capture_pallas_call)
    monkeypatch.setattr(
        efficient_module,
        "_output_shape",
        lambda shape, dtype, _like: jax.ShapeDtypeStruct(shape, dtype),
    )
    q_shape, kv_shape = (1, sequence, 4, 128), (1, sequence, 1, 128)
    profile = resolve_attention_profile(
        q_shape, kv_shape, _mesh_shape(tp=2, sp=2), compact_statistics=compact
    )
    local_shape = (1, sequence // 2, 2, 128)

    def trace(q, k, v):
        q_segments = jnp.ones(q.shape[:2], jnp.int32)
        kv_segments = jnp.ones(k.shape[:2], jnp.int32)
        q_summary = jnp.ones((1, q.shape[1] // profile.block_q, 2), jnp.int32)
        kv_summary = jnp.ones((1, sequence // profile.block_kv_outer, 2), jnp.int32)
        common = (q, k, v, q_segments, kv_segments, q_summary, kv_summary)
        offsets = jnp.asarray([sequence // 2, 0], jnp.int32)
        options = {"profile": profile, "scale": 128**-0.5, "causal": True}
        output, lse = efficient_module._local_attention_forward(*common, offsets, **options)
        delta = jnp.zeros(q.shape[:3], jnp.float32)
        backward = (*common, lse, output, delta, offsets)
        dq = efficient_module._local_attention_dq(*backward, **options)
        dk, dv = efficient_module._local_attention_dkv(*backward, **options)
        return output, lse, dq, dk, dv

    shapes = jax.eval_shape(
        trace,
        jax.ShapeDtypeStruct(local_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
    )
    assert [shape.shape for shape in shapes] == [local_shape, local_shape[:3], local_shape, kv_shape, kv_shape]
    assert len(calls) == 3
    expected = (sequence,) if compact else (*local_shape[:3], 128)
    assert calls[0][0]["out_shape"][1].shape == expected
    for options, input_shapes in calls[1:]:
        assert input_shapes[8] == input_shapes[10] == expected
        spec = options["grid_spec"].in_specs[5]
        expected_block = (profile.block_q * profile.block_h,) if compact else (
            1, profile.block_q, profile.block_h, 128
        )
        assert spec.block_shape == expected_block


@pytest.mark.parametrize(("tp", "sp", "batch"), [(1, 1, 2), (2, 2, 1)])
def test_compact_pallas_interpreter_packed_forward_and_vjp(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    sp: int,
    batch: int,
) -> None:
    """Exercise actual tile arithmetic; TPU lowering has a separate hardware gate."""

    if jax.device_count() < tp * sp:
        pytest.skip("set XLA_FLAGS=--xla_force_host_platform_device_count=4 for the TP/SP interpreter gate")
    monkeypatch.setattr(
        efficient_module.pl, "pallas_call", functools.partial(efficient_module.pl.pallas_call, interpret=True)
    )
    # JAX 0.10's HLO interpreter discharges scalar-prefetch refs using invariant
    # slice indices, which fails VMA checking on varying SP summaries. Hardware
    # gates below keep the production check_vma=True path unchanged.
    monkeypatch.setattr(
        efficient_module,
        "shard_map",
        lambda fn, **kwargs: jax.shard_map(fn, **(kwargs | {"check_vma": False})),
    )
    devices = np.asarray(jax.devices()[:tp * sp], dtype=object).reshape(1, 1, 1, tp, sp)
    mesh = Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))
    sequence, heads = 256, 2 * tp
    rng = np.random.default_rng(931)
    query = jnp.asarray(rng.normal(scale=0.05, size=(batch, sequence, heads, 128)), jnp.bfloat16)
    key = jnp.asarray(rng.normal(scale=0.05, size=(batch, sequence, 1, 128)), jnp.bfloat16)
    value = jnp.asarray(rng.normal(scale=0.05, size=key.shape), jnp.bfloat16)
    positions = jnp.broadcast_to(jnp.arange(sequence)[None, :], (batch, sequence))
    segments = jnp.where(positions < 17, 0, jnp.where(positions < 71, 1, jnp.where(positions < 211, 2, 3)))
    cotangent = jnp.asarray(rng.normal(scale=0.02, size=query.shape), jnp.bfloat16)
    # Only the right side of a record spanning the SP boundary receives dO.
    cotangent = jnp.where(((positions >= 160) & (positions < 211))[:, :, None, None], cotangent, 0)
    scale = 128**-0.5

    def value_and_vjp(fn):
        output, backward = jax.vjp(fn, query, key, value)
        return (output, *backward(cotangent))

    allowed = make_causal_segment_mask(segments != 0, segments)
    reference = value_and_vjp(lambda q, k, v: vanilla_attention(q, k, v, allowed, scale=scale))
    results = []
    with jax.set_mesh(mesh):
        for compact in (False, True):
            profile = AttentionKernelProfile(2, 128, 128, 128, compact_statistics=compact)
            metadata = prepare_attention_metadata(segments, mesh=mesh, profile=profile)
            results.append(value_and_vjp(functools.partial(
                efficient_module._efficient_attention_impl,
                metadata=metadata, mesh=mesh, profile=profile, scale=scale, causal=True,
            )))

    for compact_result, ordinary_result, expected in zip(results[1], results[0], reference, strict=True):
        actual = np.asarray(compact_result, dtype=np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        np.testing.assert_array_equal(actual, np.asarray(ordinary_result, dtype=np.float32))
        relative_l2 = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
        assert relative_l2 < 0.03
        assert not np.any(actual[np.asarray(segments == 0)])
    for gradient in results[1][2:]:
        assert np.any(np.asarray(gradient, dtype=np.float32)[0, 71:128] != 0)
        assert not np.any(np.asarray(gradient, dtype=np.float32)[np.asarray(segments != 2)])


@pytest.mark.parametrize(
    ("tp", "expected_block_h", "expected_block_q"),
    [
        (4, 8, 256),
        (8, 4, 512),
        (16, 2, 1024),
    ],
)
def test_profiles_derive_query_tiles_from_local_head_count(
    tp: int,
    expected_block_h: int,
    expected_block_q: int,
) -> None:
    profile = resolve_attention_profile(
        (1, 4096, 32, 128),
        (1, 4096, 1, 128),
        _mesh_shape(tp=tp),
    )

    assert profile.block_h == expected_block_h
    assert profile.block_q == expected_block_q
    assert profile.block_h * profile.block_q == 2048
    assert profile.block_kv_outer == 512
    assert profile.block_kv_compute == 256


def test_synthetic_head_count_uses_the_same_profile_formula() -> None:
    profile = resolve_attention_profile(
        (2, 2048, 16, 128),
        (2, 2048, 1, 128),
        _mesh_shape(tp=4),
    )

    assert profile == AttentionKernelProfile(
        block_h=4,
        block_q=512,
        block_kv_outer=512,
        block_kv_compute=256,
    )


def test_profile_resolver_uses_aligned_divisors_below_preferred_tiles() -> None:
    profile = resolve_attention_profile(
        (1, 384, 32, 128),
        (1, 384, 1, 128),
        _mesh_shape(tp=4),
    )

    assert profile == AttentionKernelProfile(
        block_h=8,
        block_q=128,
        block_kv_outer=384,
        block_kv_compute=128,
    )


def test_profile_resolver_rejects_sp_local_length_without_an_aligned_divisor() -> None:
    with pytest.raises(ValueError, match="local_query=192"):
        resolve_attention_profile(
            (1, 384, 32, 128),
            (1, 384, 1, 128),
            _mesh_shape(tp=4, sp=2),
        )


@pytest.mark.parametrize(
    ("sequence", "expected_outer", "expected_compute"),
    [
        (16_384, 512, 256),
        (32_768, 1024, 256),
        (262_144, 1024, 256),
        (524_288, 2048, 512),
    ],
)
def test_kv_profile_tiers_follow_global_sequence_length(
    sequence: int,
    expected_outer: int,
    expected_compute: int,
) -> None:
    profile = resolve_attention_profile(
        (1, sequence, 32, 128),
        (1, sequence, 1, 128),
        _mesh_shape(tp=4),
    )

    assert profile.block_kv_outer == expected_outer
    assert profile.block_kv_compute == expected_compute
    assert sequence % profile.block_kv_outer == 0
    assert profile.block_kv_outer % profile.block_kv_compute == 0


def test_profile_resolver_uses_the_sp_local_query_length() -> None:
    profile = resolve_attention_profile(
        (1, 1024, 32, 128),
        (1, 1024, 1, 128),
        _mesh_shape(tp=4, sp=8),
    )

    assert profile.block_h == 8
    assert profile.block_q == 128


@pytest.mark.parametrize(
    ("query_shape", "key_value_shape", "mesh_shape"),
    [
        ((1, 4096, 32), (1, 4096, 1, 128), _mesh_shape(tp=4)),
        ((1, 4096, 32, 128), (2, 4096, 1, 128), _mesh_shape(tp=4)),
        ((1, 4096, 32, 128), (1, 2048, 1, 128), _mesh_shape(tp=4)),
        ((1, 4096, 32, 128), (1, 4096, 1, 64), _mesh_shape(tp=4)),
        ((1, 4096, 30, 128), (1, 4096, 4, 128), _mesh_shape(tp=2)),
        ((1, 4096, 32, 128), (1, 4096, 1, 128), _mesh_shape(tp=3)),
        ((1, 4096, 32, 128), (1, 4096, 1, 128), _mesh_shape(tp=4, sp=3)),
    ],
)
def test_profile_resolver_rejects_invalid_shapes_and_meshes(
    query_shape: tuple[int, ...],
    key_value_shape: tuple[int, ...],
    mesh_shape: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        resolve_attention_profile(query_shape, key_value_shape, mesh_shape)


@pytest.mark.parametrize(
    "profile",
    [
        AttentionKernelProfile(block_h=3, block_q=128, block_kv_outer=256, block_kv_compute=128),
        AttentionKernelProfile(block_h=8, block_q=192, block_kv_outer=256, block_kv_compute=128),
        AttentionKernelProfile(block_h=8, block_q=128, block_kv_outer=384, block_kv_compute=128),
        AttentionKernelProfile(block_h=8, block_q=128, block_kv_outer=256, block_kv_compute=96),
        AttentionKernelProfile(block_h=8, block_q=128, block_kv_outer=128, block_kv_compute=256),
    ],
)
def test_profile_validation_rejects_illegal_tiles(profile: AttentionKernelProfile) -> None:
    with pytest.raises(ValueError):
        validate_attention_profile(
            profile,
            (1, 4096, 32, 128),
            (1, 4096, 1, 128),
            _mesh_shape(tp=4),
        )


def test_profile_validation_accepts_the_resolved_profile() -> None:
    query_shape = (1, 4096, 32, 128)
    key_value_shape = (1, 4096, 1, 128)
    mesh_shape = _mesh_shape(tp=4)
    profile = resolve_attention_profile(query_shape, key_value_shape, mesh_shape)

    validate_attention_profile(profile, query_shape, key_value_shape, mesh_shape)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_default_scale_is_derived_from_head_dimension(head_dim: int) -> None:
    actual = jnp.asarray(default_attention_scale(head_dim), dtype=jnp.bfloat16)
    expected = jnp.asarray(head_dim**-0.5, dtype=jnp.bfloat16)

    assert actual == expected


@pytest.mark.parametrize("head_dim", [0, -128])
def test_default_scale_rejects_non_positive_head_dimension(head_dim: int) -> None:
    with pytest.raises(ValueError):
        default_attention_scale(head_dim)


def test_public_kernel_rejects_unsupported_head_dimension_before_tpu_lowering() -> None:
    mesh = _single_device_mesh()
    query = jnp.zeros((1, 256, 32, 64), dtype=jnp.bfloat16)
    key = jnp.zeros((1, 256, 1, 64), dtype=jnp.bfloat16)
    profile = resolve_attention_profile(query.shape, key.shape, mesh)
    metadata = prepare_attention_metadata(
        jnp.ones((1, 256), dtype=jnp.int32),
        mesh=mesh,
        profile=profile,
    )

    with pytest.raises(ValueError, match="requires head_dim=128"):
        efficient_attention(
            query,
            key,
            key,
            metadata,
            mesh=mesh,
            profile=profile,
        )


def test_public_kernel_rejects_noncausal_mode_before_tpu_lowering() -> None:
    mesh = _single_device_mesh()
    query = jnp.zeros((1, 256, 32, 128), dtype=jnp.bfloat16)
    key = jnp.zeros((1, 256, 1, 128), dtype=jnp.bfloat16)
    profile = resolve_attention_profile(query.shape, key.shape, mesh)
    metadata = prepare_attention_metadata(
        jnp.ones((1, 256), dtype=jnp.int32),
        mesh=mesh,
        profile=profile,
    )

    with pytest.raises(ValueError, match="implements causal attention only"):
        efficient_attention(
            query,
            key,
            key,
            metadata,
            mesh=mesh,
            profile=profile,
            causal=False,
        )


def test_metadata_summarizes_left_padding_and_unaligned_segment_boundaries() -> None:
    # The first segment begins after a complete padding block. Segment changes
    # at 179 and 301 are unaligned with both metadata tiles.
    segment_ids = jnp.concatenate(
        (
            jnp.zeros(130, dtype=jnp.int32),
            jnp.ones(49, dtype=jnp.int32),
            jnp.full(122, 2, dtype=jnp.int32),
            jnp.full(211, 3, dtype=jnp.int32),
        )
    )[None, :]
    profile = AttentionKernelProfile(
        block_h=1,
        block_q=128,
        block_kv_outer=256,
        block_kv_compute=128,
    )

    metadata = prepare_attention_metadata(
        segment_ids,
        mesh=_single_device_mesh(),
        profile=profile,
    )

    assert isinstance(metadata, AttentionMetadata)
    np.testing.assert_array_equal(np.asarray(metadata.query_segment_ids), np.asarray(segment_ids))
    np.testing.assert_array_equal(np.asarray(metadata.kv_segment_ids), np.asarray(segment_ids))
    np.testing.assert_array_equal(
        np.asarray(metadata.q_block_segments),
        np.asarray([[[0, 0], [1, 2], [2, 3], [3, 3]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.kv_block_segments),
        np.asarray([[[1, 2], [2, 3]]], dtype=np.int32),
    )


def test_metadata_uses_zero_summary_for_fully_padded_blocks() -> None:
    segment_ids = jnp.zeros((2, 512), dtype=jnp.int32)
    profile = AttentionKernelProfile(
        block_h=1,
        block_q=128,
        block_kv_outer=256,
        block_kv_compute=128,
    )

    metadata = prepare_attention_metadata(
        segment_ids,
        mesh=_single_device_mesh(),
        profile=profile,
    )

    assert metadata.q_block_segments.shape == (2, 4, 2)
    assert metadata.kv_block_segments.shape == (2, 2, 2)
    assert metadata.q_block_segments.dtype == jnp.int32
    assert metadata.kv_block_segments.dtype == jnp.int32
    assert not bool(jnp.any(metadata.q_block_segments))
    assert not bool(jnp.any(metadata.kv_block_segments))


def test_metadata_rejects_invalid_rank_and_profile_divisibility() -> None:
    mesh = _single_device_mesh()
    profile = AttentionKernelProfile(
        block_h=1,
        block_q=128,
        block_kv_outer=256,
        block_kv_compute=128,
    )
    with pytest.raises(ValueError):
        prepare_attention_metadata(jnp.zeros((512,), dtype=jnp.int32), mesh=mesh, profile=profile)

    with pytest.raises(ValueError):
        prepare_attention_metadata(
            jnp.zeros((1, 384), dtype=jnp.int32),
            mesh=mesh,
            profile=profile,
        )


def test_partition_specs_keep_query_sharded_and_kv_replicated() -> None:
    class FakeMesh:
        axis_names = ("dp", "fsdp", "ep", "tp", "sp")
        shape: ClassVar = {"dp": 2, "fsdp": 2, "ep": 1, "tp": 4, "sp": 2}

    specs = attention_partition_specs(FakeMesh())
    values = specs.values() if isinstance(specs, Mapping) else specs

    expected = {
        PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        PartitionSpec(("dp", "fsdp"), "sp", "tp"),
        PartitionSpec(("dp", "fsdp"), None, None, None),
        PartitionSpec(("dp", "fsdp"), "sp"),
        PartitionSpec(("dp", "fsdp"), None),
        PartitionSpec(("dp", "fsdp"), "sp", None),
        PartitionSpec(("dp", "fsdp"), None, None),
    }
    assert set(values) == expected


def test_efficient_kernel_has_no_splash_import_dependency() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "causal_trainer"
        / "kernels"
        / "efficient_attention.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any("splash" in module_name.lower() for module_name in imported_modules)


def test_decoder_prepares_attention_metadata_once_outside_the_layer_body() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "causal_trainer"
        / "modeling"
        / "architecture.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def preparation_calls(function_name: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_attention_metadata"
            for node in ast.walk(functions[function_name])
        )

    assert preparation_calls("decoder_forward") == 1
    assert preparation_calls("_decoder_layer") == 0


def test_efficient_backward_owns_only_the_tp_sp_dkv_reductions() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "causal_trainer"
        / "kernels"
        / "efficient_attention.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    backward = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_distributed_backward"
    )
    psum_calls = [
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "lax"
        and node.func.attr == "psum"
    ]
    reduction_axis_literals = {
        tuple(element.value for element in node.elts)
        for node in ast.walk(backward)
        if isinstance(node, ast.Tuple)
        and node.elts
        and all(isinstance(element, ast.Constant) for element in node.elts)
        and all(isinstance(element.value, str) for element in node.elts)
    }

    assert len(psum_calls) == 2  # One each for dK and dV; no outer transpose reduction.
    assert ("tp", "sp") in reduction_axis_literals
    assert not any("dp" in axes for axes in reduction_axis_literals)
