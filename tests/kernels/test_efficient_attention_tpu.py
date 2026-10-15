"""TPU correctness gates for the efficient packed-attention kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from causal_trainer.kernels.efficient_attention import (
    efficient_attention,
    prepare_attention_metadata,
    resolve_attention_profile,
)
from causal_trainer.modeling.attention import (
    make_causal_segment_mask,
    splash_attention,
    vanilla_attention,
)


def _has_four_local_tpu_devices() -> bool:
    try:
        return jax.default_backend() == "tpu" and jax.local_device_count() >= 4
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_four_local_tpu_devices(),
    reason="requires at least four local TPU devices with JAX 0.10",
)

_AXIS_NAMES = ("dp", "fsdp", "ep", "tp", "sp")
_HEAD_DIM = 128


@dataclass(frozen=True)
class _Tolerance:
    rtol: float
    atol: float
    relative_l2: float


# Keep the established Splash-versus-vanilla acceptance bounds.  Both TPU
# kernels use blockwise reductions, while the small reference reduces the full
# key dimension in one XLA program.  dK/dV additionally sum TP (and for the
# second test SP) partials.
_TOLERANCES = {
    "out": _Tolerance(rtol=0.08, atol=0.04, relative_l2=0.04),
    "dq": _Tolerance(rtol=0.15, atol=0.08, relative_l2=0.10),
    "dk": _Tolerance(rtol=0.20, atol=0.12, relative_l2=0.12),
    "dv": _Tolerance(rtol=0.15, atol=0.08, relative_l2=0.10),
}


def _local_mesh(*, tp: int, sp: int) -> Mesh:
    if tp * sp != 4:
        raise ValueError(f"test mesh must use exactly four local devices, got tp={tp}, sp={sp}")
    # Use four process-local devices for the rank-local TP4 gate.
    devices = np.asarray(jax.local_devices()[:4], dtype=object).reshape(1, 1, 1, tp, sp)
    return Mesh(devices, _AXIS_NAMES)


def _bf16_normal(seed: int, shape: tuple[int, ...], *, scale: float = 0.05) -> jax.Array:
    return (jax.random.normal(jax.random.PRNGKey(seed), shape) * scale).astype(jnp.bfloat16)


def _value_and_vjp(
    forward: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    def run(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        cotangent: jax.Array,
    ) -> tuple[jax.Array, ...]:
        output, pullback = jax.vjp(forward, query, key, value)
        dquery, dkey, dvalue = pullback(cotangent)
        return output, dquery, dkey, dvalue

    return run


def _host_float32(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value)).astype(np.float32)


def _assert_close(
    name: str,
    actual: Any,
    reference: Any,
    *,
    actual_name: str,
    reference_name: str,
) -> None:
    actual_host = _host_float32(actual)
    reference_host = _host_float32(reference)
    assert actual_host.shape == reference_host.shape
    assert np.isfinite(actual_host).all(), f"{actual_name} {name} contains non-finite values"
    assert np.isfinite(reference_host).all(), f"{reference_name} {name} contains non-finite values"

    difference = actual_host - reference_host
    max_abs = float(np.abs(difference).max(initial=0.0))
    reference_norm = float(np.linalg.norm(reference_host.ravel()))
    relative_l2 = float(np.linalg.norm(difference.ravel()) / max(reference_norm, 1e-12))
    tolerance = _TOLERANCES[name]
    elementwise_close = np.allclose(
        actual_host,
        reference_host,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
    )
    assert elementwise_close and relative_l2 <= tolerance.relative_l2, (
        f"{actual_name} {name} differs from {reference_name}: "
        f"max_abs={max_abs:.8g}, relative_l2={relative_l2:.8g}, tolerance={tolerance}"
    )


def _assert_result_matches(
    actual: tuple[jax.Array, ...],
    reference: tuple[jax.Array, ...],
    *,
    actual_name: str,
    reference_name: str = "vanilla",
) -> None:
    for name, actual_value, reference_value in zip(
        ("out", "dq", "dk", "dv"),
        actual,
        reference,
        strict=True,
    ):
        _assert_close(
            name,
            actual_value,
            reference_value,
            actual_name=actual_name,
            reference_name=reference_name,
        )


def _assert_exact_zero(name: str, value: Any, positions: np.ndarray) -> None:
    host = _host_float32(value)
    selected = host[positions]
    max_abs = float(np.abs(selected).max(initial=0.0))
    assert max_abs == 0.0, f"{name} must be exactly zero, max_abs={max_abs}"


def _distributed_inputs(
    mesh: Mesh,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    sequence_axis = "sp" if int(mesh.shape["sp"]) > 1 else None
    tensor_axis = "tp" if int(mesh.shape["tp"]) > 1 else None
    query_sharding = NamedSharding(mesh, PartitionSpec(None, sequence_axis, tensor_axis, None))
    key_value_sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))
    return (
        jax.device_put(query, query_sharding),
        jax.device_put(key, key_value_sharding),
        jax.device_put(value, key_value_sharding),
        jax.device_put(cotangent, query_sharding),
    )


def _packed_segments(sequence: int, segment_count: int) -> jax.Array:
    padding = 128
    valid_tokens = sequence - padding
    if segment_count == 8:
        lengths = (37, 61, 43, 55, 29, 71, 41, 47)
    elif segment_count == 32:
        lengths = (12,) * 32
    else:
        raise ValueError(f"unsupported packed test segment count: {segment_count}")
    if sum(lengths) != valid_tokens:
        raise AssertionError(
            f"packed test lengths sum to {sum(lengths)}, expected {valid_tokens}"
        )
    row = np.concatenate(
        (
            np.zeros(padding, dtype=np.int32),
            *(np.full(length, index, dtype=np.int32) for index, length in enumerate(lengths, 1)),
        )
    )
    return jnp.asarray(row[None, :], dtype=jnp.int32)


@pytest.mark.parametrize(
    ("sequence", "tp", "sp", "compact_statistics"),
    [(524288, 1, 4, False), (524288, 2, 2, False),
     (1048576, 1, 4, True), (1048576, 2, 2, True)],
)
def test_real_long_sequence_forward_and_vjp_compile(
    sequence: int,
    tp: int,
    sp: int,
    compact_statistics: bool,
) -> None:
    # H4/TP1/SP4 and H4/TP2/SP2 reproduce the per-chip H32/TP8/SP4
    # and H32/TP16/SP2 working sets on four local devices. Keep the actual
    # sequence length: short inputs miss SMEM padding and output buffering.
    # Compile with abstract inputs only; numerical gates use small sequences.
    mesh = _local_mesh(tp=tp, sp=sp)
    query_shape = (1, sequence, 4, _HEAD_DIM)
    kv_shape = (1, sequence, 1, _HEAD_DIM)
    query_sharding = NamedSharding(mesh, PartitionSpec(None, "sp", "tp", None))
    kv_sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))
    segment_sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))
    profile = resolve_attention_profile(
        query_shape, kv_shape, mesh, compact_statistics=compact_statistics,
    )

    def run(q, k, v, cotangent, segments):
        metadata = prepare_attention_metadata(segments, mesh=mesh, profile=profile)
        return _value_and_vjp(
            lambda q, k, v: efficient_attention(q, k, v, metadata, mesh=mesh, profile=profile)
        )(q, k, v, cotangent)

    query = jax.ShapeDtypeStruct(query_shape, jnp.bfloat16, sharding=query_sharding)
    kv = jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16, sharding=kv_sharding)
    segments = jax.ShapeDtypeStruct((1, sequence), jnp.int32, sharding=segment_sharding)
    with jax.set_mesh(mesh):
        compiled = jax.jit(run).lower(query, kv, kv, query, segments).compile()
    assert compiled.memory_analysis() is not None


@pytest.mark.parametrize(
    ("query_heads", "target_sequence", "compact_statistics"),
    [(4, 2048, False), (4, 262144, False), (4, 1048576, False), (4, 1048576, True),
     (8, 524288, False), (8, 1048576, True)],
)
def test_common_and_long_profiles_fit_v4_and_match_vanilla(
    query_heads: int,
    target_sequence: int,
    compact_statistics: bool,
) -> None:
    # Local TP2 with H4/H8 has the target H32/TP16 or H32/TP8 tile layout.
    # Keep multiple KV output windows and an SP-crossing segment without a
    # quadratic 1M reference. Resolve tiles from the real target length.
    mesh = _local_mesh(tp=2, sp=2)
    sequence = 4096 if target_sequence > 262144 else 2048
    query = _bf16_normal(700, (1, sequence, query_heads, _HEAD_DIM))
    key = _bf16_normal(701, (1, sequence, 1, _HEAD_DIM))
    value = _bf16_normal(702, key.shape)
    cotangent = _bf16_normal(703, query.shape, scale=0.02)
    segments = jnp.asarray(
        np.concatenate((np.zeros(97), np.ones(1101), np.full(sequence - 1198, 2)))[None], jnp.int32,
    )
    profile = resolve_attention_profile(
        (1, target_sequence, query_heads, _HEAD_DIM),
        (1, target_sequence, 1, _HEAD_DIM),
        mesh,
        compact_statistics=compact_statistics,
    )
    assert profile.block_q == (256 if target_sequence > 262144 else 512)
    allowed = make_causal_segment_mask(segments != 0, segments)
    reference = jax.jit(_value_and_vjp(
        lambda q, k, v: vanilla_attention(q, k, v, allowed, scale=_HEAD_DIM**-0.5)
    ))(query, key, value, cotangent)
    with jax.set_mesh(mesh):
        metadata = prepare_attention_metadata(segments, mesh=mesh, profile=profile)
        actual = jax.jit(_value_and_vjp(
            lambda q, k, v: efficient_attention(q, k, v, metadata, mesh=mesh, profile=profile)
        ))(*_distributed_inputs(mesh, query, key, value, cotangent))
    _assert_result_matches(actual, reference, actual_name="efficient")
    for name, result in zip(("out", "dq", "dk", "dv"), actual, strict=True):
        _assert_exact_zero(name, result, np.asarray(segments == 0))


@pytest.mark.parametrize("segment_count", [8, 32])
@pytest.mark.parametrize("compact_statistics", [False, True])
def test_rank0_local_tp4_packed_efficient_attention_forward_and_vjp_match_references(
    segment_count: int,
    compact_statistics: bool,
) -> None:
    batch, sequence, query_heads = 1, 512, 32
    query = _bf16_normal(100, (batch, sequence, query_heads, _HEAD_DIM))
    key = _bf16_normal(101, (batch, sequence, 1, _HEAD_DIM))
    value = _bf16_normal(102, (batch, sequence, 1, _HEAD_DIM))
    cotangent = _bf16_normal(103, query.shape, scale=0.02)

    # The first half of the 256-query block is left padding, and packed
    # boundaries are unaligned with the 256 Q and 512 K/V tiles.
    segments = _packed_segments(sequence, segment_count)
    valid = segments != 0
    allowed = make_causal_segment_mask(valid, segments)
    scale = _HEAD_DIM**-0.5
    mesh = _local_mesh(tp=4, sp=1)
    profile = resolve_attention_profile(
        query.shape, key.shape, mesh.shape, compact_statistics=compact_statistics
    )

    assert key.shape[2] == value.shape[2] == 1
    assert profile.block_q == 256
    assert profile.block_kv_outer == 512

    def vanilla_forward(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return vanilla_attention(q, k, v, allowed, scale=scale)

    vanilla_run = jax.jit(_value_and_vjp(vanilla_forward))
    vanilla_result = vanilla_run(query, key, value, cotangent)

    with jax.set_mesh(mesh):
        metadata = prepare_attention_metadata(segments, mesh=mesh, profile=profile)
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(metadata.q_block_segments))[0, 0],
            np.asarray([1, 3 if segment_count == 8 else 11], dtype=np.int32),
        )

        def efficient_forward(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
            return efficient_attention(
                q,
                k,
                v,
                metadata,
                mesh=mesh,
                profile=profile,
                scale=scale,
            )

        def splash_forward(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
            output = splash_attention(
                q,
                k,
                v,
                segments,
                scale=scale,
                block_q=profile.block_q,
                block_k=profile.block_kv_outer,
                mesh=mesh,
            )
            # Direct Splash treats zero as another segment.  The production
            # wrapper applies this query-validity gate; Efficient must implement
            # the equivalent fully-masked-row zero semantics internally.
            return output * valid[:, :, None, None].astype(output.dtype)

        efficient_run = jax.jit(_value_and_vjp(efficient_forward))
        splash_run = jax.jit(_value_and_vjp(splash_forward))
        distributed = _distributed_inputs(mesh, query, key, value, cotangent)
        efficient_result = efficient_run(*distributed)
        splash_result = splash_run(*distributed)

    _assert_result_matches(efficient_result, vanilla_result, actual_name="efficient")
    _assert_result_matches(splash_result, vanilla_result, actual_name="splash")

    padding = np.asarray(segments == 0)
    for implementation, result in (
        ("efficient", efficient_result),
        ("splash", splash_result),
        ("vanilla", vanilla_result),
    ):
        for name, array in zip(("out", "dq", "dk", "dv"), result, strict=True):
            _assert_exact_zero(f"{implementation} {name} on padding", array, padding)

    # Reuse the same compiled programs with a cotangent isolated to segment 2.
    # Any gradient on padding or either neighbouring packed record is leakage.
    isolated_cotangent = jnp.where(
        (segments == 2)[:, :, None, None],
        cotangent,
        jnp.asarray(0.0, cotangent.dtype),
    )
    isolated_vanilla = vanilla_run(query, key, value, isolated_cotangent)
    with jax.set_mesh(mesh):
        isolated_distributed = _distributed_inputs(mesh, query, key, value, isolated_cotangent)
        isolated_efficient = efficient_run(*isolated_distributed)
        isolated_splash = splash_run(*isolated_distributed)

    _assert_result_matches(isolated_efficient, isolated_vanilla, actual_name="isolated efficient")
    _assert_result_matches(isolated_splash, isolated_vanilla, actual_name="isolated splash")

    outside_segment_two = np.asarray(segments != 2)
    inside_segment_two = np.asarray(segments == 2)
    for implementation, result in (
        ("efficient", isolated_efficient),
        ("splash", isolated_splash),
        ("vanilla", isolated_vanilla),
    ):
        for name, gradient in zip(("dq", "dk", "dv"), result[1:], strict=True):
            _assert_exact_zero(
                f"isolated {implementation} {name} outside segment 2",
                gradient,
                outside_segment_two,
            )
            assert np.any(_host_float32(gradient)[inside_segment_two] != 0.0), (
                f"isolated {implementation} {name} comparison is vacuous"
            )

    # Perturb record 1 only, then reuse the compiled Efficient value+VJP.  The
    # primal and all isolated-cotangent gradients on record 2 must remain
    # bitwise unchanged; a tolerance here could hide packed-mask leakage.
    record_one = (segments == 1)[:, :, None, None]
    perturbed_query = jnp.where(record_one, query + jnp.asarray(0.25, query.dtype), query)
    perturbed_key = jnp.where(record_one, key - jnp.asarray(0.125, key.dtype), key)
    perturbed_value = jnp.where(record_one, value + jnp.asarray(0.5, value.dtype), value)
    with jax.set_mesh(mesh):
        perturbed_distributed = _distributed_inputs(
            mesh,
            perturbed_query,
            perturbed_key,
            perturbed_value,
            isolated_cotangent,
        )
        perturbed_efficient = efficient_run(*perturbed_distributed)

    for name, baseline, perturbed in zip(
        ("out", "dq", "dk", "dv"),
        isolated_efficient,
        perturbed_efficient,
        strict=True,
    ):
        baseline_record_two = _host_float32(baseline)[inside_segment_two]
        perturbed_record_two = _host_float32(perturbed)[inside_segment_two]
        np.testing.assert_array_equal(
            perturbed_record_two,
            baseline_record_two,
            err_msg=f"Efficient {name} leaked record-1 perturbations into record 2",
        )

    assert efficient_result[2].shape[2] == efficient_result[3].shape[2] == 1


@pytest.mark.parametrize("compact_statistics", [False, True])
def test_tp2_sp2_cross_shard_segment_vjp_reduces_remote_dkv(compact_statistics: bool) -> None:
    batch, sequence, query_heads = 1, 512, 4
    query = _bf16_normal(200, (batch, sequence, query_heads, _HEAD_DIM))
    key = _bf16_normal(201, (batch, sequence, 1, _HEAD_DIM))
    value = _bf16_normal(202, (batch, sequence, 1, _HEAD_DIM))
    full_cotangent = _bf16_normal(203, query.shape, scale=0.02)
    positions = jnp.arange(sequence)[None, :]
    # Segment 2 crosses the SP boundary at token 256.  Only queries on SP shard
    # 1 receive a cotangent, so gradients on its keys in [96,160) prove that
    # dKV was accumulated from remote queries and reduced over both TP and SP.
    segments = jnp.where(
        positions < 71,
        jnp.asarray(1, jnp.int32),
        jnp.where(
            positions < 333,
            jnp.asarray(2, jnp.int32),
            jnp.asarray(3, jnp.int32),
        ),
    )
    cotangent = jnp.where(
        ((positions >= 288) & (positions < 321))[:, :, None, None],
        full_cotangent,
        jnp.asarray(0.0, full_cotangent.dtype),
    )
    valid = jnp.ones((batch, sequence), dtype=jnp.bool_)
    allowed = make_causal_segment_mask(valid, segments)
    scale = _HEAD_DIM**-0.5
    mesh = _local_mesh(tp=2, sp=2)
    profile = resolve_attention_profile(
        query.shape, key.shape, mesh.shape, compact_statistics=compact_statistics
    )

    assert key.shape[2] == value.shape[2] == 1
    assert sequence // int(mesh.shape["sp"]) % profile.block_q == 0

    def vanilla_forward(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        return vanilla_attention(q, k, v, allowed, scale=scale)

    vanilla_run = jax.jit(_value_and_vjp(vanilla_forward))
    vanilla_result = vanilla_run(query, key, value, cotangent)

    with jax.set_mesh(mesh):
        metadata = prepare_attention_metadata(segments, mesh=mesh, profile=profile)

        def efficient_forward(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
            return efficient_attention(
                q,
                k,
                v,
                metadata,
                mesh=mesh,
                profile=profile,
                scale=scale,
            )

        efficient_run = jax.jit(_value_and_vjp(efficient_forward))
        distributed = _distributed_inputs(mesh, query, key, value, cotangent)
        efficient_result = efficient_run(*distributed)

    _assert_result_matches(efficient_result, vanilla_result, actual_name="tp2/sp2 efficient")

    outside_cross_shard_segment = np.asarray(segments != 2)
    for name, gradient in zip(("dq", "dk", "dv"), efficient_result[1:], strict=True):
        _assert_exact_zero(
            f"tp2/sp2 {name} outside cross-shard segment",
            gradient,
            outside_cross_shard_segment,
        )

    remote_key_positions = np.zeros((batch, sequence), dtype=np.bool_)
    remote_key_positions[:, 96:160] = True
    for name, actual, expected in (
        ("dk", efficient_result[2], vanilla_result[2]),
        ("dv", efficient_result[3], vanilla_result[3]),
    ):
        actual_remote = _host_float32(actual)[remote_key_positions]
        expected_remote = _host_float32(expected)[remote_key_positions]
        assert np.any(expected_remote != 0.0), f"vanilla remote {name} is unexpectedly vacuous"
        assert np.any(actual_remote != 0.0), f"Efficient failed to produce remote {name}"

        second_compute_tile = np.zeros((batch, sequence), dtype=np.bool_)
        second_compute_tile[:, 288:321] = True
        actual_second_tile = _host_float32(actual)[second_compute_tile]
        expected_second_tile = _host_float32(expected)[second_compute_tile]
        assert np.any(expected_second_tile != 0.0), (
            f"vanilla second-compute-tile {name} is unexpectedly vacuous"
        )
        assert np.any(actual_second_tile != 0.0), (
            f"Efficient skipped reachable second-compute-tile {name}"
        )

    assert efficient_result[2].shape[2] == efficient_result[3].shape[2] == 1
