"""CPU numerical gates for shard-local decoder projection tiles.

Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 to include TP/SP
and DP tests.  Single-device numerical and validation gates always execute.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from causal_trainer.modeling.architecture import linear, rms_norm
from causal_trainer.modeling.token_tiling import (
    tiled_final_norm,
    tiled_norm_mlp,
    tiled_norm_qkv,
    tiled_output_projection,
)


def _mesh(shape):
    count = int(np.prod(shape))
    if jax.local_device_count() < count:
        pytest.skip(f"requires {count} local devices; use four virtual CPU devices")
    devices = np.asarray(jax.local_devices()[:count], dtype=object).reshape(shape)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def _sample(shape, seed, scale=0.05):
    return jnp.asarray(np.random.default_rng(seed).normal(0.0, scale, shape), jnp.bfloat16)


def _inputs():
    hidden = _sample((2, 14, 16), 1, scale=0.3)
    scale = jnp.asarray(np.linspace(0.6, 1.2, 16), jnp.bfloat16)
    mlp = {
        "gate_proj": {"kernel": _sample((16, 24), 2)},
        "up_proj": {"kernel": _sample((16, 24), 3)},
        "down_proj": {"kernel": _sample((24, 16), 4)},
    }
    attention = {
        "q_proj": {"kernel": _sample((16, 16), 5), "bias": _sample((16,), 6)},
        "k_proj": {"kernel": _sample((16, 4), 7), "bias": _sample((4,), 8)},
        "v_proj": {"kernel": _sample((16, 4), 9), "bias": _sample((4,), 10)},
        "o_proj": {"kernel": _sample((16, 16), 11)},
    }
    return hidden, scale, mlp, attention


def _compare_forward_and_vjp(actual, reference, arguments):
    expected = reference(*arguments)
    cotangents = jax.tree.map(lambda value: _sample(value.shape, 71, scale=0.07), expected)

    def run(function, *args):
        output, backward = jax.vjp(function, *args)
        return output, backward(cotangents)

    actual_output, actual_gradients = jax.jit(lambda *args: run(actual, *args))(*arguments)
    expected_output, expected_gradients = jax.jit(lambda *args: run(reference, *args))(*arguments)
    for observed, expected_value in zip(
        jax.tree.leaves((actual_output, actual_gradients)),
        jax.tree.leaves((expected_output, expected_gradients)),
        strict=True,
    ):
        observed = np.asarray(observed, dtype=np.float32)
        expected_value = np.asarray(expected_value, dtype=np.float32)
        assert np.all(np.isfinite(observed))
        # Tile/TP decomposition changes reduction order. Comparisons cover
        # BF16 numerical equivalence rather than bit identity.
        # A norm after a projection can produce nearly cancelling weight
        # gradients. Scale the absolute tolerance to the tensor peak so those
        # small entries do not demand precision beyond BF16 reduction order.
        absolute_tolerance = max(0.002, float(np.max(np.abs(expected_value))) * 0.004)
        np.testing.assert_allclose(observed, expected_value, rtol=0.05, atol=absolute_tolerance)
        denominator = max(float(np.linalg.norm(expected_value)), 1e-8)
        assert float(np.linalg.norm(observed - expected_value)) / denominator < 0.015
    return actual_output


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2), (2, 1, 1, 2, 1)])
@pytest.mark.parametrize("chunk_size", [4, 32])
def test_tiled_norm_mlp_matches_untiled_forward_and_parameter_input_gradients(shape, chunk_size):
    mesh = _mesh(shape)
    hidden, scale, mlp, _ = _inputs()

    def reference(value, norm_scale, params):
        normalized = rms_norm(value, norm_scale, 1e-6, jnp.bfloat16)
        gate = jax.nn.silu(linear(normalized, params["gate_proj"], jnp.bfloat16))
        up = linear(normalized, params["up_proj"], jnp.bfloat16)
        return linear(gate * up, params["down_proj"], jnp.bfloat16)

    def actual(value, norm_scale, params):
        return tiled_norm_mlp(value, norm_scale, params, mesh=mesh, rms_norm_eps=1e-6, chunk_size=chunk_size)

    result = _compare_forward_and_vjp(actual, reference, (hidden, scale, mlp))
    assert result.sharding.is_equivalent_to(NamedSharding(mesh, P(("dp", "fsdp"), "sp", "tp")), 3)


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2), (2, 1, 1, 2, 1)])
@pytest.mark.parametrize("bias", [False, True])
def test_tiled_qkv_preserves_tp_sp_order_and_bias_gradients(shape, bias):
    mesh = _mesh(shape)
    hidden, scale, _, attention = _inputs()
    projections = {name: attention[name] for name in ("q_proj", "k_proj", "v_proj")}
    if not bias:
        projections = {name: {"kernel": params["kernel"]} for name, params in projections.items()}

    def reference(value, norm_scale, params):
        normalized = rms_norm(value, norm_scale, 1e-6, jnp.bfloat16)
        return tuple(
            linear(normalized, params[name], jnp.bfloat16).reshape(2, 14, heads, 4)
            for name, heads in (("q_proj", 4), ("k_proj", 1), ("v_proj", 1))
        )

    def actual(value, norm_scale, params):
        return tiled_norm_qkv(
            value, norm_scale, params, mesh=mesh, rms_norm_eps=1e-6,
            num_attention_heads=4, head_dim=4, chunk_size=4,
        )

    q, k, v = _compare_forward_and_vjp(actual, reference, (hidden, scale, projections))
    assert q.sharding.is_equivalent_to(NamedSharding(mesh, P(("dp", "fsdp"), "sp", "tp", None)), 4)
    for shared in (k, v):
        assert shared.sharding.is_equivalent_to(NamedSharding(mesh, P(("dp", "fsdp"), "sp", None, None)), 4)


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2), (2, 1, 1, 2, 1)])
def test_tiled_output_projection_and_final_norm_forward_backward(shape):
    mesh = _mesh(shape)
    hidden, scale, _, attention = _inputs()

    def actual(value, norm_scale, params):
        projected = tiled_output_projection(value, params, mesh=mesh, chunk_size=4)
        return tiled_final_norm(projected, norm_scale, mesh=mesh, rms_norm_eps=1e-6, chunk_size=4)

    def reference(value, norm_scale, params):
        return rms_norm(linear(value, params, jnp.bfloat16), norm_scale, 1e-6, jnp.bfloat16)

    _compare_forward_and_vjp(actual, reference, (hidden, scale, attention["o_proj"]))


def test_tile_contract_rejects_unsupported_mesh_and_dtype():
    hidden, scale, mlp, _ = _inputs()
    mesh = _mesh((1, 1, 1, 1, 1))
    with pytest.raises(TypeError, match="BF16"):
        tiled_norm_mlp(hidden.astype(jnp.float32), scale, mlp, mesh=mesh, rms_norm_eps=1e-6)
    with pytest.raises(ValueError, match="positive"):
        tiled_final_norm(hidden, scale, mesh=mesh, rms_norm_eps=1e-6, chunk_size=0)
    with pytest.raises(TypeError, match="positive integer"):
        tiled_final_norm(hidden, scale, mesh=mesh, rms_norm_eps=1e-6, chunk_size=True)


def test_tile_contract_rejects_fsdp_without_reinterpreting_it_as_sp():
    mesh = _mesh((1, 2, 1, 2, 1))
    hidden, scale, mlp, _ = _inputs()
    with pytest.raises(ValueError, match="fsdp=ep=1"):
        tiled_norm_mlp(hidden, scale, mlp, mesh=mesh, rms_norm_eps=1e-6)


def test_hundreds_of_tiles_accumulate_weight_gradients_before_bf16_rounding():
    mesh = _mesh((1, 1, 1, 2, 2))
    hidden = jnp.ones((1, 1024, 4), jnp.bfloat16)
    kernel = jnp.ones((4, 4), jnp.bfloat16)

    def loss(weight):
        result = tiled_output_projection(hidden, {"kernel": weight}, mesh=mesh, chunk_size=1)
        return jnp.sum(result.astype(jnp.float32))

    gradient = jax.jit(jax.grad(loss))(kernel)
    assert gradient.dtype == jnp.bfloat16
    # Each SP shard contributes 512 one-token tiles. BF16 scan accumulation
    # would stop changing after 256 additions and incorrectly return 512.
    np.testing.assert_array_equal(np.asarray(gradient, dtype=np.float32), np.full((4, 4), 1024.0))


def test_sequence_loop_is_shard_local_and_hidden_collectives_are_tile_sized():
    mesh = _mesh((1, 1, 1, 2, 2))
    hidden, scale, mlp, _ = _inputs()

    def actual(value, norm_scale, params):
        return tiled_norm_mlp(value, norm_scale, params, mesh=mesh, rms_norm_eps=1e-6, chunk_size=4)

    scans = []
    gathers = []

    def visit(value):
        if hasattr(value, "jaxpr"):
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            for equation in value.eqns:
                if equation.primitive.name == "scan":
                    scans.append(equation.params["length"])
                if equation.primitive.name == "all_gather":
                    gathers.append((equation.params["axis_name"], equation.invars[0].aval.shape))
                for parameter in equation.params.values():
                    visit(parameter)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(jax.make_jaxpr(actual)(hidden, scale, mlp))
    # Global S=14 becomes local S=7, requiring two padded tiles, not four
    # global tiles. No SP all-gather or complete local hidden TP gather occurs.
    assert scans == [2]
    assert gathers == [(("tp",), (2, 4, 8))]
