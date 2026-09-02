from types import SimpleNamespace

import pytest

from causal_trainer.modeling.config import ModelConfig
from causal_trainer.training.runner import _validate_supported_config


def _supported_raw_config() -> dict:
    return {
        "model_type": "opaque-metadata",
        "attention_bias": True,
        "attention_dropout": 0.0,
        "eos_token_id": 151650,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 13568,
        "max_position_embeddings": 2097152,
        "num_attention_heads": 32,
        "num_hidden_layers": 40,
        "num_key_value_heads": 1,
        "pad_token_id": 151329,
        "partial_rotary_factor": 0.5,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_style": "gpt-j",
        "rope_theta": 1_000_000_000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 185600,
    }


def _validation_args():
    return SimpleNamespace(
        max_sequence_length=4096,
        dataset_format="raw",
        endprompt_enable=False,
        total_batch_size=32,
        attn_mechanism="vanilla",
        mlp_chunk_size=0,
        lora=False,
        lora_rank=256,
    )


def _mesh(*, fsdp: int = 1, sp: int = 1):
    return SimpleNamespace(shape={"dp": 1, "fsdp": fsdp, "ep": 1, "tp": 1, "sp": sp})


def test_supported_architecture_ignores_opaque_model_type_metadata() -> None:
    raw = _supported_raw_config()
    config = ModelConfig.from_dict(raw)
    _validate_supported_config(raw, config, _validation_args(), _mesh())
    assert config.rotary_dim == 64
    assert config.rope_style == "gpt-j"
    assert config.query_width == 4096
    assert config.key_value_width == 128


def test_supported_architecture_rejects_a_missing_semantic_field() -> None:
    raw = _supported_raw_config()
    del raw["rope_theta"]

    with pytest.raises(ValueError, match="must define RoPE"):
        _validate_supported_config(raw, ModelConfig.from_dict(raw), _validation_args(), _mesh())


def test_transformers_v5_config_is_read_but_emits_the_fixed_source_format() -> None:
    raw = _supported_raw_config()
    raw["dtype"] = raw.pop("torch_dtype")
    raw["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": raw.pop("rope_theta"),
        "partial_rotary_factor": raw.pop("partial_rotary_factor"),
    }
    raw.pop("rope_scaling")
    raw.pop("sliding_window")
    raw.pop("use_sliding_window")

    config = ModelConfig.from_dict(raw)
    _validate_supported_config(raw, config, _validation_args(), _mesh())
    serialized = config.to_dict()
    assert "dtype" not in serialized
    assert "rope_parameters" not in serialized
    assert serialized["torch_dtype"] == "bfloat16"
    assert serialized["rope_scaling"] is None
    assert serialized["rope_style"] == "gpt-j"
    assert serialized["rope_theta"] == 1_000_000_000.0
    assert serialized["partial_rotary_factor"] == 0.5


def test_transformers_v5_canonical_values_take_priority_when_reading() -> None:
    raw = _supported_raw_config()
    raw["torch_dtype"] = "float32"
    raw["dtype"] = "bfloat16"
    raw["rope_theta"] = 10_000.0
    raw["partial_rotary_factor"] = 1.0
    raw["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 1_000_000_000.0,
        "partial_rotary_factor": 0.5,
    }

    config = ModelConfig.from_dict(raw)
    assert config.torch_dtype == "bfloat16"
    assert config.rope_theta == 1_000_000_000.0
    assert config.partial_rotary_factor == 0.5


def test_rope_style_accepts_gpt_neox_and_defaults_to_gpt_j() -> None:
    raw = _supported_raw_config()
    raw["rope_style"] = "gpt-neox"
    assert ModelConfig.from_dict(raw).rope_style == "gpt-neox"

    del raw["rope_style"]
    assert ModelConfig.from_dict(raw).rope_style == "gpt-j"


def test_rope_style_rejects_unknown_value() -> None:
    raw = _supported_raw_config()
    raw["rope_style"] = "llama"
    with pytest.raises(ValueError, match="rope_style"):
        ModelConfig.from_dict(raw)


def test_supported_architecture_rejects_incompatible_fsdp_partitioning() -> None:
    raw = _supported_raw_config()

    with pytest.raises(ValueError, match="FSDP=3 does not divide"):
        _validate_supported_config(raw, ModelConfig.from_dict(raw), _validation_args(), _mesh(fsdp=3))


def test_lora_rank_must_be_divisible_by_fsdp_partitioning() -> None:
    raw = _supported_raw_config()
    args = _validation_args()
    args.lora_rank = 3

    _validate_supported_config(raw, ModelConfig.from_dict(raw), args, _mesh(fsdp=2))

    args.lora = True
    with pytest.raises(ValueError, match="lora_rank=3 must be divisible by FSDP=2"):
        _validate_supported_config(raw, ModelConfig.from_dict(raw), args, _mesh(fsdp=2))

    args.lora_rank = 4
    _validate_supported_config(raw, ModelConfig.from_dict(raw), args, _mesh(fsdp=2))


def test_supported_architecture_accepts_sequence_parallel_activation_sharding() -> None:
    raw = _supported_raw_config()
    _validate_supported_config(
        raw,
        ModelConfig.from_dict(raw),
        _validation_args(),
        _mesh(sp=4),
    )


def test_sequence_parallel_rejects_global_sequence_mlp_scan() -> None:
    raw = _supported_raw_config()
    args = _validation_args()
    args.mlp_chunk_size = 1024
    with pytest.raises(ValueError, match="mlp_chunk_size must be 0 when SP>1"):
        _validate_supported_config(
            raw,
            ModelConfig.from_dict(raw),
            args,
            _mesh(sp=2),
        )


def test_sequence_parallel_requires_a_divisible_global_sequence() -> None:
    raw = _supported_raw_config()
    args = _validation_args()
    args.max_sequence_length = 4095
    with pytest.raises(ValueError, match="must be divisible by SP=4"):
        _validate_supported_config(raw, ModelConfig.from_dict(raw), args, _mesh(sp=4))


def test_splash_query_block_must_divide_the_sp_local_sequence() -> None:
    raw = _supported_raw_config()
    args = _validation_args()
    args.attn_mechanism = "splash"
    args.dtype = "bfloat16"
    args.block_size_q = 2048
    args.block_size_k = 128
    with pytest.raises(ValueError, match="SP-local query length 1024"):
        _validate_supported_config(raw, ModelConfig.from_dict(raw), args, _mesh(sp=4))
