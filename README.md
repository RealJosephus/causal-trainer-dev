# Causal Trainer

Causal Trainer is a high-performance JAX platform for CausalLM model training on Cloud TPU infrastructure. It unifies pretraining, supervised fine-tuning, and parameter-efficient adaptation in a native multi-host execution stack, with distributed data preparation, TPU-optimized kernels, and scalable checkpointing.

## Capabilities

- Segment-isolated packing for attention, shifted loss, assistant-only supervision, and EndPrompt.
- Standalone preprocessing into process-assigned Hugging Face Parquet shards.
- Bounded-memory Hugging Face dataset streaming with online packing.
- A custom Efficient attention kernel that combines tiled execution, online softmax, and a custom backward path to reduce HBM traffic.
- Projection-fused Cut cross-entropy and chunked Pallas cross-entropy for reduced activation memory.
- Streaming multi-host checkpoint export with bounded host memory.

## Requirements

- Python 3.11 or later.
- Offline preparation requires a serialized fast tokenizer containing `tokenizer.json`.
- Training requires JAX and JAXLIB 0.10.x; TPU training also requires a compatible `libtpu` runtime.
- Training requires a supported checkpoint in Hugging Face tensor layout.

The trainer reads Hugging Face configuration, tokenizer, and Safetensors artifacts directly. Transformers is not a runtime dependency.

## Installation

Create an isolated environment. The base install is deliberately limited to the dataset, Hugging Face Hub, template, NumPy, and tokenizer stack needed by `causal-prepare`; it does not install JAX or checkpoint dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install either the complete CPU training stack or the complete TPU training stack separately, and combine `training` with `test` for CPU development:

```bash
python -m pip install -e '.[training]'
python -m pip install -e '.[tpu]'
python -m pip install -e '.[training,test]'
```

The package installs `causal-train` and the standalone `causal-prepare` data command. `causal-train` requires either the `training` or `tpu` extra. The equivalent module entry points are `python -m causal_trainer.train` and `python -m causal_trainer.prepare`.

## Quick start

### Raw-text pretraining

```bash
causal-train \
  --repo_id /path/to/model-or-hf-repo \
  --dataset_name /path/to/train.jsonl \
  --dataset_text_field text \
  --max_sequence_length 4096 \
  --packing True \
  --total_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./output-pt
```

Raw text is tokenized with `add_special_tokens=False`. EOS placement is preserved from the source records.

### Supervised fine-tuning

```bash
causal-train \
  --repo_id /path/to/model-or-hf-repo \
  --dataset_name /path/to/messages.jsonl \
  --dataset_text_field messages \
  --max_sequence_length 4096 \
  --packing True \
  --assistant_only_loss True \
  --total_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --output_dir ./output-sft
```

Assistant-only supervision requires a chat template that marks assistant generation spans with `{% generation %}` and `{% endgeneration %}`. `--assistant_only_loss True` supervises every marked assistant span. Use `--last_assistant_only_loss True` instead to supervise only the final contiguous marked assistant span in each source conversation. The two options are mutually exclusive and require `--dataset_text_field messages`. Records without a valid supervised target after truncation are filtered.

## Model and tokenizer sources

`--repo_id` accepts either a local checkpoint directory or a Hugging Face repository ID. Source checkpoints may contain sharded Safetensors weights. The configuration is validated against the architecture supported by this package before parameters are loaded.

By default, tokenizer assets are read from the model source. Use `--processor_repo_id` to select a separate local directory or repository. The source must contain `tokenizer.json`; standard companion assets are preserved when checkpoints are exported, including:

- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `chat_templates/*.jinja`

Use the `HF_TOKEN` environment variable for authenticated repositories. `--revision` selects the model revision. When `--processor_repo_id` points to a separate tokenizer repository, pin it independently with `--processor_revision`; otherwise the tokenizer is read from the already resolved model revision.

## Dataset inputs

`--dataset_name` accepts a Hugging Face dataset ID or a local JSON, JSONL, Parquet, CSV, or TSV source. `--dataset_split` defaults to `train`.

Raw-text records contain a string field, which defaults to `text`:

```json
{"text": "A training document."}
```

Conversational records contain a `messages` field:

```json
{
  "messages": [
    {"role": "user", "content": "Explain sequence packing."},
    {
      "role": "assistant",
      "content": "Sequence packing combines multiple samples into one row."
    }
  ]
}
```

Message records may also provide `tools` and `chat_template_kwargs` when the selected chat template requires them.

When `--assistant_only_loss True` or `--last_assistant_only_loss True` is active, an assistant message may set an optional boolean `loss` field. `false` keeps that turn in the rendered context while excluding its chat-template generation span from supervision. A missing or null value remains trainable. Non-boolean, non-null assistant values are rejected. Datasets without assistant `loss` annotations retain the existing rendering and masking path unchanged. Selective message loss requires exactly one direct `{% generation %}` block per assistant message in the main message loop; do not hide these blocks inside macros or capture blocks.

### Offline prepared datasets

Use `causal-prepare` on a separate, ordinary single machine to finish tokenization, optional packing, attention/position/segment masks, assistant loss masks, loss weights, and EndPrompt before the accelerator job starts. The command uses the existing dataset and tokenizer dependencies and does not import JAX, Transformers, PyTorch, or the training runtime.

Both dataset forms are supported. Use `--dataset_text_field text` (or another string-column name) for ordinary text, and use `--dataset_text_field messages` for role/content conversations rendered with the tokenizer's chat template. Assistant-only loss options are available for `messages` input.

```bash
causal-prepare \
  --dataset_name org/raw-corpus \
  --dataset_split train \
  --dataset_text_field text \
  --tokenizer_name_or_path org/model \
  --tokenizer_revision TOKENIZER_COMMIT \
  --output_dir ./prepared-corpus \
  --num_shards 8 \
  --max_sequence_length 4096 \
  --packing True \
  --packing_batch_size 1000 \
  --hub_repo_id org/prepared-corpus
```

For conversational SFT data, change the input mode and optionally precompute assistant-only supervision:

```bash
causal-prepare \
  --dataset_name org/messages-corpus \
  --dataset_split train \
  --dataset_text_field messages \
  --tokenizer_name_or_path org/model \
  --output_dir ./prepared-messages \
  --num_shards 8 \
  --max_sequence_length 4096 \
  --packing True \
  --assistant_only_loss True \
  --hub_repo_id org/prepared-messages
```

`--num_shards` is an artifact-layout choice, not a compiled-in host or device count. The command processes source intervals sequentially on one machine and writes independent Parquet files plus a small `causal-trainer-prepared.json` manifest. EndPrompt continues to use the original global source-row index across file boundaries. The manifest records the current canonical schema, tokenizer/source identity, preprocessing settings, row counts, and file checksums; this project intentionally expects prepared artifacts to be produced by the current code rather than maintaining historical format variants. When preprocessing uses a separate tokenizer repository or revision, pass that same source to training with `--processor_repo_id` and `--processor_revision` so the prepared tokenizer digest can be verified.

Train directly from the result with no online tokenizer or packer:

```bash
causal-train \
  --repo_id org/model \
  --dataset_name org/prepared-corpus \
  --dataset_format prepared \
  --dataset_split train \
  --dataset_revision DATASET_COMMIT \
  --max_sequence_length 4096 \
  --total_batch_size 32 \
  --output_dir ./output
```

For a remote prepared repository, the trainer resolves `--dataset_revision` (or the repository's default branch when it is omitted) to one immutable commit before reading either the manifest or its Parquet files.

For materialized input, the trainer derives the assignment from the actual JAX process index/count and downloads only that process's contiguous interval of physical files. The files are then merged over the initialized JAX network in manifest order, so each real prepared row is cached on only one process while all processes retain the reconstructed dataset in host RAM for the existing global-batch path. A common 8-host × 4-device setup can use eight files and one JAX process per host, but neither value is hard-coded; each process controls all of its local devices normally.

When even one process's assigned files or the reconstructed host arrays are too large, stream the prepared artifact instead:

```text
--dataset_format prepared \
--dataset_streaming True \
--dataset_revision DATASET_COMMIT
```

Prepared streaming validates and batches the already-final canonical rows directly. It never tokenizes, packs, or recomputes masks, and it keeps only bounded shuffle/window/prefetch state locally. To preserve the trainer's global-batch placement contract, every JAX process reproduces the same remote prepared stream; this mode minimizes local disk rather than network reads. Since a packed row can be large, use `--no-shuffle` for offline ordering or set a deliberately small `--streaming_shuffle_buffer` instead of retaining the default 10,000 rows.

### Streaming datasets

Enable `--dataset_streaming True` to train from a Hugging Face dataset that is larger than local memory or disk. Streaming keeps only bounded working state in memory: the shuffle buffer, the active packing window, and the configured prefetched batches. It does not materialize or download the complete split before training.

```text
--dataset_streaming True \
--dataset_config_name optional-subset \
--dataset_revision dataset-commit \
--streaming_shuffle_buffer 10000
```

`--dataset_config_name` selects an optional Hugging Face dataset configuration. `--dataset_revision` pins the dataset independently of the model `--revision` and the optional separate `--processor_revision`.

One complete traversal of the source split is exactly one streaming epoch. Without an explicit positive `--max_steps`, `--num_train_epochs` must therefore be a positive integer. `--max_steps` may intentionally stop in the middle of an epoch.

When the Hugging Face split metadata provides its number of examples, epoch-driven training uses it only as an initial estimate for progress, ETA, and schedule horizon. The metadata count may be inaccurate: it never caps iteration, triggers retry, or decides whether training is complete. One epoch always ends at the stream's real EOF, and actual committed source examples are allowed to differ from the estimate. After each completed real epoch, the remaining progress total and ETA are re-estimated from observed consumption. Packing and filtering can make source examples differ from packed rows or optimizer steps. A source example is committed only after its corresponding device update completes successfully. If the metadata estimate is unavailable, progress begins without a percentage and obtains an observed estimate after the first real epoch. With an explicit `--max_steps`, the progress bar uses that exact step limit and source examples remain an additional metric.

For epoch-driven streaming with a metadata estimate, cosine and linear learning-rate decay use a scheduler horizon fixed at 105% of the metadata-derived source-example total; `warmup_steps` is converted to source-example units using the configured effective global batch. Within each real epoch, scheduler progress is capped at the original, unbuffered metadata row count. If the real stream contains more rows, every batch beyond that position keeps the same learning rate while the data continues to real EOF. The reserved 5% tail is not consumed, so epoch-driven streaming is not forced to reach `learning_rate_end`. Real EOF advances training to the next epoch but does not correct or extend the scheduler. Without a row-count estimate, epoch-driven training remains enabled and uses optimizer-step warmup followed by a constant learning rate because no defensible decay horizon exists.

Streaming shuffle uses `--streaming_shuffle_buffer` records and remains deterministic for the configured seed and epoch.

Transient Hugging Face network, transport, and server read failures are retried without an attempt limit. Retry delays begin at `--hf_retry_initial_delay` seconds and back off to at most `--hf_retry_max_delay` seconds. A disconnected stream is reopened at the same epoch, its delivered prefix is replayed, and a rolling digest rejects a changed prefix instead of mixing dataset revisions. A natural EOF completes the epoch regardless of the metadata estimate. Authentication or authorization failures, a missing dataset, malformed data, and preprocessing errors are not retried.

### Packing

Enable fixed-length packing with `--packing True`. Each source record receives an independent segment ID, and position IDs restart at the beginning of each segment. Attention and shifted language-model loss both enforce segment boundaries.

With `--last_assistant_only_loss True`, the final assistant span is selected independently in every source conversation before packing. A packed row therefore retains one final supervised span per trainable source record, while cross-record prediction edges remain masked by segment IDs.

`--packing_batch_size` controls the per-process packing window. The final short window is retained.

For streaming input, each completed packing window is emitted immediately. The final short window is flushed when the source epoch is exhausted, and packing state never crosses an epoch boundary.

### Distributed preprocessing

The default preprocessing mode is `shard_then_merge`:

```text
--preprocessing_mode shard_then_merge
```

For materialized datasets, source rows are divided among JAX processes, preprocessed locally, and merged over the initialized JAX network before training. Use `--preprocessing_mode replicated` to preprocess the complete dataset on every process. `--preprocessing_num_workers` controls tokenizer worker processes within each JAX process.

Streaming uses a deterministic replicated-read design: every JAX process opens the same deterministic stream and constructs the same global batches before JAX retains its addressable device slices. Pinning `--dataset_revision` to an immutable commit is strongly recommended. Before a batch is released to training, processes collectively compare `BATCH`/`EOF`/`ERROR` state and a digest of batch contents, shapes, dtypes, epoch, and source cursor; successful local device placement is part of this coordinated boundary. This avoids mixed-host batches and mismatched collective order. The design avoids materializing and merging the complete prepared dataset, while intentionally duplicating source reads across hosts. `--preprocessing_mode` is ignored for streaming input, and `--preprocessing_num_workers` is not supported. Source progress is counted once for the logical global stream, not once per process.

Streaming checkpoints store both committed source progress and a rolling digest of every trained global batch. Resume replays the bounded stream from the beginning, verifies the exact checkpoint boundary even when it coincides with EOF, and rejects changed historical data.

## Training modes

### Full-parameter training

Full-parameter training is the default. The base parameters and AdamW state use the configured parameter dtype, while numerically sensitive normalization, softmax, loss-reduction, and gradient-scratch operations use FP32 where required.

To keep the token embedding, language-model head, and every normalization scale frozen while training the remaining full-rank parameters:

```text
--frozen_parameters "lm_head|embed|norm"
```

Supported components are `lm_head`, `embed`, and `norm`; separate multiple components with `|` or `,`. The `norm` component includes the final normalization and both normalization scales in every decoder layer. When omitted, no full-rank parameters are frozen.

The default precision options are:

```text
--param_dtype bfloat16
--dtype bfloat16
--gradient_checkpointing nothing_saveable
```

### LoRA

Enable LoRA with:

```text
--lora True \
--lora_rank 256
```

Adapters are applied to the attention and gated-MLP projections. Embeddings and the language model head remain frozen by default.
When FSDP is greater than one, `--lora_rank` must be divisible by the FSDP mesh size because FSDP shards the adapter rank axis.

To train the embedding and language model head alongside the adapters:

```text
--lora True \
--lora_train_embed_and_lm_head True
```

LoRA exports merged model weights by default. To write a PEFT-compatible adapter instead:

```text
--lora True \
--lora_save_adapter_only True
```

Adapter-only output contains `adapter_model.safetensors` and `adapter_config.json`. When embedding and head training is enabled, those tensors are included as PEFT modules to save.

### EndPrompt

EndPrompt appends a terminal prompt at an anchored logical position while preserving the source token stream. It supports raw text, messages, packing, assistant-only supervision, full-parameter training, and LoRA.

```text
--endprompt_enable True \
--endprompt_logical_length 2097152 \
--endprompt_logical_length_min 4096 \
--endprompt_prompts "This is the end of text, please pay attention here" \
--endprompt_context_loss_weight 1.0 \
--endprompt_prompt_loss_weight 0.1
```

Separate multiple terminal prompts with `||`. Selection and logical-position sampling are deterministic for each source record.

## Attention and loss backends

### Attention

The default attention backend is:

```text
--attn_mechanism efficient
```

The default `efficient` backend uses a tiled custom kernel with online softmax and a custom VJP, avoiding materialization of the full attention matrix. It supports packed segments and sequence parallelism.

Official JAX Splash is available as an independent backend selected with `--attn_mechanism splash`. The `--block_size_q` and `--block_size_k` options apply to Splash only. The `vanilla` backend provides a reference implementation for numerical checks.

### Optional MoBA

`--attention_method dense` is the default. MoBA is an optional approximation that preserves the existing MQA projections, parameter layout, and RoPE settings. Changing a tuning parameter alone does not enable MoBA.

Enable MoBA in addition to the existing training arguments:

```bash
# MoBA on every layer; top-k includes the mandatory current block.
--attention_method moba \
--moba_block_size 4096 \
--moba_top_k 12
```

`--attn_mechanism` selects the dense backend for retained dense layers and dense recovery. Approximate layers use Pallas forward/backward kernels on TPU and the JAX reference on CPU. Explicit `--attention_sparse_backend pallas` requires Pallas; kernel errors do not trigger a reference fallback. MoBA requires MQA and zero stochastic attention dropout. The Pallas kernels require head dimension 128.

| Option | Default | Meaning |
|---|---:|---|
| `attention_method` | `dense` | `dense` or `moba`. |
| `attention_sparse_backend` | `auto` | Pallas on TPU, reference on CPU; explicit `pallas` / `reference` overrides. |
| `moba_block_size` | 4096 | Tokens per sample-relative routing block. |
| `moba_top_k` | 12 | Maximum selected blocks per Query/head, including the current block. |
| `attention_query_chunk_size` | 128 | Query workspace target. Pallas aligns to 128 matrix rows across local heads and bounds the mask workspace; K512 uses at most Q64 with 8 local heads or Q128 with 4. |
| `attention_key_chunk_size` | 512 | KV workspace target; Pallas uses 128-aligned tiles and may widen grouped history tiles for large local query streams. Independent of the routing block size. |
| `attention_dense_prefix_layers` | 0 | First layers kept dense. |
| `attention_dense_suffix_layers` | 0 | Last layers kept dense. |
| `attention_dense_after_step` | unset | Switch all layers to the chosen dense backend at this zero-based optimizer step. |

Dashed spellings such as `--attention-method` are also accepted. Dense prefix and suffix counts must fit within the model's layer count. For example, `--attention_dense_suffix_layers 4` retains four dense layers at the end of the model.

**MoBA.** Each sample starts its own block grid. Each Query head selects up to `top_k - 1` completed historical blocks using their mean Keys and includes its current block with a causal mask. One softmax covers all selected tokens. A budget covering the entire sample history recovers dense connections. This follows [MoBA](https://arxiv.org/abs/2502.13189), adapted to shared MQA Keys and Values.

The TPU current-block kernel schedules sample-relative causal intervals in one Pallas grid. Forward query tiles run independently; backward tiles accumulate shared Key/Value gradients in order. Grouped history execution selects its Query/Key tile sizes from the local query-row count while keeping the score workspace bounded. These execution choices preserve the configured routing block size and top-k.

MoBA uses the existing token batch and supports `text`, `messages`, packing, message loss weights, assistant-only loss, EndPrompt position jumps, prepared datasets, and streaming. Full updates, LoRA, frozen components, gradient accumulation, scanning/rematerialization, and loss backends retain their existing behavior and restrictions. Routing follows physical token order and sample boundaries independently of supervision masks.

`--attention_dense_after_step 12000` switches to dense attention at zero-based optimizer step 12000 while keeping optimizer state, data order, and the LR schedule. Resume with the same attention options to preserve the phase and counter. Exported weights retain the standard dense/MQA parameter layout.

### Cross-entropy

The default loss selection is:

```text
--loss_implementation auto
```

Available values are:

| Value | Description |
|---|---|
| `auto` | Selects Cut or Pallas according to mesh compatibility. |
| `cut` | Uses projection-fused cross-entropy on supported meshes. |
| `pallas` | Uses chunked projected logits with a TPU Pallas kernel. |
| `xla` | Uses the reference XLA implementation. |

`--loss_token_budget` controls the local projected-logits chunk used by the Pallas and XLA paths. Cut uses its own internal tiling.

### Execution controls

- `--mlp_chunk_size` controls static sequence tiling for the gated MLP. Use `0` to disable tiling. When omitted, the trainer selects a value from the training mode, context length, and sequence-parallel configuration.
- `--projection_chunk_size` controls shard-local token tiling for attention projections and normalization. The default `0` keeps the original projection path. With a positive value, hidden states remain feature-sharded between layers, and `--mlp_chunk_size` independently controls shard-local MLP tile sizes. MLP tiling with `sp>1` requires this projection path.
- `--remat_group_size` controls the number of decoder layers in each recomputation group. The default `1` keeps the original layer checkpointing. Larger values require `--scan_layers True`, `--gradient_checkpointing nothing_saveable`, and must divide the layer count.
- `--cut_chunk_size` controls shard-local token chunks for Cut loss, independently of decoder tiling. The default `0` keeps the original Cut execution; positive values require the Cut backend.
- `--scan_layers` represents the decoder layer loop with `lax.scan`. `--no-scan-layers` selects the unrolled form. Automatic scanning requires full-parameter training, `nothing_saveable` rematerialization and an untiled MLP. It applies at 512K and above, and at 128K (Dense or MoBA) / 256K (MoBA) for BF16 Efficient training with global batch 1. Dense 256K and explicitly tiled runs keep scanning opt-in.
- `--async_dispatch_steps` controls deferred host synchronization for completed step metrics.
- `--prefetch_batches` controls prepared global batches retained ahead of the active training step.

Projection tiling currently supports full-parameter BF16 training with Efficient attention, zero attention dropout, a single-example microbatch, and `dp=fsdp=ep=1`. It also uses compact attention statistics. These controls change storage and recomputation while preserving packed-segment boundaries, position IDs, padding, and loss weights. The new controls retain the original execution at their defaults; incompatible explicit settings raise an error. Non-default values are included in checkpoint compatibility checks, as with the existing execution controls.

## Distributed training

The mesh axes are ordered as:

```text
(dp, fsdp, ep, tp, sp)
```

The default mesh is:

```text
--sharding_axis=-1,1,1,4,1
```

`-1` infers the data-parallel dimension from the available device count. The trainer requires `ep=1`. Set `sp>1` explicitly when sequence parallelism is needed. `--sharding_dcn_axis` may be used to provide the corresponding inter-host mesh shape.

Launch the same command on every host. For example:

```bash
eopod run --retry 1 --worker all \
  "cd /root/causal-trainer && \
   PYTHONPATH=src .venv/bin/causal-train <arguments>"
```

Standard Cloud TPU environments are detected automatically. Explicit cluster configuration is also supported:

```text
--coordinator_address host:port \
--num_processes N \
--process_id R \
--local_device_ids 0,1,...
```

`--total_batch_size` is the global micro-batch size. The effective batch per optimizer update is:

```text
total_batch_size * gradient_accumulation_steps
```

## Checkpoints and resume

Every process participates in checkpoint collectives; process zero writes the artifacts. All processes must observe the same completed checkpoint contents when resuming a multi-host run.

The trainer writes:

| Training mode | Default output |
|---|---|
| Full-parameter | `model.safetensors` with config and tokenizer assets. |
| LoRA | A merged `model.safetensors` with config and tokenizer assets. |
| Adapter-only LoRA | `adapter_model.safetensors` and `adapter_config.json`. |

Periodic checkpoints are configured with:

```text
--save_steps 100 \
--save_total_limit 1
```

Enable `--save_optimizer_state True` when exact optimizer-state resume is required. Adapter-only LoRA supports optimizer-state checkpoints; merged LoRA exports are final model artifacts rather than resumable adapter checkpoints.

At startup, the trainer discovers the latest compatible completed checkpoint under `output_dir` and resumes automatically.

## Logging

Enable Weights & Biases on process zero with:

```text
--use_wandb True \
--wandb_project causal-trainer \
--wandb_run_name my-run
```

Additional controls include `--logging_steps`, `--track_memory`, and `--weight_distribution_log_steps`.

## License

Causal Trainer is licensed under the [Apache License 2.0](LICENSE).
