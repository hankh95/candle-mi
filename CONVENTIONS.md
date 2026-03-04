# candle-mi Coding Conventions (Grit + Grit-MI Extensions)

This document describes the [Amphigraphic coding](https://github.com/PCfVW/Amphigraphic-Strict) conventions used in candle-mi. It is a superset of
the [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit).

## Annotation Patterns

Every annotation below is mandatory when the corresponding situation applies.

### `// TRAIT_OBJECT: <reason>`
Required on every `Box<dyn Trait>` or `&dyn Trait` usage.
> Example: `// TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch`

### `// EXHAUSTIVE: <reason>`
Required on `#[allow(clippy::exhaustive_enums)]`.
> Example: `// EXHAUSTIVE: internal dispatch enum; crate owns and matches all variants`

### `// EXPLICIT: <reason>`
Required when a match arm is intentionally a no-op, or when an imperative
loop is used instead of an iterator chain for a stateful computation.
> Example: `// EXPLICIT: WKV recurrence is stateful; .map() would hide the state update`

### `// PROMOTE: <reason>`
Required immediately before any `.to_dtype(DType::F32)?` call that promotes
a tensor from a lower-precision dtype (F16, BF16) to F32 for numerical
correctness. Common reasons include:

- **Numerical functions**: softmax, log, exp, norm, sqrt
- **Matmul precision**: decoder weights stored as BF16 on disk
- **Accumulation**: running sums, averages, WKV recurrence
- **Dot-product precision**: matching a Python reference implementation
- **DType extraction**: `to_vec1::<f32>()` from BF16 safetensors

The reason must be specific to the call site, not a generic "numerical stability".

> Example: `// PROMOTE: softmax over F16 produces NaN; compute in F32`
> Example: `// PROMOTE: decoder weights are BF16 on disk; F32 for matmul precision`
> Example: `// PROMOTE: WKV recurrence must be in F32 for numerical stability`

### `// CONTIGUOUS: <reason>`
Required immediately before any `.contiguous()?` call that precedes a matmul.
> Example: `// CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout`

### `// BORROW: <what is converted>`
Required on explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions (Grit Rule 2).
> Example: `// BORROW: explicit .as_str() instead of Deref coercion`

### `// SAFETY: <invariants>`
Required on every `unsafe` block or function (inline comment, not a doc comment).
Not expected in candle-mi (`#![forbid(unsafe_code)]`); included for completeness.

## Shape Documentation Format (Rule 12)

All public functions that accept or return `Tensor` must document shapes in
their doc comment using the following format:

    /// # Shapes
    /// - `q`: `[batch, n_heads, seq_q, head_dim]` -- query tensor
    /// - `k`: `[batch, n_kv_heads, seq_k, head_dim]` -- key tensor
    /// - returns: `[batch, n_heads, seq_q, head_dim]`

Rules:
- Use concrete dimension names, never `d0`/`d1`.
- Batch dimension is always first.
- Document every tensor argument and the return tensor.

## #[non_exhaustive] Policy (Rule 11)

- Public enums that may gain new variants: `#[non_exhaustive]`.
- Internal dispatch enums matched exhaustively by this crate:
  `#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: <reason>`.

## Hook Purity Contract (Rule 16)

- `HookSpec::capture()` takes only a hook point name -- no callback. The
  captured tensor is stored in `HookCache` and retrieved after `forward()`.
  The absence of a mutation mechanism is the enforcement.
- `HookSpec::intervene()` takes a typed `Intervention` value. All mutations
  go through this path and are visible at the call site.

## `#[must_use]` Policy (Rule 17)

All public functions and methods that return a value and have no side effects
must be annotated `#[must_use]`.  This includes constructors (`new`,
`with_capacity`), accessors (`len`, `is_empty`, `get_*`), and pure queries.
Without the annotation, a caller can silently discard the return value — which
for these functions is always a bug, since the call has no other effect.

The `clippy::must_use_candidate` lint enforces this at `warn` level
(promoted to error by `#![deny(warnings)]`).

## `# Errors` Doc Section

All public fallible methods (`-> Result<T>`) must include an `# Errors` section
in their doc comment. Each bullet uses the format:

    /// # Errors
    /// Returns [`MIError::Config`] if the model type is unsupported.
    /// Returns [`MIError::Model`] on weight loading failure.

Rules:
- Start each bullet with `Returns` followed by the variant in rustdoc link
  syntax, e.g., `` [`MIError::Config`] ``.
- Follow with `if` (condition), `on` (event), or `when` (circumstance).
- Use the concrete variant name, not the generic `MIError`.
- One bullet per distinct error path.

## Error Message Wording

Error strings passed to `MIError` variants follow two patterns:

- **External failures** (I/O, serde, network): `"failed to <verb>: {e}"`
  > Example: `MIError::Config(format!("failed to parse config: {e}"))`
- **Validation failures** (range, shape, lookup): `"<noun> <problem> (<context>)"`
  > Example: `MIError::Config(format!("source_layer {src} out of range (max {max})"))`
  > Example: `MIError::Hook(format!("hook point {point:?} not captured"))`

Rules:
- Use lowercase, no trailing period.
- Include the offending value and the valid range or constraint when applicable.
- Wrap external errors with `: {e}`, not `.to_string()`.

## `# Memory` Doc Section

Public methods that load large files (>100 MB, typically safetensors decoder
files) must include a `# Memory` section documenting:

1. **Peak allocation** — how much memory the method allocates at its peak.
2. **Residency** — whether the large allocation lives on CPU, GPU, or both.
3. **Lifetime** — whether the allocation is dropped before the method returns
   or persists in the returned value.

Format:

    /// # Memory
    /// Loads one decoder file (~2 GB) to CPU per source layer. Each file is
    /// dropped before loading the next. Peak: ~2 GB CPU.

## OOM-safe Decoder Loading Pattern

When loading large safetensors files (decoder weights, encoder weights),
follow the 7-step pattern to bound peak memory:

1. `ensure_path()` — resolve the file path (may trigger download).
2. `fs::read(&path)` — read the entire file into a `Vec<u8>` on CPU.
3. `SafeTensors::deserialize(&bytes)` — zero-copy parse of the byte buffer.
4. Extract the tensor view and build a candle `Tensor` on CPU.
5. Slice or narrow to the needed subset.
6. `drop(bytes)` (or let it go out of scope) — free the raw file buffer
   **before** loading the next file.
7. Optionally `.to_device(device)` if GPU computation follows.

The key invariant is: **at most one large file buffer is alive at any time**.
This bounds peak memory to roughly 1× the largest decoder file (~2 GB for
Gemma 2 2B CLTs).

> Example location: `CrossLayerTranscoder::score_features_by_decoder_projection`

## HashMap Grouping Idiom

When operations must be batched by a key (e.g., grouping features by source
layer to load each decoder file only once), use the `Entry` API:

```rust
let mut by_source: HashMap<usize, Vec<Item>> = HashMap::new();
for item in items {
    by_source.entry(item.key()).or_default().push(item);
}
```

Rules:
- Name the map `by_<grouping_key>` (e.g., `by_source`, `by_layer`).
- Use `.entry(key).or_default().push()` — never `if let Some` + `else insert`.
- Iterate the map to perform the batched operation (one file load per key).
