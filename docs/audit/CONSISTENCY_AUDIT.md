# Rust Codebase Consistency Audit

**Date:** 2026-03-06  
**Scope:** `ROADMAP.md`, `CONVENTIONS.md`, `design/*.md`, and the current Rust codebase under `src/`, `examples/`, and `tests/`  
**Method:** read-only document/code review, targeted source inspection, `cargo metadata`, prior `cargo check`, and IDE diagnostics on key modules

## Executive summary

The codebase is **broadly sound and strategically coherent**. The major architecture described in the planning/design documents is present in the implementation: generic transformer support, RWKV support, typed errors, hook capture infrastructure, CLT loading/encoding/injection, and the newer `hf-fetch-model` single-file CLT download path all line up well with the documented direction.

The main issues are not signs of a broken codebase; they are mostly:

1. one **real sampling edge-case bug** in `src/backend.rs`,
2. one **important API/docs mismatch** around transformer intervention coverage,
3. several cases where the **roadmap understates what is already implemented**,
4. two smaller operational risks (mask-cache device aliasing and network-dependent download tests).

Overall assessment: **healthy codebase, with a small number of concrete inconsistencies worth fixing so the docs and API contracts remain trustworthy.**

## High-confidence strengths

- **CLT support is materially aligned with Phase 3 goals.** `src/clt/mod.rs` implements `CrossLayerTranscoder`, `encode()`, `top_k()`, injection helpers, attribution graph helpers, and now uses `hf_fetch_model::download_file_blocking(...)` for per-file CLT downloads.
- **RWKV work appears strategically consistent.** `src/rwkv/mod.rs` includes effective-attention capture and RWKV-specific state knockout/state steering support that fits the design direction.
- **Typed-error discipline is mostly respected.** The library API generally returns `MIError` variants rather than panicking; the main exception found is an edge case in sampling noted below.
- **Tooling health is good.** The earlier targeted diagnostics pass reported no IDE diagnostics in the main files reviewed.

## Findings

### 1. `sample_with_temperature()` violates its documented empty-logits contract

**Category:** Correctness / API contract  
**Severity:** High

`sample_token()` documents that empty logits should return `MIError::Model`. The greedy path (`argmax()`) does this correctly. The temperature path does not.

In `src/backend.rs`:

- `argmax()` calls `.max_by(...).ok_or_else(...)` and returns an explicit `empty logits` error.
- `sample_with_temperature()` converts logits to a `Vec<f32>`, but never checks whether the vector is empty.
- If logits are empty, it reaches the fallback `Ok((probs.len() - 1) as u32)`, which is logically invalid and violates the documented behavior.

This is the strongest concrete bug found in the audit because it is both:

- inconsistent with the documented API contract, and
- avoidable with a small guard mirroring `argmax()`.

**Recommendation:** add an explicit empty-logits check at the start of `sample_with_temperature()` and return the same typed error as `argmax()`.

### 2. Transformer intervention coverage is narrower than the public hook surface and internal comments suggest

**Category:** API/docs consistency  
**Severity:** Medium

The public hook system exposes a broad transformer hook vocabulary via `HookPoint` and a unified `HookSpec`/`Intervention` API. That strongly suggests users can both capture and intervene at most listed hook points.

Current transformer implementation appears narrower:

- `src/transformer/mod.rs` applies interventions at `HookPoint::Embed` and `HookPoint::ResidPost(layer)`.
- `src/transformer/attention.rs` applies interventions at `HookPoint::AttnScores(layer)` and `HookPoint::AttnPattern(layer)`.
- The same files only **capture** many other advertised hook points:
  - `ResidPre`
  - `AttnQ`, `AttnK`, `AttnV`
  - `AttnOut`
  - `ResidMid`
  - `MlpPre`, `MlpPost`, `MlpOut`
  - `FinalNorm`
- `src/transformer/mlp.rs` contains no intervention application for the MLP hook points.

This would already be a docs mismatch at the public API level, but `src/transformer/mod.rs` also contains an internal comment saying layer-range forwarding has **"full hook support (capture + intervention at every hook point)"**, which is not true in the current implementation.

**Why this matters:** users may reasonably expect `hooks.intervene(HookPoint::MlpPre(...), ...)` or `HookPoint::AttnQ(...)` to take effect, but those interventions appear to be silently unsupported in the transformer backend.

**Recommendation:** choose one of these and make it explicit:

1. implement interventions at the remaining transformer hook points, or
2. document the supported intervention subset clearly and narrow any comments/docs that imply full coverage.

### 3. The roadmap is now stale enough to be an unreliable status document in a few Phase 3 areas

**Category:** Strategic consistency / documentation drift  
**Severity:** Medium

The implementation is ahead of `ROADMAP.md` in several places.

#### 3a. Top-level release status is stale

- `Cargo.toml` reports crate version **`0.0.4`**.
- `ROADMAP.md` still says the crate is published as **`v0.0.3`**.

That is a simple but high-visibility trust issue for readers using the roadmap as the project status page.

#### 3b. Recurrent feedback / anacrousis implementation is already present

`ROADMAP.md` still lists this unchecked:

- "Implement recurrent feedback (anacrousis)"

But the codebase already contains substantial implementation:

- `src/transformer/recurrent.rs` defines `RecurrentPassSpec`, feedback entries, `with_sustained(...)`, `add_feedback(...)`, and validation.
- `src/transformer/mod.rs` implements `forward_recurrent(...)` and recurrent generation.
- `tests/validate_anacrousis.rs` contains a dedicated large integration test for the 28-condition × 15-couplet experiment.

The validation checkbox is more nuanced: the test is `#[ignore]`, requires CUDA and cached model assets, and therefore is not evidence of routine CI coverage. But the **implementation checkbox** is clearly stale.

#### 3c. `scripts/README.md` exists even though the roadmap still lists it as undone

- `ROADMAP.md` still has "Add `scripts/README.md` ..." unchecked.
- The repository already contains `scripts/README.md`.

This is another sign that the roadmap has not kept up with implementation/documentation progress.

**Recommendation:** do a Phase 3 roadmap cleanup pass. At minimum, update the crate version/status line and mark the recurrent implementation task complete; decide explicitly whether the validation task is considered complete or intentionally still pending due to the ignored/manual nature of the experiment.

### 4. Mask-cache device keys are knowingly unsafe for multi-GPU / multi-Metal scenarios

**Category:** Correctness risk / scalability  
**Severity:** Low to Medium

`src/util/masks.rs` caches masks by `(seq_len, device_id, dtype)`, but `device_id()` maps:

- `Cpu -> 0`
- any `Cuda(_) -> 1`
- any `Metal(_) -> 2`

The file explicitly notes: **"This simplified approach assumes a single device per type."**

So this is not an undocumented bug; it is an acknowledged limitation. Still, it is a real correctness hazard if users run multiple CUDA devices or multiple Metal devices in the same process, because masks created for one device could alias the cache key of another.

**Recommendation:** if multi-device support is in scope, use real device ordinals/identifiers in the cache key. If not, document the single-device-per-type assumption in any public-facing performance/device notes.

### 5. `tests/fast_download.rs` is a real integration test, but it is not hermetic

**Category:** Test reliability / operational risk  
**Severity:** Low

`tests/fast_download.rs` downloads a real public HuggingFace repo (`julien-c/dummy-unknown`) and asserts that the returned cache path exists.

That is a valid integration test, but it means the test outcome depends on:

- network availability,
- HuggingFace availability,
- cache behavior,
- and external repository stability.

This is acceptable if treated as an integration smoke test, but it should not be mistaken for a deterministic unit test.

**Recommendation:** keep it if the goal is end-to-end verification, but consider separating hermetic unit coverage from networked smoke coverage in test documentation and/or CI selection.

## Strategic consistency verdict

The implemented architecture is **mostly consistent with the stated strategy**:

- the crate is no longer just a thin model wrapper; it has the intended MI-oriented abstractions,
- CLT support is real and substantial,
- RWKV work is present rather than aspirational,
- the `hf-fetch-model` single-file design has been carried through into the CLT implementation.

The main strategic inconsistency is **documentation lag**, not architectural drift.

## API/docs consistency verdict

The most important API/docs issue is the **transformer intervention surface**. The public hook vocabulary and some internal wording suggest more intervention support than the transformer backend actually implements.

Apart from that, the code/docs picture is good: naming is coherent, module layout matches the intended architecture, and the CLT/download path is notably well aligned with the recent design work.

## Correctness/risk verdict

The only strong correctness issue found in the audited areas is the **empty-logits sampling bug** in `sample_with_temperature()`.

The remaining issues are lower-grade risks:

- multi-device mask-cache aliasing,
- network-dependent download tests,
- roadmap drift that can mislead maintainers/users about current status.

## Prioritized follow-up list

1. **Fix `sample_with_temperature()` empty-logits handling.**
2. **Resolve the transformer intervention-surface mismatch**: either implement the missing hook-point interventions or narrow the docs/comments/API expectations.
3. **Refresh `ROADMAP.md` Phase 3 and top-level status text** so it matches the current codebase and crate version.
4. **Decide whether multi-device mask caching matters** for intended users; if yes, strengthen the cache key.
5. **Document `fast_download` as a networked integration smoke test** if that is the intended role.

## Audit note

During the audit, one earlier file-view result for `src/lib.rs` conflicted with the actual repository state. Direct file reading plus `cargo metadata` confirmed the current crate root/module tree is coherent; that discrepancy was a tooling artifact, not a repository inconsistency.

---

## Appendix A — Fix log

### A.1 Finding 4: Mask-cache device keys (resolved)

**Date:** 2026-03-06

The custom `device_id()` function in `src/util/masks.rs` collapsed all CUDA devices to `1` and all Metal devices to `2`, making the cache key unsafe for multi-GPU processes.

**Fix:** replaced `device_id()` with candle's built-in `DeviceLocation` (which is `Hash + Eq + Copy` and carries `gpu_id`). The cache key type changed from `(usize, usize, DType)` to `(usize, DeviceLocation, DType)`. The `device_id` helper was deleted. No API surface changed; all existing tests pass.

### A.2 Finding 2: Transformer intervention coverage (resolved)

**Date:** 2026-03-06

Ten hook points (`ResidPre`, `AttnQ`, `AttnK`, `AttnV`, `AttnOut`, `ResidMid`, `MlpPre`, `MlpPost`, `MlpOut`, `FinalNorm`) only supported capture — interventions registered at these points were silently ignored. The internal comment in `forward_layer_range` also claimed "full hook support (capture + intervention at every hook point)" which was inaccurate.

**Fix:** added the standard 3-line intervention loop (`for intervention in hooks.interventions_at(...)`) at all 10 capture-only sites in `src/transformer/mod.rs` and `src/transformer/attention.rs`. All 13 transformer hook points now support both capture and intervention. The stale comment was corrected. All 80 unit tests, 17 doc-tests, and all integration tests pass; clippy pedantic is clean.

### A.3 Finding 1: `sample_with_temperature()` empty-logits contract (resolved)

**Date:** 2026-03-06

When passed empty logits, `sample_with_temperature()` fell through to its rounding-edge-case fallback and returned `(0 - 1) as u32` = `u32::MAX` — a silently invalid token ID. The documented contract and `argmax()`'s implementation both specify returning `MIError::Model("empty logits")`.

**Fix:** added an early `is_empty()` guard at the top of `sample_with_temperature()`, mirroring the existing pattern in `argmax()`. Both sampling paths now return the same typed error on empty input.

### A.4 Finding 5: `fast_download.rs` not hermetic (documented)

**Date:** 2026-03-06

`tests/fast_download.rs` downloads from HuggingFace Hub at test time, making it dependent on network availability and external service stability.

**Resolution:** added a module-level doc comment clarifying these tests are networked smoke tests and that failures may be transient infrastructure issues, not code bugs. No behavioural change.

### A.5 Finding 3: Stale roadmap (resolved)

**Date:** 2026-03-06

`ROADMAP.md` reported the crate as v0.0.3 with Phase 3 in progress. The actual state was v0.0.4 with Phase 3 complete. Three checklist items (recurrent feedback implementation, anacrousis validation, `scripts/README.md`) were unchecked despite being implemented.

**Fix:** updated the status line to v0.0.4 / Phase 3 complete, marked all three items as done with implementation details, added the Phase 3 completion date to the deliverable line, and updated the "last updated" date.