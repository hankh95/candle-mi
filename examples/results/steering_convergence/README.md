# Steering Convergence — Attractor Dynamics in LLM Residual Streams

## What is this?

This folder contains the results of a **steering convergence** experiment run
with [candle-mi](https://crates.io/crates/candle-mi). The experiment answers:

> **When we externally steer a model toward a target output via residual stream
> injection, does the model's internal activation pattern converge to what it
> naturally computes, or does it take a different internal path?**

This combines controllable generation with mechanistic interpretability —
systematic external control with full internal observation.

## Method

1. **Baseline run** — forward pass on "The capital of France is", capturing
   `ResidPost` at every layer.
2. **Contrastive run** — forward pass on "The capital of Germany is". Per-layer
   steering vectors are computed as `ResidPost_france − ResidPost_germany` at
   the last token position.
3. **Injection sweep** — for each layer L, the steering vector is injected via
   `Intervention::Add` at `ResidPost(L)` (last token only). All layers are
   captured in the steered forward pass.
4. **Convergence matrix** — cosine similarity between steered and natural
   activations at every (injection_layer, observation_layer) pair. Values
   below the diagonal are 1.0 (unaffected). Values on/above the diagonal
   reveal how quickly the model absorbs the perturbation.
5. **Absorption boundary** — the earliest layer after injection where cosine
   similarity exceeds a threshold (default 0.95).
6. **Strength sweep** — at the deepest absorbing layer, sweep strength from
   0.5× to 6.0× to find the critical threshold where absorption breaks.

## Results

### Llama 3.2 1B (16 layers)

| Metric | Value |
|--------|-------|
| Absorption rate | 81% of layers (13/16) |
| Avg absorption depth | 1.5 layers |
| Critical strength | ~1.2× contrastive distance |
| Boundary layer | 12 (absorption at layer 15) |
| Baseline P("Paris") | 39.3% |
| Pattern | **ATTRACTOR** |

The convergence matrix shows near-perfect similarity (>0.99) for early-layer
injections, with a sharp drop in the last 3 layers. The strength sweep at
layer 12 reveals a critical threshold between 1.0× and 1.5×: below it, the
model absorbs the perturbation within 3 layers; above it, convergence breaks
permanently.

![Llama 3.2 1B convergence matrix](plots/meta-llama_Llama-3.2-1B_convergence_matrix.png)

![Llama 3.2 1B P(target) by injection layer](plots/meta-llama_Llama-3.2-1B_p_target_by_layer.png)

![Llama 3.2 1B strength sweep](plots/meta-llama_Llama-3.2-1B_strength_sweep.png)

### Gemma 2 2B (26 layers)

| Metric | Value |
|--------|-------|
| Absorption rate | 92% of layers (24/26) |
| Avg absorption depth | 1.0 layers |
| Critical strength | ~1.2× contrastive distance |
| Boundary layer | 23 (absorption at layer 24) |
| Baseline P("Paris") | 3.9% (softcapping flattens distribution) |
| Pattern | **ATTRACTOR** |

Gemma shows an even stronger attractor — perturbations are absorbed within a
single layer. The critical strength threshold is identical (~1.2×), suggesting
this is a universal property of pre-norm transformer residual streams.

![Gemma 2 2B convergence matrix](plots/google_gemma-2-2b_convergence_matrix.png)

![Gemma 2 2B P(target) by injection layer](plots/google_gemma-2-2b_p_target_by_layer.png)

![Gemma 2 2B strength sweep](plots/google_gemma-2-2b_strength_sweep.png)

### Cross-model comparison

| Property | Llama 3.2 1B | Gemma 2 2B |
|----------|-------------|------------|
| Absorption rate | 81% | 92% |
| Avg depth | 1.5 layers | 1.0 layers |
| Critical strength | ~1.2× | ~1.2× |
| Boundary site (% depth) | 75% | 88% |
| Strength response | U-shaped | Monotonic decline |

**Key finding:** The attractor basin has a consistent radius (~1.2× the
contrastive direction) across architectures, scales, and normalization
strategies. This suggests that the residual stream's self-correcting behavior
is a structural property of pre-norm transformers, not a model-specific
artifact.

## Why does this matter?

1. **MI circuit validity.** If the model always converges to the same attractor,
   then circuits discovered via intervention (e.g., CLT suppress/inject) are
   likely the same circuits used during natural computation. The internal path
   is unique, not one of many degenerate solutions.

2. **Steering safety.** The sharp critical threshold means moderate steering
   is absorbed harmlessly, but exceeding ~1.2× the contrastive distance
   causes permanent divergence. This quantifies the "safe steering range."

3. **Attractor depth.** The ~1-2 layer absorption depth means the model's
   computation is highly local: each layer mostly corrects toward the
   attractor independently, rather than requiring multi-layer coordination.

## Rhyme planning: what we learnt from 20 rhyme groups

We ran the same convergence analysis on 20 rhyme groups from the
[plip-rs poetry corpus](https://github.com/PCfVW/plip-rs/tree/melometis/docs/planning-in-poems),
using couplets validated during the Figure 13 reverse-engineering work.

### The experiment

For each of 20 rhyme groups (-air, -ake, -all, ..., -ove, -ow, -own), we:
- Used a 4-line couplet where line 3 ends with a rhyme-setting word
- Swapped that word with a word from a different rhyme group (cross-group contrastive)
- Tracked P(rhyme_word) at the last token position

**Batch command:** `--batch-file batch_rhyme_groups.json` (20 experiments, ~30s total)

### Results: the model doesn't plan rhymes at the last token

| Metric | Factual (Paris) | Rhyme (20 groups avg) |
|--------|----------------|----------------------|
| Baseline P(target) | **39.3%** | **<0.5%** (highest: 0.41%) |
| Min cosine similarity | 0.90 | 0.97 |
| Absorption rate | 81% | 93% |
| Max KL divergence | 1.5 | 0.3 |

**None of the 20 rhyme groups showed meaningful P(target) at the last token.**
The model predicts semantically sensible completions ("light" 62%, "treasure" 54%,
"fly" 27%), not rhyme words. The contrastive steering vectors produce small
perturbations because the model's last-token residual stream barely distinguishes
between rhyme-setting words in different groups.

### Why this is a valuable negative result

This confirms what was discovered during the
[plip-rs Figure 13 replication](https://github.com/PCfVW/plip-rs/tree/melometis/docs/planning-in-poems/03-figure13-replication.md):
**rhyme planning is not a last-token phenomenon.** The model decides the rhyme
word at the *planning site* — an earlier position in the sequence (typically where
the rhyme-setting word appears in line 3) — not at the output position. CLT
features for rhyme planning fire at that specific earlier position and produce
massive probability spikes (up to 10^7 fold) there, while having zero effect
elsewhere.

Closing this door is important: **the steering_convergence tool as built
perfectly measures factual recall attractors (last-token computation), but
planning attractors require position-aware injection at the planning site.**

### Additional insight: within-group vs cross-group contrastive

We also tested swapping "about" → "around" (same rhyme group: -AW1-N-D). This
produced essentially zero perturbation (KL < 0.02, cosine > 0.993). The model's
residual stream clusters words in the same rhyme family so tightly that they're
nearly indistinguishable — a direct validation of the CLT rhyme group structure
discovered in plip-rs.

### Position sweep: `--inject-position` doesn't help either

We added `--inject-position` (explicit number or `auto` to find the first
differing token) and swept positions 20-27 on the "-out" couplet. Position 23
(the "about"/"ahead" token) showed the strongest signal — but even there, max
KL was only 0.015 and P(" out") barely moved (1.0-1.4%).

This rules out position as the issue. The contrastive residual stream approach
**fundamentally cannot capture rhyme planning** because:

1. **Factual recall** is encoded as a *direction* in the residual stream — the
   "France direction" points toward "Paris" at every layer. Steering along
   that direction directly boosts the target.
2. **Rhyme planning** is encoded as *CLT feature activations* — sparse,
   nonlinear decompositions that can't be captured by a simple vector
   subtraction between two prompts. This is exactly why CLTs exist and why
   Figure 13 needed them.

### Conclusion: two tools for two regimes

| Computation type | Measurement tool | Why |
|-----------------|-----------------|-----|
| **Factual recall** | Contrastive steering (this example) | Recall is a direction in residual stream space |
| **Rhyme planning** | CLT feature injection (Figure 13) | Planning is encoded in sparse feature activations |

These are complementary, not competing. Steering convergence cleanly measures
recall attractors (~1.2× critical strength, 1-2 layer absorption). Planning
attractors require CLT decoder vectors as steering directions — the natural
next experiment.

### CLT-based steering: the experiment and what it revealed

We implemented `--clt` mode in steering_convergence: load a CLT, extract per-layer
decoder vectors for a specific feature, and inject them at the planning site
(position 21, auto-detected). We tested with the "around" feature (L22:10243)
from the 426K CLT on Gemma 2 2B — the same feature that achieves 48.3% redirect
in Figure 13.

**Three iterations, same result: zero effect on the last token.**

| Attempt | Injection strategy | Max strength | P(" out") change |
|---------|-------------------|-------------|------------------|
| Single decoder (layer 25) | One vector, all layers | 6.0 | None |
| Per-layer decoders | Layer-matched vectors, one at a time | 100.0 | None |
| Multi-layer simultaneous | All downstream layers at once (like Figure 13) | 20.0 | None |

**Diagnostic (multi-layer, strength 1.0):**
```
Layer 22: pos 21 diff=0.000000 (norm=449.2), pos 27 diff=0.000000 (norm=465.9)
Layer 25: pos 21 diff=9.918104 (norm=635.4), pos 27 diff=0.159336 (norm=1080.0)
```

The injection IS working at the planning site (position 21): the residual stream
at layer 25, position 21 moves by 9.9 units (1.5% of its norm). But at the last
token position (27), the change is only 0.159 units — 0.015% of the residual
norm. Even at strength 20, this doesn't register.

### Why Figure 13 works but convergence doesn't

Figure 13 and steering_convergence measure different things:

- **Figure 13** measures P("around") at the **planning site position** — where
  the CLT injection directly modifies the residual stream. The 48.3% spike
  happens at position 21, not at the last token.
- **Steering convergence** measures the convergence matrix and P(target) at
  the **last token position** — where the model makes its output prediction.

The planning perturbation at position 21 doesn't propagate to position 27
within a single forward pass. The residual stream at position 27 is dominated
by its own local computation (norm=1080) and barely sees the 0.015% ripple
from position 21.

**This is the deep finding:** planning circuits operate through
**position-specific, attention-mediated routing** — not through residual stream
perturbation at the output position. The CLT feature changes what the model
"thinks" at the planning site, and later generation steps naturally produce
the rhyme word. But within a single forward pass, the last token's residual
stream is nearly immune to changes at earlier positions.

### Revised conclusion: three regimes, not two

| Computation type | Where it lives | How to measure |
|-----------------|---------------|---------------|
| **Factual recall** | Last-token residual stream | Contrastive steering convergence |
| **Rhyme planning (local)** | Planning-site residual stream | CLT feature injection at planning site (Figure 13) |
| **Rhyme planning (output)** | Attention routing from planning site to output | Multi-step generation or attention pattern analysis |

The convergence matrix cleanly measures the first regime. The second regime
is captured by Figure 13's position sweep. The third regime — how the planning
decision propagates from the planning site to the output through attention —
is the next frontier for investigation.

## Experiment setup

| Parameter | Value |
|-----------|-------|
| **candle-mi version** | v0.1.3 + unreleased commits |
| **Hardware** | NVIDIA RTX 5060 Ti (16 GB VRAM) |
| **Precision** | F32 |
| **Prompt** | "The capital of France is" |
| **Contrastive** | "The capital of Germany is" |
| **Threshold** | 0.95 (cosine similarity) |
| **Strength sweep** | 0.5 to 6.0 in 12 steps |

## Reproducing

```bash
# Generate JSON output
cargo run --features transformer,mmap,memory --release --example steering_convergence -- "meta-llama/Llama-3.2-1B" --output examples/results/steering_convergence/llama-3.2-1b.json
cargo run --features transformer,mmap,memory --release --example steering_convergence -- "google/gemma-2-2b" --output examples/results/steering_convergence/gemma-2-2b.json

# Batch: 20 rhyme groups (~30s on GPU)
cargo run --features transformer,mmap,memory --release --example steering_convergence -- "meta-llama/Llama-3.2-1B" --batch-file examples/results/steering_convergence/batch_rhyme_groups.json --output examples/results/steering_convergence/batch_llama

# Plot with Mathematica
# Open convergence_plot.wl and evaluate all — it auto-iterates over all JSON files.
# Plots are exported to the plots/ subfolder.
```

## Files

| File | Description |
|------|-------------|
| `llama-3.2-1b.json` | Full output for Llama 3.2 1B (factual recall) |
| `gemma-2-2b.json` | Full output for Gemma 2 2B (factual recall) |
| `batch_rhyme_groups.json` | Batch input: 20 rhyme groups from plip-rs corpus |
| `batch_llama/` | Per-group JSON output (20 files, one per rhyme group) |
| `convergence_plot.wl` | Mathematica plotting script (auto-iterates all JSON files) |
| `plots/` | Generated PNG plots (convergence matrix, P(target), strength sweep) |
| `README.md` | This file |
