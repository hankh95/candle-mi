// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation patching (causal tracing): identify which layers are causally
//! responsible for a model's factual prediction.
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer --example activation_patching -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models (no argument)
//! cargo run --release --features transformer,mmap --example activation_patching
//!
//! # With real memory reporting (RAM + VRAM)
//! cargo run --release --features transformer,memory --example activation_patching -- "meta-llama/Llama-3.2-1B"
//! ```
//!
//! **What it does:**
//!
//! 1. Runs a **clean** forward pass on "The capital of France is", capturing
//!    the residual stream at every layer via
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint) and building a
//!    [`FullActivationCache`](candle_mi::FullActivationCache).
//! 2. Runs a **corrupted** forward pass on "The capital of Poland is" (same
//!    structure, different country) and captures all residual streams.
//! 3. For each layer, runs a **patching** pass: the corrupted forward pass
//!    with the clean residual stream at the **subject token position** only
//!    restored via [`Intervention::Replace`](candle_mi::Intervention). This
//!    isolates the effect of the subject token's representation at each layer.
//! 4. Prints a layer-by-layer recovery table showing how much the "Paris"
//!    prediction recovers when clean information is injected at each layer.
//!
//! This is the standard "causal tracing" technique from:
//!
//! > Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.
//! > "Locating and Editing Factual Associations in GPT."
//! > *Advances in Neural Information Processing Systems* (NeurIPS), 2022.
//! > <https://arxiv.org/abs/2202.05262>
//!
//! Layers with the highest recovery are the causal site for factual recall.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use candle_mi::interp::intervention::kl_divergence;
use candle_mi::interp::logit_lens::format_probability;
use candle_mi::{
    FullActivationCache, HookPoint, HookSpec, Intervention, MIModel, MITokenizer,
    SUPPORTED_MODEL_TYPES,
};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Clean prompt and a list of candidate corrupted prompts.
///
/// The first candidate whose tokenization has the same length as the clean
/// prompt is used. This handles tokenizers that split country names differently.
const CLEAN_PROMPT: &str = "The capital of France is";
const CORRUPTED_CANDIDATES: &[&str] = &[
    "The capital of Poland is",
    "The capital of Brazil is",
    "The capital of Russia is",
    "The capital of Canada is",
    "The capital of Turkey is",
];

fn run() -> candle_mi::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // If a model ID is provided, run only that model
    if args.len() > 1 {
        // INDEX: args[1] is safe — checked len() > 1 above
        #[allow(clippy::indexing_slicing)]
        let model_id = &args[1];
        return run_single_model(model_id);
    }

    // Otherwise, discover and run all cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return Ok(());
    }

    println!(
        "Found {} supported transformer(s) in HF cache:\n",
        cached.len()
    );

    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single model (by ID)
// ---------------------------------------------------------------------------

/// Load a model by ID, run activation patching, and print results.
fn run_single_model(model_id: &str) -> candle_mi::Result<()> {
    println!("=== {model_id} ===");

    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!(
        "  Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {:?}",
        model.device()
    );
    println!("  Estimated F32 weight size: {weight_mb:.0} MB");
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    let corrupted_prompt = find_corrupted_prompt(tokenizer)?;
    run_patching(&model, tokenizer, CLEAN_PROMPT, &corrupted_prompt)
}

/// Pick the first corrupted prompt whose tokenization matches the clean
/// prompt's token count.
fn find_corrupted_prompt(tokenizer: &MITokenizer) -> candle_mi::Result<String> {
    let clean_len = tokenizer.encode(CLEAN_PROMPT)?.len();
    for &candidate in CORRUPTED_CANDIDATES {
        if tokenizer.encode(candidate)?.len() == clean_len {
            return Ok(candidate.into());
        }
    }
    Err(candle_mi::MIError::Tokenizer(
        "no corrupted prompt candidate matches clean prompt token count".into(),
    ))
}

// ---------------------------------------------------------------------------
// Cache discovery (same pattern as other examples)
// ---------------------------------------------------------------------------

/// Return the `HuggingFace` Hub cache directory.
fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(cache).join("hub"));
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Find the first snapshot directory for a cached model.
fn find_snapshot(cache_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = cache_dir.join(dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Read `model_type` from a cached `config.json`.
fn read_model_type(snapshot: &Path) -> Option<String> {
    let config_path = snapshot.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    // BORROW: explicit .as_str() — serde_json::Value → &str
    json.get("model_type")?.as_str().map(String::from)
}

/// Scan the HF cache and return `(model_id, model_type, snapshot_path)`.
fn discover_cached_models() -> Vec<(String, String, PathBuf)> {
    let Some(cache_dir) = hf_cache_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return Vec::new();
    };

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(dir_name) = name.to_str() else {
            continue;
        };
        let Some(rest) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let model_id = rest.replacen("--", "/", 1);
        let Some(snapshot) = find_snapshot(&cache_dir, &model_id) else {
            continue;
        };
        let Some(model_type) = read_model_type(&snapshot) else {
            continue;
        };
        // BORROW: explicit .as_str() — String → &str for slice lookup
        if SUPPORTED_MODEL_TYPES.contains(&model_type.as_str()) {
            models.push((model_id, model_type, snapshot));
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

// ---------------------------------------------------------------------------
// Per-model patching experiment (cache discovery mode)
// ---------------------------------------------------------------------------

/// Load a model from a snapshot, run patching, and print the analysis.
fn run_model(model_id: &str, snapshot: &Path) -> candle_mi::Result<()> {
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!(
        "  {} layers, {} heads, {} hidden, device: {:?}",
        n_layers,
        n_heads,
        hidden,
        model.device()
    );
    println!("  Estimated F32 weight size: {weight_mb:.0} MB  |  Load: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Memory");
    }

    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    let corrupted_prompt = find_corrupted_prompt(&tokenizer)?;
    run_patching(&model, &tokenizer, CLEAN_PROMPT, &corrupted_prompt)
}

// ---------------------------------------------------------------------------
// Core activation patching logic
// ---------------------------------------------------------------------------

/// Run clean, corrupted, and patching passes, then print the recovery table.
fn run_patching(
    model: &MIModel,
    tokenizer: &MITokenizer,
    clean_prompt: &str,
    corrupted_prompt: &str,
) -> candle_mi::Result<()> {
    let n_layers = model.num_layers();
    let hidden = model.hidden_size();

    // Encode both prompts
    let clean_ids = tokenizer.encode(clean_prompt)?;
    let corrupted_ids = tokenizer.encode(corrupted_prompt)?;
    let seq_len = clean_ids.len();

    if clean_ids.len() != corrupted_ids.len() {
        return Err(candle_mi::MIError::Config(format!(
            "clean ({}) and corrupted ({}) prompts have different token counts",
            clean_ids.len(),
            corrupted_ids.len(),
        )));
    }

    // Find the subject token position (first token that differs)
    let subject_pos = clean_ids
        .iter()
        .zip(corrupted_ids.iter())
        .position(|(a, b)| a != b)
        .ok_or_else(|| {
            candle_mi::MIError::Config("clean and corrupted prompts are identical".into())
        })?;

    // INDEX: subject_pos is a valid index — it came from .position() over
    // clean_ids/corrupted_ids which have length seq_len (checked equal above)
    #[allow(clippy::indexing_slicing)]
    let clean_subject_id = clean_ids[subject_pos];
    #[allow(clippy::indexing_slicing)]
    let corrupted_subject_id = corrupted_ids[subject_pos];
    let clean_subject = tokenizer.decode(&[clean_subject_id])?;
    let corrupted_subject = tokenizer.decode(&[corrupted_subject_id])?;
    println!("  Clean: \"{clean_prompt}\" ({seq_len} tokens)");
    println!("  Corrupted: \"{corrupted_prompt}\"");
    println!("  Subject position: {subject_pos} (\"{clean_subject}\" → \"{corrupted_subject}\")");

    let clean_input = candle_core::Tensor::new(&clean_ids[..], model.device())?.unsqueeze(0)?;
    let corrupted_input =
        candle_core::Tensor::new(&corrupted_ids[..], model.device())?.unsqueeze(0)?;

    // ── Step 1: Clean forward pass ──────────────────────────────────────
    let mut capture_hooks = HookSpec::new();
    for layer in 0..n_layers {
        capture_hooks.capture(HookPoint::ResidPost(layer));
    }

    let t1 = Instant::now();
    let clean_cache = model.forward(&clean_input, &capture_hooks)?;
    let clean_time = t1.elapsed();
    let clean_logits = clean_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]
    println!("  Clean forward ({n_layers} captures): {clean_time:.2?}");

    // Build FullActivationCache from clean captures
    let mut clean_acts = FullActivationCache::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = clean_cache.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]
        clean_acts.push(resid.get(0)?); // [seq, hidden]
    }

    // ── Step 2: Corrupted forward pass ──────────────────────────────────
    let t2 = Instant::now();
    let corrupted_cache = model.forward(&corrupted_input, &capture_hooks)?;
    let corrupted_time = t2.elapsed();
    let corrupted_logits = corrupted_cache.output().get(0)?.get(seq_len - 1)?;
    println!("  Corrupted forward ({n_layers} captures): {corrupted_time:.2?}");

    // Build FullActivationCache from corrupted captures
    let mut corrupted_acts = FullActivationCache::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = corrupted_cache.require(&HookPoint::ResidPost(layer))?;
        corrupted_acts.push(resid.get(0)?);
    }

    // Baseline metrics
    let corrupted_kl = kl_divergence(&clean_logits, &corrupted_logits)?;
    println!("  KL(clean || corrupted): {corrupted_kl:.6}");

    // Find "Paris" token
    let paris_id = tokenizer.encode(" Paris")?.into_iter().last();
    if let Some(pid) = paris_id {
        // PROMOTE: softmax requires F32 for numerical stability
        let clean_p = extract_prob(
            &candle_nn::ops::softmax_last_dim(&clean_logits.to_dtype(candle_core::DType::F32)?)?,
            pid,
        )?;
        let corrupted_p = extract_prob(
            &candle_nn::ops::softmax_last_dim(
                &corrupted_logits.to_dtype(candle_core::DType::F32)?,
            )?,
            pid,
        )?;
        println!(
            "  P(Paris) clean: {}  |  corrupted: {}",
            format_probability(clean_p),
            format_probability(corrupted_p),
        );
    }

    // ── Step 3: Patching sweep ──────────────────────────────────────────
    // For each layer, run the corrupted forward pass but restore the clean
    // residual at the subject token position only.
    println!(
        "\n  {:>5}  {:>10}  {:>10}  {:>12}",
        "Layer", "KL", "Recovery", "P(Paris)"
    );
    println!("  {:->5}  {:->10}  {:->10}  {:->12}", "", "", "", "");

    let t3 = Instant::now();
    let mut best_layer = 0;
    let mut best_recovery = f32::NEG_INFINITY;

    for layer in 0..n_layers {
        // Build a mixed tensor: corrupted at all positions, clean at subject_pos
        let corrupted_resid = corrupted_acts
            .get_layer(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("layer {layer} not in cache")))?;
        let clean_resid = clean_acts
            .get_layer(layer)
            .ok_or_else(|| candle_mi::MIError::Hook(format!("layer {layer} not in cache")))?;

        // Construct patched tensor: corrupted everywhere, clean at subject_pos
        // patched[pos] = corrupted[pos] for pos != subject_pos
        // patched[subject_pos] = clean[subject_pos]
        let patched_resid = patch_position(
            corrupted_resid,
            clean_resid,
            subject_pos,
            seq_len,
            hidden,
            model.device(),
        )?
        .unsqueeze(0)?; // [1, seq, hidden]

        let mut patch_hooks = HookSpec::new();
        patch_hooks.intervene(
            HookPoint::ResidPost(layer),
            Intervention::Replace(patched_resid),
        );

        let patched_cache = model.forward(&corrupted_input, &patch_hooks)?;
        let patched_logits = patched_cache.output().get(0)?.get(seq_len - 1)?;

        let patched_kl = kl_divergence(&clean_logits, &patched_logits)?;

        // Recovery = 1 - (patched_kl / corrupted_kl), as percentage
        let recovery = if corrupted_kl > 1e-10 {
            (1.0 - patched_kl / corrupted_kl) * 100.0
        } else {
            100.0
        };

        if recovery > best_recovery {
            best_recovery = recovery;
            best_layer = layer;
        }

        let paris_str = if let Some(pid) = paris_id {
            // PROMOTE: softmax requires F32 for numerical stability
            let patched_probs = candle_nn::ops::softmax_last_dim(
                &patched_logits.to_dtype(candle_core::DType::F32)?,
            )?;
            format_probability(extract_prob(&patched_probs, pid)?)
        } else {
            String::from("--")
        };

        println!("  {layer:>5}  {patched_kl:>10.6}  {recovery:>9.1}%  {paris_str:>12}");
    }
    let patch_time = t3.elapsed();

    println!("\n  Best recovery: layer {best_layer} ({best_recovery:.1}%)");
    println!("  Patching sweep ({n_layers} passes): {patch_time:.2?}");
    println!();

    Ok(())
}

/// Build a tensor that is `base` everywhere except at `patch_pos` where it
/// takes values from `patch_source`.
///
/// Both tensors have shape `[seq_len, hidden]`. The result has the same shape.
fn patch_position(
    base: &candle_core::Tensor,
    patch_source: &candle_core::Tensor,
    patch_pos: usize,
    seq_len: usize,
    hidden: usize,
    device: &candle_core::Device,
) -> candle_mi::Result<candle_core::Tensor> {
    // Build a binary mask: 0 everywhere, 1 at patch_pos
    let mut mask_data = vec![0.0_f32; seq_len * hidden];
    for i in 0..hidden {
        // INDEX: patch_pos * hidden + i bounded by seq_len * hidden
        #[allow(clippy::indexing_slicing)]
        {
            mask_data[patch_pos * hidden + i] = 1.0;
        }
    }
    let mask = candle_core::Tensor::from_vec(mask_data, (seq_len, hidden), device)?;

    // patched = base * (1 - mask) + patch_source * mask
    let one_minus_mask = (1.0 - &mask)?;
    let result = (base * &one_minus_mask)? + (patch_source * &mask)?;
    Ok(result?)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the probability of a specific token from a probability tensor.
fn extract_prob(probs: &candle_core::Tensor, token_id: u32) -> candle_mi::Result<f32> {
    let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    // CAST: u32 → usize, token ID is a valid index
    #[allow(clippy::as_conversions)]
    let idx = token_id as usize;
    Ok(probs_vec.get(idx).copied().unwrap_or(0.0))
}

/// Rough estimate of F32 weight memory in MB.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    let params_per_layer = 12.0 * (hidden as f64) * (hidden as f64);
    let total_params = (n_layers as f64) * params_per_layer;
    total_params * 4.0 / 1_000_000.0
}
