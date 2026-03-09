// SPDX-License-Identifier: MIT OR Apache-2.0

//! Logit lens: track how the final prediction forms layer by layer.
//!
//! ```bash
//! cargo run --release --features transformer --example logit_lens
//! ```
//!
//! **What it does:**
//!
//! 1. Scans the local `HuggingFace` Hub cache for supported transformers.
//! 2. For each cached model, captures the residual stream
//!    ([`HookPoint::ResidPost`](candle_mi::HookPoint::ResidPost)) at every
//!    layer in a single forward pass.
//! 3. Projects each layer's hidden state through the unembedding matrix via
//!    [`MIModel::project_to_vocab`](candle_mi::MIModel::project_to_vocab),
//!    applies softmax, and collects the top-k predictions.
//! 4. Builds a [`LogitLensAnalysis`](candle_mi::LogitLensAnalysis) and
//!    prints both a summary and a detailed view, plus the first layer at
//!    which the expected answer token appears.
//!
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::interp::logit_lens::decode_predictions_with;
use candle_mi::{
    HookPoint, HookSpec, LogitLensAnalysis, LogitLensResult, MIModel, MITokenizer,
    SUPPORTED_MODEL_TYPES,
};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let prompt = "The capital of France is";
    let top_k: usize = 10;

    // 1. Discover cached models
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

    // 2. Run logit lens on each model
    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot, prompt, top_k) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cache discovery (same pattern as quick_start_transformer)
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
// Per-model logit lens
// ---------------------------------------------------------------------------

/// Load a model, run logit lens, and print the analysis.
fn run_model(model_id: &str, snapshot: &Path, prompt: &str, top_k: usize) -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained(model_id)?;
    let n_layers = model.num_layers();
    println!(
        "  {} layers, {} hidden, device: {:?}",
        n_layers,
        model.hidden_size(),
        model.device()
    );

    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    // Encode prompt
    let token_ids = tokenizer.encode(prompt)?;
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq]
    println!("  Prompt: \"{prompt}\" ({} tokens)", token_ids.len());

    // Capture ResidPost at every layer
    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidPost(layer));
    }

    // Single forward pass with all captures
    let cache = model.forward(&input, &hooks)?;

    // Build logit lens analysis
    let analysis = build_analysis(&model, &tokenizer, &cache, prompt, n_layers, top_k)?;

    // Print results
    analysis.print_summary();
    println!();
    analysis.print_detailed(top_k);
    println!();

    // Show convergence: when does the expected token first appear?
    if let Some(layer) = analysis.first_appearance("Paris", top_k) {
        println!("  \"Paris\" first appears in top-{top_k} at layer {layer}");
    } else {
        println!("  \"Paris\" never appears in top-{top_k}");
    }
    println!();

    Ok(())
}

// ---------------------------------------------------------------------------
// Analysis builder
// ---------------------------------------------------------------------------

/// Project each layer's residual stream to vocabulary and build a
/// [`LogitLensAnalysis`].
fn build_analysis(
    model: &MIModel,
    tokenizer: &MITokenizer,
    cache: &candle_mi::HookCache,
    prompt: &str,
    n_layers: usize,
    top_k: usize,
) -> candle_mi::Result<LogitLensAnalysis> {
    let mut analysis = LogitLensAnalysis::new(prompt.into(), n_layers);

    for layer in 0..n_layers {
        let resid = cache.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]

        // Extract last position: [1, hidden]
        let seq_len = resid.dim(1)?;
        let last_hidden = resid.get(0)?.get(seq_len - 1)?.unsqueeze(0)?; // [1, hidden]

        // Project to vocabulary: [1, vocab]
        let logits = model.project_to_vocab(&last_hidden)?;

        // PROMOTE: softmax requires F32 for numerical stability
        let logits_f32 = logits.to_dtype(candle_core::DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?; // [1, vocab]
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        // Top-k by probability
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_pairs: Vec<(u32, f32)> = indexed
            .iter()
            .take(top_k)
            .map(|&(idx, prob)| {
                // CAST: usize → u32, vocab index fits in u32
                #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
                let id = idx as u32;
                (id, prob)
            })
            .collect();

        let predictions = decode_predictions_with(&top_pairs, |id| {
            tokenizer
                .decode(&[id])
                .unwrap_or_else(|_| format!("[{id}]"))
        });

        analysis.push(LogitLensResult { layer, predictions });
    }

    Ok(analysis)
}
