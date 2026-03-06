#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Validate SAE encoding against Python/PyTorch reference.

Loads Gemma 2 2B, runs a forward pass capturing residual stream activations,
then encodes through a Gemma Scope SAE (loaded directly from NPZ, no SAELens)
and saves reference outputs for comparison with candle-mi's Rust implementation.

The SAE weights are loaded from the same HuggingFace repo and path that
candle-mi uses: google/gemma-scope-2b-pt-res, layer_0/width_16k/average_l0_105/params.npz.

Requires: pip install torch transformers huggingface_hub numpy

Usage:
    python scripts/sae_validation.py

Outputs:
    scripts/sae_reference.json
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
PROMPT = "The capital of France is"
HOOK_LAYER = 0  # Layer for SAE encoding
TOP_K = 10  # Number of top features to report

# Must match the Rust constants in tests/validate_sae.rs:
#   SAE_REPO = "google/gemma-scope-2b-pt-res"
#   SAE_NPZ  = "layer_0/width_16k/average_l0_105/params.npz"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_NPZ_PATH = "layer_0/width_16k/average_l0_105/params.npz"


def print_environment():
    """Print version info for reproducibility."""
    print("=== Environment ===")
    print(f"Python:       {sys.version}")
    print(f"Platform:     {platform.platform()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"transformers: {transformers.__version__}")
    print(f"numpy:        {np.__version__}")
    print()


def load_sae_from_npz(device):
    """Download and load SAE weights from NPZ via huggingface_hub.

    Returns dict of PyTorch tensors on device.
    """
    print(f"Downloading SAE from {SAE_REPO} / {SAE_NPZ_PATH} ...")

    npz_path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_NPZ_PATH)
    print(f"  NPZ path: {npz_path}")

    data = np.load(npz_path)
    print(f"  Arrays: {data.files}")

    weights = {}
    for key in data.files:
        arr = data[key]
        tensor = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float32)
        weights[key] = tensor
        print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    return weights


def sae_encode(x, weights):
    """Manual SAE encode: JumpReLU (threshold present) or ReLU.

    Matches candle-mi's SparseAutoencoder::encode() exactly.
    No apply_b_dec_to_input (Gemma Scope NPZ does not use it).

    Args:
        x: [batch, seq, d_in] activations (F32)
        weights: dict of SAE weight tensors

    Returns:
        [batch, seq, d_sae] encoded features
    """
    x_f32 = x.float()

    # pre_acts = x @ W_enc + b_enc
    # W_enc: [d_in, d_sae]
    pre_acts = x_f32 @ weights["W_enc"] + weights["b_enc"]

    # JumpReLU if threshold present, else ReLU
    if "threshold" in weights:
        mask = (pre_acts > weights["threshold"]).float()
        return pre_acts * mask
    else:
        return torch.relu(pre_acts)


def sae_decode(features, weights):
    """Manual SAE decode: x_hat = features @ W_dec + b_dec.

    Args:
        features: [batch, seq, d_sae]
        weights: dict of SAE weight tensors

    Returns:
        [batch, seq, d_in] reconstructed activations
    """
    return features @ weights["W_dec"] + weights["b_dec"]


def main():
    print_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Match candle-mi F32

    # --- Load model ---
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )
    model.eval()

    # --- Tokenize ---
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    tokens_safe = [t.encode("ascii", errors="replace").decode() for t in tokens]
    print(f"Tokens ({len(tokens)}): {tokens_safe}")

    # --- Forward pass with hidden state capture ---
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    # Hidden states: tuple of (n_layers + 1) tensors [batch, seq, d_model]
    # Index 0 = embedding, index i = output of layer i-1
    # resid_post at layer L = hidden_states[L + 1]
    hidden_states = outputs.hidden_states
    resid_post = hidden_states[HOOK_LAYER + 1]  # [1, seq, d_model]
    d_model = resid_post.shape[-1]
    print(f"resid_post shape: {resid_post.shape}")
    print(f"resid_post dtype: {resid_post.dtype}")
    print(f"d_model: {d_model}")

    # --- Load SAE from NPZ (no SAELens) ---
    weights = load_sae_from_npz(device)

    d_in = weights["W_enc"].shape[0]
    d_sae = weights["W_enc"].shape[1]
    print(f"\nSAE d_in={d_in}, d_sae={d_sae}")
    has_threshold = "threshold" in weights
    arch = "JumpReLU" if has_threshold else "ReLU"
    print(f"SAE architecture: {arch}")

    # Verify dimensions match
    assert d_in == d_model, (
        f"SAE d_in ({d_in}) != model d_model ({d_model}). "
        f"Wrong SAE for this model?"
    )

    # --- Encode activations ---
    with torch.no_grad():
        encoded = sae_encode(resid_post.float(), weights)  # [1, seq, d_sae]
        decoded = sae_decode(encoded, weights)  # [1, seq, d_in]

    # --- Compute metrics ---
    mse = torch.mean((resid_post.float() - decoded) ** 2).item()
    print(f"\nReconstruction MSE: {mse:.6f}")

    # --- Per-position top features (last token) ---
    last_pos = encoded.shape[1] - 1
    last_encoded = encoded[0, last_pos]  # [d_sae]
    nonzero_mask = last_encoded > 0
    n_active = nonzero_mask.sum().item()
    print(f"Active features at last position: {n_active}")

    # Top-k features
    top_vals, top_idxs = torch.topk(last_encoded, min(TOP_K, int(n_active)))
    top_features = [
        {"index": int(idx), "value": float(val)}
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
    ]
    print(f"Top-{TOP_K} features: {top_features}")

    # --- Save reference ---
    resid_last = resid_post[0, last_pos].cpu().tolist()
    encoded_last = encoded[0, last_pos].cpu().tolist()
    decoded_last = decoded[0, last_pos].cpu().tolist()

    reference = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "hook_layer": HOOK_LAYER,
        "hook_name": f"blocks.{HOOK_LAYER}.hook_resid_post",
        "sae_repo": SAE_REPO,
        "sae_npz_path": SAE_NPZ_PATH,
        "d_in": d_in,
        "d_sae": d_sae,
        "architecture": arch,
        "tokens": tokens_safe,
        "n_tokens": len(tokens),
        "reconstruction_mse": mse,
        "n_active_last_pos": int(n_active),
        "top_features_last_pos": top_features,
        "resid_last_first10": resid_last[:10],
        "encoded_last_first10": encoded_last[:10],
        "decoded_last_first10": decoded_last[:10],
        "resid_last_norm": float(torch.norm(resid_post[0, last_pos]).item()),
        "encoded_last_norm": float(torch.norm(encoded[0, last_pos]).item()),
        "decoded_last_norm": float(torch.norm(decoded[0, last_pos]).item()),
    }

    out_path = Path(__file__).parent / "sae_reference.json"
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference to {out_path}")


if __name__ == "__main__":
    main()
