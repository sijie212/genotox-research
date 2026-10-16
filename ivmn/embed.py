#!/usr/bin/env python3
"""Hugging Face molecular embedding extraction (frozen ChemBERTa).

Loads a pretrained SMILES transformer from Hugging Face (default
DeepChem/ChemBERTa-77M-MLM) and writes one fixed-length embedding per
compound: mean-pooled last-hidden-state over non-padding tokens.

Proxy note: huggingface_hub's HTTP client crashes on NO_PROXY/no_proxy
values containing bracketed IPv6 entries (e.g. "[::1]") with
"Invalid port: ':1]'". ``_sanitize_proxy_env`` strips those entries
before transformers is imported. This matters on machines behind a
corporate egress proxy (the reason this helper exists).

The embedding model itself is NOT committed: it downloads from Hugging
Face on first run and is cached locally (~/.cache/huggingface) for
offline reuse afterwards.

Usage:
    python embed.py --input data/mnvit_baderna.csv --smiles-col smiles \
        --out data/embeddings.npz
"""
import os
import re
import sys


def _sanitize_proxy_env():
    """Drop bracketed IPv6 entries from NO_PROXY/no_proxy.

    huggingface_hub (httpx) parses NO_PROXY entries as URL patterns and
    raises InvalidURL on entries like "[::1]". Removing only the bracketed
    entries keeps proxy behavior for normal hosts intact.
    """
    for key in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(key)
        if not val:
            continue
        parts = [p.strip() for p in val.split(",")]
        cleaned = [p for p in parts if "[" not in p]
        os.environ[key] = ",".join(cleaned)


_sanitize_proxy_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

DEFAULT_MODEL = "DeepChem/ChemBERTa-77M-MLM"


def load_embedding_model(model_name=DEFAULT_MODEL):
    """Return (tokenizer, model) on CPU in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_smiles(smiles_list, tokenizer, model, batch_size=32):
    """Mean-pooled last-hidden-state embeddings, shape (n, hidden_size)."""
    embeddings = []
    for start in range(0, len(smiles_list), batch_size):
        batch = list(smiles_list[start:start + batch_size])
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(emb.cpu().numpy())
    if not embeddings:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Embed SMILES with ChemBERTa")
    ap.add_argument("--input", required=True, help="CSV with a SMILES column")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    smiles = df[args.smiles_col].astype(str).tolist()
    tokenizer, model = load_embedding_model(args.model)
    emb = embed_smiles(smiles, tokenizer, model, batch_size=args.batch_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, embeddings=emb,
                        smiles=np.array(smiles))
    print(f"wrote {args.out}: {emb.shape[0]} x {emb.shape[1]} "
          f"({args.model})")


if __name__ == "__main__":
    main()
