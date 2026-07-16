"""Compute mean sentence lengths (words per sentence) for WildChat user messages.

Writes results to <data_dir>/sentence_lengths.parquet
Run from the repository root.

Usage:
    python compute_sentence_lengths.py                # wildchat-4.8m (default)
    python compute_sentence_lengths.py wildchat-1m    # or wildchat-4.8m
"""

import os
import sys
import polars as pl
import spacy
from tqdm.auto import tqdm

from wildchat_metrics import DATASETS, load_source_text

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "wildchat-4.8m"
cfg = DATASETS[dataset_name]
DATA_DIR = cfg.data_dir

SENT_LEN_CACHE = f"{DATA_DIR}/sentence_lengths.parquet"

if os.path.exists(SENT_LEN_CACHE):
    print(f"{SENT_LEN_CACHE} already exists – nothing to do.")
    raise SystemExit(0)

# --- Load input data --------------------------------------------------------
key_cols = ["conversation_hash", "model", "timestamp", "turn", "language", "hashed_ip", "state", "country"]

print("Loading source conversations...")
df = load_source_text(cfg)

# --- Extract user message text ---------------------------------------------
print("Extracting user message text...")
user_texts = (
    df.select(
        pl.col("Text")
        .str.extract_all(r"<\| start user message \|>[\s\S]*?<\| end user message \|>")
        .list.eval(
            pl.element()
            .str.replace(r"<\| start user message \|>", "")
            .str.replace(r"<\| end user message \|>", "")
            .str.strip_chars()
        )
        .list.join(" ")
        .alias("_user_text")
    )["_user_text"]
    .to_list()
)

# --- Compute mean sentence length using spaCy -------------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler", "tagger"])

print("Computing sentence lengths...")
sent_lens = []
num_processes = min(100, os.cpu_count() or 1)
for doc in tqdm(nlp.pipe(user_texts, batch_size=1000, n_process=num_processes), total=len(user_texts), desc="Sentence lengths"):
    lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in doc.sents]
    sent_lens.append(sum(lengths) / len(lengths) if lengths else None)

# --- Save -------------------------------------------------------------------
result = df.select(key_cols).with_columns(pl.Series("mean_sentence_len_words", sent_lens))
result.write_parquet(SENT_LEN_CACHE)
print(f"Saved sentence lengths to {SENT_LEN_CACHE}")
