#!/usr/bin/env python3
"""
Add language detection to existing chunks.jsonl without re-running embeddings.

Usage:
    python scripts/add_language_to_chunks.py

This script:
1. Reads artifacts/chunks.jsonl
2. Detects language for each job using langdetect
3. Writes updated chunks with language field in meta
"""

import json
from pathlib import Path

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Reproducible results
except ImportError:
    print("Error: langdetect not installed. Run: pip install langdetect")
    exit(1)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
BACKUP_PATH = ARTIFACTS_DIR / "chunks.jsonl.bak"


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not text or len(text.strip()) < 20:
        return "unknown"
    try:
        return detect(text[:500])
    except Exception:
        return "unknown"


def main():
    if not CHUNKS_PATH.exists():
        print(f"Error: {CHUNKS_PATH} not found. Run pipeline build first.")
        exit(1)

    # Load chunks
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks")

    # Group by job and detect language once per job
    print("Detecting languages...")
    job_languages = {}
    job_texts = {}

    for ch in chunks:
        jk = ch.get("job_key", "")
        if jk and jk not in job_texts:
            job_texts[jk] = ch.get("text", "")

    total_jobs = len(job_texts)
    for i, (jk, text) in enumerate(job_texts.items()):
        job_languages[jk] = detect_language(text)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_jobs} jobs...")

    # Count languages
    from collections import Counter
    lang_counts = Counter(job_languages.values())
    print(f"Languages detected: {dict(lang_counts.most_common(10))}")

    # Update chunks with language
    updated = 0
    for ch in chunks:
        jk = ch.get("job_key", "")
        lang = job_languages.get(jk, "unknown")
        meta = ch.get("meta", {})
        if meta.get("language") != lang:
            meta["language"] = lang
            ch["meta"] = meta
            updated += 1

    # Backup original
    print(f"Backing up to {BACKUP_PATH}...")
    CHUNKS_PATH.rename(BACKUP_PATH)

    # Write updated chunks
    print(f"Writing updated chunks...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"Done! Updated {updated} chunks with language field.")
    print(f"Backup saved to {BACKUP_PATH}")


if __name__ == "__main__":
    main()
