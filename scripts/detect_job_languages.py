#!/usr/bin/env python3
"""
Detect languages for jobs in sample_jobs_multi_lang.json and create filtered English version.

Usage:
    python scripts/detect_job_languages.py

This script:
1. Reads data/sample_jobs_multi_lang.json
2. Detects language for each job using langdetect
3. Adds language field to each job
4. Saves updated version back to sample_jobs_multi_lang.json
5. Creates sample_jobs.json with only English jobs
"""

import json
from pathlib import Path
from collections import Counter

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Reproducible results
except ImportError:
    print("Error: langdetect not installed. Run: pip install langdetect")
    exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"
MULTI_LANG_PATH = DATA_DIR / "sample_jobs_multi_lang.json"
ENGLISH_ONLY_PATH = DATA_DIR / "sample_jobs.json"


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not text or len(text.strip()) < 20:
        return "unknown"
    try:
        # Use first 1000 chars for detection (more reliable than 500)
        return detect(text[:1000])
    except Exception:
        return "unknown"


def main():
    if not MULTI_LANG_PATH.exists():
        print(f"Error: {MULTI_LANG_PATH} not found.")
        exit(1)

    # Load jobs
    print(f"Loading jobs from {MULTI_LANG_PATH}...")
    with open(MULTI_LANG_PATH, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} jobs")

    # Detect languages
    print("Detecting languages...")
    lang_counts = Counter()
    
    for i, job in enumerate(jobs):
        # Use description field for language detection (most comprehensive text)
        description = job.get("description", "")
        if not description:
            # Fallback to title if description is empty
            description = job.get("title", "")
        
        lang = detect_language(description)
        job["language"] = lang
        lang_counts[lang] += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(jobs)} jobs...")

    # Show language distribution
    print(f"\nLanguage distribution:")
    for lang, count in lang_counts.most_common(15):
        print(f"  {lang}: {count}")

    # Save updated multi-lang version
    print(f"\nSaving updated jobs to {MULTI_LANG_PATH}...")
    with open(MULTI_LANG_PATH, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(jobs)} jobs with language field")

    # Filter English jobs only
    english_jobs = [job for job in jobs if job.get("language") == "en"]
    print(f"\nFiltering English jobs: {len(english_jobs)} out of {len(jobs)}")

    # Save English-only version
    print(f"Saving English-only jobs to {ENGLISH_ONLY_PATH}...")
    with open(ENGLISH_ONLY_PATH, "w", encoding="utf-8") as f:
        json.dump(english_jobs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(english_jobs)} English jobs")

    print("\nDone!")


if __name__ == "__main__":
    main()
