"""Standalone entry point for just the near-dedup stage.

Kept separate from data_preprocessor.py's __main__ so it can be re-run
without redoing the already-completed cleaning and exact-dedup stages.
"""
import json

from src.utils.download_public import load_data_config
from src.data.data_preprocessor import run_near_dedup_stage

if __name__ == "__main__":
    config = load_data_config()
    stats = run_near_dedup_stage(config["dedup"])
    print(json.dumps(stats, ensure_ascii=False, indent=2))
