#!/usr/bin/env python3
"""Backward-compatible wrapper for Beauty SASRec++ 5-seed runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sasrec_5seed import main

if __name__ == "__main__":
    raise SystemExit(main(["--category", "Beauty", *sys.argv[1:]]))
