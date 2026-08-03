#!/usr/bin/env python3
"""A deterministic target used to demonstrate ConfigFuzz's runtime loop."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-size", type=int, required=True)
    args = parser.parse_args()

    value = args.parallel_size
    if value < 1:
        print("CONFIG_INVALID: parallel size must be positive")
        return 2
    if 32 % value != 0:
        print("CONFIG_INVALID: hidden size must be divisible by parallel size")
        return 2

    print("MILESTONE: configuration accepted")
    if value == 16:
        print("BUG_ORACLE: simulated post-validation crash")
        return 3
    print("RUN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
