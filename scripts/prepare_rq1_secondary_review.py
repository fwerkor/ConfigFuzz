#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from configfuzz.rq1_secondary_review import (
    build_secondary_review_packet,
    validate_secondary_review_template,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an RQ1 secondary-review packet with primary labels hidden."
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--primary-adjudication", type=Path, required=True)
    parser.add_argument("--packet-output", type=Path, required=True)
    parser.add_argument("--template-output", type=Path, required=True)
    parser.add_argument("--shuffle-seed", default="configfuzz-rq1-secondary-v1")
    args = parser.parse_args()

    packet, template = build_secondary_review_packet(
        args.audit,
        args.primary_adjudication,
        shuffle_seed=args.shuffle_seed,
    )
    for path, payload in (
        (args.packet_output, packet),
        (args.template_output, template),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
    validation = validate_secondary_review_template(args.template_output)
    print(
        json.dumps(
            {
                "packet_output": str(args.packet_output),
                "template_output": str(args.template_output),
                **validation,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
