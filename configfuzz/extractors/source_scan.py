from __future__ import annotations

from pathlib import Path
from typing import Iterable

from configfuzz.model import ConstraintSet

from .declarations import scan_declaration_paths_multi
from .python_ast import scan_python_paths_multi


def scan_source_paths_multi(
    paths: Iterable[Path],
    parameters: Iterable[str],
    *,
    strict: bool = True,
    jobs: int = 1,
) -> dict[str, ConstraintSet]:
    """Combine executable Python guards with declarative configuration schemas."""

    source_paths = list(paths)
    ordered_parameters = list(dict.fromkeys(str(item) for item in parameters))
    code_results = scan_python_paths_multi(
        source_paths,
        ordered_parameters,
        strict=strict,
        jobs=jobs,
    )
    declaration_results = scan_declaration_paths_multi(source_paths, ordered_parameters)

    merged: dict[str, ConstraintSet] = {}
    for parameter in ordered_parameters:
        code = code_results[parameter]
        declarations = declaration_results[parameter]
        result = ConstraintSet(
            parameter=parameter,
            metadata={
                "extractor": "combined_static",
                "mode": "strict" if strict else "broad",
                "code": code.metadata,
                "declarations": declarations.metadata,
            },
        )
        result.extend(code.constraints)
        result.extend(declarations.constraints)
        result.metadata["accepted_candidates"] = len(result.constraints)
        merged[parameter] = result
    return merged
