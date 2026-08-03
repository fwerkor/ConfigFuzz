from typing import Any, Callable, Dict, List, Optional, Type

from msadapter.utils._pytree import PyTree, TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]
FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]

SUPPORTED_NODES: Dict[Type[Any], FlattenFuncSpec] = {}
SUPPORTED_NODES_EXACT_MATCH: Dict[Type[Any], Optional[FlattenFuncExactMatchSpec]] = {}


def register_pytree_flatten_spec(
    cls: Type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None,
) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec
