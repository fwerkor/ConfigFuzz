# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict
from typing import Dict, List, Set

import msadapter
from msadapter.export.unflatten import _ModuleFrame, _SubmoduleEntry


def _outline_submodules(orig_graph: msadapter.fx.Graph):
    # Create an empty GraphModule to hold the outlined modules
    new_module = msadapter.fx.GraphModule(msadapter.nn.Module(), msadapter.fx.Graph())
    seen_nodes: Dict[str, msadapter.fx.Node] = {}
    seen_modules: Dict[int, List[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: Dict[str, Set[str]] = defaultdict(set)
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        None,
        [("", 0)],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
