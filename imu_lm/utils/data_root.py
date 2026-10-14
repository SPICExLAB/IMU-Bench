"""Where the benchmark's datasets live.

Task cards, templates and a few script defaults name datasets under our cluster's data root. Set
IMUBENCH_DATA_ROOT to a directory with the same layout (DATASETS.md, "Directory layout") and every such
path is rewritten to it when the registry is loaded and when a cell's config is generated.
"""
import os
from typing import Any

CLUSTER_DATA_ROOT = "/projects/b53017/vbc1591"


def data_path(p: Any) -> Any:
    root = os.environ.get("IMUBENCH_DATA_ROOT")
    if root and isinstance(p, str) and (p == CLUSTER_DATA_ROOT or p.startswith(CLUSTER_DATA_ROOT + "/")):
        return os.path.join(os.path.abspath(os.path.expanduser(root)), p[len(CLUSTER_DATA_ROOT):].lstrip("/"))
    return p


def remap_tree(obj: Any) -> Any:
    """data_path applied to every string in a nested dict / list (returns a new object)."""
    if not os.environ.get("IMUBENCH_DATA_ROOT"):
        return obj
    if isinstance(obj, dict):
        return {k: remap_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_tree(v) for v in obj]
    return data_path(obj)
