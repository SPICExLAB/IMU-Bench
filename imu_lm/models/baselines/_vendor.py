"""Where the third-party baseline code lives.

The adapters import upstream code (Inertia-1, ssl-wearables, NormWear) from a `baselines/` tree. In a
clone of the release that is `<repo>/baselines`; set IMUBENCH_BASELINES_DIR to use a copy elsewhere
(on our cluster the tree lives on project storage, the last fallback below).
"""
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CANDIDATES = (
    os.environ.get("IMUBENCH_BASELINES_DIR"),
    os.path.join(_REPO_ROOT, "baselines"),
    "/projects/b53017/vbc1591/SSL_baselines/baselines",
)


def baselines_dir(*parts: str) -> str:
    """First existing `baselines/` root joined with `parts` (the repo default if none exists)."""
    for root in _CANDIDATES:
        if root and os.path.isdir(root):
            return os.path.normpath(os.path.join(root, *parts))
    return os.path.normpath(os.path.join(_REPO_ROOT, "baselines", *parts))
