"""
Registry of incomplete / placeholder parts of the simulator.

Anything decorated with @draft(...) or calling draft_notice(...) is NOT a final
model. A warning is printed the first time each placeholder is used, and
`print_draft_report()` lists every placeholder touched in a run, so draft
components can never silently end up behind a plotted result.

Search the code for "DRAFT" to find all of them.
"""
from __future__ import annotations

import functools
import warnings

DRAFT_ITEMS: dict[str, str] = {}
_USED: set[str] = set()


class DraftWarning(UserWarning):
    pass


def draft_notice(tag: str, message: str) -> None:
    DRAFT_ITEMS.setdefault(tag, message)
    if tag not in _USED:
        _USED.add(tag)
        warnings.warn(f"[DRAFT:{tag}] {message}", DraftWarning, stacklevel=3)


def draft(tag: str, message: str):
    DRAFT_ITEMS.setdefault(tag, message)

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            draft_notice(tag, message)
            return fn(*a, **kw)
        wrapper.__draft__ = (tag, message)
        return wrapper
    return deco


def used_drafts() -> dict[str, str]:
    return {t: DRAFT_ITEMS[t] for t in sorted(_USED)}


def print_draft_report() -> None:
    items = used_drafts()
    if not items:
        print("No draft placeholders were used.")
        return
    bar = "=" * 78
    print(bar)
    print(" DRAFT / PLACEHOLDER COMPONENTS USED IN THIS RUN — results are provisional")
    print(bar)
    for t, m in items.items():
        print(f" * {t}: {m}")
    print(bar)
