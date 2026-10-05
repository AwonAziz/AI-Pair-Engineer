"""Prompt loading.

Prompts ship inside the package and are read through ``importlib.resources`` so
they resolve regardless of the current working directory. The previous
``open("ai-pair-engineer/prompts/analyzer.txt")`` broke the moment Streamlit
ran from anywhere other than the parent of that folder.
"""

from __future__ import annotations

from functools import cache

PROMPTS = ("analyzer", "tester", "refactor", "reviewer")


@cache
def load_prompt(name: str) -> str:
    """Return the text of a packaged prompt by stem name.

    Raises:
        KeyError: if ``name`` is not a known prompt.
    """
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt {name!r}; expected one of {', '.join(PROMPTS)}")

    # importlib.resources.files is the supported path-based API from 3.9+.
    from importlib.resources import files

    resource = files(__package__).joinpath(f"prompts/{name}.txt")
    return resource.read_text(encoding="utf-8")
