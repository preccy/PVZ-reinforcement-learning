from __future__ import annotations

from typing import Any


def format_text_snapshot(snapshot: dict[str, Any]) -> str:
    return (
        f"Step {snapshot['step']} | Sun {snapshot['sun']} | "
        f"Plants {len(snapshot['plants'])} | Zombies {len(snapshot['zombies'])} | "
        f"Mowers {snapshot['mowers']}"
    )
