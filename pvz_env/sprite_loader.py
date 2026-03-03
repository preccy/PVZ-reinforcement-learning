from __future__ import annotations

from pathlib import Path
from typing import Optional

import pygame

_SURFACE_CACHE: dict[str, pygame.Surface] = {}


def load_png(path: str) -> pygame.Surface:
    resolved = str(Path(path).resolve())
    cached = _SURFACE_CACHE.get(resolved)
    if cached is not None:
        return cached
    surface = pygame.image.load(resolved).convert_alpha()
    _SURFACE_CACHE[resolved] = surface
    return surface


def safe_load_png(path: str) -> Optional[pygame.Surface]:
    try:
        return load_png(path)
    except (FileNotFoundError, pygame.error, OSError):
        return None


def scale_surface(surface: pygame.Surface, target_w: int, target_h: int) -> pygame.Surface:
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    return pygame.transform.smoothscale(surface, (target_w, target_h))
