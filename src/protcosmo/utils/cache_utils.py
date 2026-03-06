"""Shared cache helpers."""

from __future__ import annotations

from typing import Dict


def lookup_cache_get(cache: Dict[str, object], key: str, loader, lock=None):
    """Get-or-load a cache value by key with optional coarse lock protection."""

    if lock is None:
        if key not in cache:
            cache[key] = loader(key)
        return cache[key]

    sentinel = object()
    with lock:
        cached = cache.get(key, sentinel)
    if cached is not sentinel:
        return cached

    loaded = loader(key)
    with lock:
        cached = cache.get(key, sentinel)
        if cached is sentinel:
            cache[key] = loaded
            return loaded
        return cached
