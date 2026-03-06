"""Runtime logging helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TextIO

from .report_writer import ensure_dir


class PipelineLogger:
    """Emit logs to screen and optionally mirror them into one log file."""

    def __init__(self, log_path: Optional[Path]) -> None:
        self.log_path = log_path
        self._handle: Optional[TextIO] = None
        if log_path is not None:
            ensure_dir(log_path.parent)
            self._handle = log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _emit(self, text: str, stream: TextIO) -> None:
        if not text:
            return
        out = text if text.endswith("\n") else f"{text}\n"
        stream.write(out)
        stream.flush()
        if self._handle is not None:
            self._handle.write(out)
            self._handle.flush()

    def info(self, text: str) -> None:
        self._emit(text, sys.stdout)

    def stderr(self, text: str) -> None:
        self._emit(text, sys.stderr)

    def warning(self, text: str) -> None:
        self._emit(f"WARNING: {text}", sys.stderr)
