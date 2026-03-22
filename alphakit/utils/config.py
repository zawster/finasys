"""Global configuration for alphakit."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class AlphaKitConfig(BaseModel):
    """Global configuration."""

    cache_dir: Path = Path.home() / ".alphakit" / "cache"
    cache_enabled: bool = True
    default_backend: str = "polars"

    def ensure_cache_dir(self) -> Path:
        """Create cache directory if it doesn't exist and return the path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


# Global singleton config
config = AlphaKitConfig()
