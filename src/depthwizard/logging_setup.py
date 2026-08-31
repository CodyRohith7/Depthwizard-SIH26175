"""Central logging configuration for DepthWizard.

M0 requirement: real, visible logging and error handling instead of the
prototype's bare `except Exception: <silently do something else>` pattern.

Usage:
    from depthwizard.logging_setup import configure_logging
    configure_logging()  # call once, e.g. at process start
    import logging
    log = logging.getLogger("depthwizard.depth")
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "logging.yaml"
_configured = False


def configure_logging(config_path: Path | None = None, level: str | None = None) -> None:
    """Configure logging from configs/logging.yaml (idempotent).

    `level` overrides the configured root/depthwizard level (also readable
    from the DEPTHWIZARD_LOG_LEVEL environment variable) -- useful for tests
    and CLI tools that want quieter or louder output without editing YAML.
    """
    global _configured
    path = config_path or _DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
    effective_level = level or os.environ.get("DEPTHWIZARD_LOG_LEVEL")
    if effective_level:
        logging.getLogger("depthwizard").setLevel(effective_level.upper())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
