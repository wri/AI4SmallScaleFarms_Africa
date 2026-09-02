"""Earth Engine initialisation and access to the Azzari et al. ``eetc`` tools.

Keeping these concerns in one place means every other module can simply call
``initialize_ee(config)`` / ``add_eetc_to_path(config)`` without repeating auth
boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import ee

from .config import PipelineConfig

# Scopes needed for Earth Engine + Drive/Cloud Storage exports.
EE_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/drive",
]

_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"


def initialize_ee(config: Optional[PipelineConfig] = None) -> None:
    """Initialise Earth Engine, authenticating interactively if required.

    Behaviour is driven by the config:
      * ``ee_service_account_key`` set -> service-account auth (best for batch
        exports / automation).
      * otherwise -> try ``ee.Initialize`` and fall back to ``ee.Authenticate``.
    """
    config = config or PipelineConfig()

    opt_url = _HIGH_VOLUME_URL if config.ee_high_volume else None

    if config.ee_service_account_key:
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            str(config.ee_service_account_key), scopes=EE_SCOPES
        )
        ee.Initialize(credentials=credentials, project=config.ee_project, opt_url=opt_url)
        return

    try:
        ee.Initialize(project=config.ee_project, opt_url=opt_url)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.ee_project, opt_url=opt_url)


def add_eetc_to_path(config: PipelineConfig) -> Path:
    """Make the cloned ``eetc`` repo importable and return its path.

    Clone it first with:  ``git clone https://github.com/shrutijain90/eetc.git``
    Then set ``eetc_path`` in the config (defaults to ``<project_root>/eetc``).
    """
    eetc_path = config.eetc_path or (config.project_root / "eetc")
    eetc_path = Path(eetc_path)
    if not eetc_path.exists():
        raise FileNotFoundError(
            f"eetc repo not found at {eetc_path}. Clone it with:\n"
            "    git clone https://github.com/shrutijain90/eetc.git\n"
            "and set `eetc_path` in your config."
        )
    for p in (str(eetc_path / "gee_tools"), str(eetc_path)):
        if p in sys.path:
            sys.path.remove(p)
    # Repo root first so `import gee_tools` resolves to this clone.
    # gee_tools/ is also on the path so `import harmonics` still works.
    sys.path.insert(0, str(eetc_path))
    sys.path.insert(1, str(eetc_path / "gee_tools"))
    return eetc_path


def import_harmonics():
    """Load Azzari ``harmonics`` from the cloned ``eetc`` repo.

    ``gee_tools.harmonics`` is not an installed package; it becomes importable
    after :func:`add_eetc_to_path`.
    """
    add_eetc_to_path(PipelineConfig())
    from gee_tools import harmonics  # type: ignore[import-not-found]

    return harmonics
