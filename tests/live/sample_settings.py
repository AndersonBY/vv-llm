from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SHARED_SETTINGS_FILE = Path(__file__).resolve().parents[1] / "sample_settings.py"
_spec = importlib.util.spec_from_file_location("_shared_sample_settings", _SHARED_SETTINGS_FILE)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Cannot load shared sample settings from {_SHARED_SETTINGS_FILE}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

sample_settings = _module.sample_settings
VECTORVEIN_API_KEY = getattr(_module, "VECTORVEIN_API_KEY", "")
VECTORVEIN_BASE_URL = getattr(_module, "VECTORVEIN_BASE_URL", "")
VECTORVEIN_APP_ID = getattr(_module, "VECTORVEIN_APP_ID", "")
VECTORVEIN_WORKFLOW_ID = getattr(_module, "VECTORVEIN_WORKFLOW_ID", "")
VECTORVEIN_VPP_API_KEY = getattr(_module, "VECTORVEIN_VPP_API_KEY", "")
VECTORVEIN_VAPP_ID = getattr(_module, "VECTORVEIN_VAPP_ID", "")
