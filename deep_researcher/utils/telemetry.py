"""Optional Monocle telemetry integration.

Monocle (``monocle_apptrace``) is an optional observability dependency, activated
only when ``MONOCLE_TRACING`` is truthy AND the package is installed
(``pip install deep-researcher[monocle]``). The env interface is deliberately
unprefixed (``MONOCLE_TRACING`` / ``MONOCLE_EXPORTERS`` / ``OKAHU_API_KEY``) so it
matches the surface used across the other Monocle-enabled apps.

When telemetry is disabled or absent, every helper here is a clean no-op: no
import errors, no extra spans, no behaviour change.
"""

import contextlib
import os
from typing import Any, Optional

_TRUTHY = {"1", "true", "yes", "on"}

# Mirror of monocle_apptrace's supported exporters, kept local so a typo fails
# fast with a clear message instead of an opaque upstream error.
_MONOCLE_EXPORTERS = ("file", "console", "okahu", "s3", "blob", "gcs")

# Set True by setup_telemetry only when Monocle is both enabled and installed.
_ACTIVE = False


def monocle_enabled() -> bool:
    """Return True if Monocle telemetry is enabled via MONOCLE_TRACING."""
    value = os.environ.get("MONOCLE_TRACING", "")
    return value.strip().lower() in _TRUTHY


def _exporters() -> str:
    """The configured, comma-separated exporter string (default ``file``)."""
    value = os.environ.get("MONOCLE_EXPORTERS", "")
    return value.strip() if value and value.strip() else "file"


def setup_telemetry(workflow_name: str) -> bool:
    """Initialise Monocle telemetry when enabled; a clean no-op otherwise.

    Reads MONOCLE_EXPORTERS, validates it, then forwards the comma-separated
    string to setup_monocle_telemetry. Returns True when telemetry was activated.
    """
    global _ACTIVE
    if not monocle_enabled():
        return False

    exporters = _exporters()
    selected = [e.strip() for e in exporters.split(",") if e.strip()]

    # Fail fast on an unknown exporter or okahu without a key, before instrumenting.
    unknown = [e for e in selected if e not in _MONOCLE_EXPORTERS]
    if unknown:
        raise ValueError(
            f"MONOCLE_EXPORTERS has unknown exporter(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(_MONOCLE_EXPORTERS)}."
        )
    if "okahu" in selected and not os.environ.get("OKAHU_API_KEY"):
        raise ValueError("Monocle 'okahu' exporter is selected but OKAHU_API_KEY is not set.")

    try:
        from monocle_apptrace import setup_monocle_telemetry
    except ImportError as exc:
        raise RuntimeError(
            "MONOCLE_TRACING is enabled but monocle_apptrace is not installed. "
            "Install the 'monocle' extra: pip install \"deep-researcher[monocle]\"."
        ) from exc

    # monocle_exporters_list takes the comma-separated string as-is (monocle's API).
    setup_monocle_telemetry(workflow_name=workflow_name, monocle_exporters_list=exporters)
    _ACTIVE = True
    return True


@contextlib.asynccontextmanager
async def trace_span(*args: Any, **kwargs: Any):
    """Open a Monocle span when telemetry is active; a no-op otherwise."""
    if _ACTIVE:
        from monocle_apptrace.instrumentation.common.instrumentor import amonocle_trace

        async with amonocle_trace(*args, **kwargs):
            yield
    else:
        yield


@contextlib.asynccontextmanager
async def trace_scope(*args: Any, **kwargs: Any):
    """Open a Monocle scope when telemetry is active; a no-op otherwise."""
    if _ACTIVE:
        from monocle_apptrace.instrumentation.common.scope_wrapper import (
            amonocle_trace_scope,
        )

        async with amonocle_trace_scope(*args, **kwargs):
            yield
    else:
        yield


def current_span() -> Optional[Any]:
    """Return the current Monocle span when telemetry is active, else None."""
    if _ACTIVE:
        from monocle_apptrace.instrumentation.common.wrapper import (
            get_current_monocle_span,
        )

        return get_current_monocle_span()
    return None
