import os
from typing import Optional

def get_env_with_prefix(base_name: str, prefix: str = "DR_", default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves an environment variable, checking for a prefixed version first.

    Args:
        base_name: The base name of the environment variable (e.g., "OPENAI_API_KEY").
        prefix: The prefix to check for (e.g., "DR_"). Defaults to "DR_".
        default: The default value to return if neither the prefixed nor the
                 base variable is found.

    Returns:
        The value of the environment variable, or the default value, or None.
    """
    prefixed_name = f"{prefix}{base_name}"
    value = os.getenv(prefixed_name)
    if value is not None:
        return value
    return os.getenv(base_name, default)
