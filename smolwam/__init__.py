"""SmolWAM: a latent world-action model for robotic manipulation on edge devices."""

__version__ = "1.0.0"

from smolwam.config import SmolWAMConfig  # noqa: E402
from smolwam.policy import SmolWAM  # noqa: E402

__all__ = ["SmolWAM", "SmolWAMConfig", "__version__"]
