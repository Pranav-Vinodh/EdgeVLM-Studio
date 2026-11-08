"""
Remote Colab GPU worker and client bridge for EdgeVLM-Studio.
"""

from edgevlm.remote.colab_server import ColabServerApp, create_colab_server

__all__ = ["ColabServerApp", "create_colab_server"]
