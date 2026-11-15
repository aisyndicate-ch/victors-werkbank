"""Shared fail-closed signal: the container executor cannot be used."""


class SandboxUnavailable(RuntimeError):
    """Missing or unusable isolation; never fall back to host execution."""
