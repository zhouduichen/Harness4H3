"""Shared MiniMax-H3 sigma schedules for Student training and sampling."""

from __future__ import annotations


def h3_video_sigmas(steps: int, *, shift: float = 12.0) -> tuple[float, ...]:
    """Return the shifted video sigma grid used by the real H3 sampler."""

    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    amount = float(shift)
    if amount <= 0:
        raise ValueError("shift must be positive")

    def shifted(value: float) -> float:
        return amount * value / (1.0 + (amount - 1.0) * value)

    base = [1.0 - index / float(steps) for index in range(int(steps) + 1)]
    return tuple(shifted(value) for value in base)


__all__ = ["h3_video_sigmas"]
