"""Composite (harder) subcategories: drag, Kepler orbits, driven and coupled
oscillators. Same task protocol as the original four; only the data differ."""
from .damping import DampingTask


class DragTask(DampingTask):
    """Object Motion -> Projectile Motion with Drag. Quadratic air drag, no closed form."""


class KeplerTask(DampingTask):
    """Gravitation -> Two-Body Orbits. Inverse-square central force from perihelion."""


class DrivenTask(DampingTask):
    """Oscillations -> Driven Damped Oscillator. Transient plus steady state."""


class CoupledTask(DampingTask):
    """Oscillations -> Coupled Oscillators. Two masses, three springs; pos = [x1, x2]."""
