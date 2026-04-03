# Orbital Simulator — Stage 1
# Models a satellite orbiting Earth under Newtonian gravity
# Integration method: Euler (first-order)
# Author: Yashaswi Burugupalli
# Started: March 2026


import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

G = 6.674e-11       # Gravitational constant (m³ kg⁻¹ s⁻²)
M = 5.9722e24       # Mass of Earth (kg)


# ─────────────────────────────────────────
# INITIAL CONDITIONS
# ISS-like orbit: 400km above Earth's surface
# ─────────────────────────────────────────

# Distance = Earth's radius (6,371,000m) + altitude of satellite (400,000m).
# Satellite starts on positive x-axis at 400,000m altitude
# Circular orbit velocity: v = √(GM/r) — from gravitational and centripetal force balance 
v_circ = np.sqrt((G*M) / 6.771e6)   # Circular orbit speed at 400km altitude (m s⁻¹) 
r0 = np.array([6.771e6, 0.0])       # Initial position (m)      
v0 = np.array([0.0, v_circ])        # Initial velocity (ms⁻¹) — perpendicular to r0 for circular orbit
