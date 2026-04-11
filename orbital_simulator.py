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
r_orbit = 6.771e6   # Orbit radius: Earth's radius (6,371,000m) + altitude (400,000m)


# ─────────────────────────────────────────
# INITIAL CONDITIONS
# ISS-like orbit: 400km above Earth's surface
# ─────────────────────────────────────────

# Distance = Earth's radius (6,371,000m) + altitude of satellite (400,000m).
# Satellite starts on positive x-axis at 400,000m altitude
# Circular orbit velocity: v = √(GM/r) — from gravitational and centripetal force balance 
v_circ = np.sqrt((G*M) / r_orbit)   # Circular orbit speed at 400km altitude (m s⁻¹) 
r0 = np.array([r_orbit, 0.0])       # Initial position (m)      
v0 = np.array([0.0, v_circ])  # Initial velocity (ms⁻¹) — perpendicular to r0 for circular orbit


# ─────────────────────────────────────────
# Gravity Function
# ─────────────────────────────────────────

def acceleration(r):
    # Satellite mass cancels in F=ma — result independent of satellite mass
    # Dividing by r_mag^3 combines magnitude (GM/r^2) and unit direction
    r_mag = np.linalg.norm(r)
    return -(G * M / r_mag**3) * r    # Negative sign: acceleration points towards Earth

# Verification: magnitude should be ~8.69 m/s² at 400,000m altitude
_test = acceleration(r0)
print(f"Gravity check — magnitude: {np.linalg.norm(_test):.4f} m/s²")