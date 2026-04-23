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
R_earth = 6.371e6   # Radius of Earth (m)


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

# ─────────────────────────────────────────
# Euler Integration
# ─────────────────────────────────────────
# At each timestep:
# new_position = old_position + acceleration * dt
# new_velocity = old_velocity + acceleration * dt
num_orbits = 5
dt = 10.0                           # Timestep (s)
t = 5560.0                          # one orbital period (appromimation)
steps = int((num_orbits * t) / dt)
positions = np.zeros((steps, 2))    # pre-allocate: faster than appending

r = r0.copy()
v = v0.copy()

for i in range(steps):
    positions[i] = r       # record current position
    a = acceleration(r)    # acceleration at current position
    r = r + v * dt         # Update position using current velocity
    v = v + a * dt         # Update velocity using current acceleration

# ─────────────────────────────────────────
# Plotting the Orbit
# ─────────────────────────────────────────
x_coords = positions[:, 0]
y_coords = positions[:, 1]

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x_coords, y_coords, color="blue", lw=1.2, label="Orbit")

# Earth — Radius 6,371,000m
earth = plt.Circle((0,0), R_earth, color="dodgerblue", label="Earth")
ax.add_patch(earth)

ax.set_aspect("equal")
ax.set_title("Stage 1: 2D Euler Orbit (400,000m altitude)", fontsize=13)
ax.set_xlabel("x position (m)", fontsize=10)
ax.set_ylabel("y position (m)", fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig("/Users/yasha/Desktop/orbital-simulator/img/stage1_orbit_v2.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────
# Kepler's Thrid Law Verification
# ─────────────────────────────────────────
# Independent check: does Newtonian gravity produce the correct orbital period?
# Kepler's Third Law: T = 2*pi * sqrt(r^3 / GM) — derived analytically
T_kepler = 2 * np.pi * np.sqrt(r_orbit**3 / (G * M))

T_simulated =  (steps * dt) / num_orbits 
error_pct = abs(T_simulated - T_kepler) / T_kepler * 100
print(f"Kepler period: {T_kepler:.1f} s ({T_kepler/60:.2f} min)")
print(f"Simulated Period: {T_simulated:.1f} s")
print((f"period error: {error_pct:.2f}%"))
print(f"Direction check: T_simulated {'>' if T_simulated > T_kepler else '<'} T_kepler — "
      f"{'consistent with Euler energy gain' if T_simulated > T_kepler else 'unexpected — check integrator'}")