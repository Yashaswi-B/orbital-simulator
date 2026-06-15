# Orbital Simulator
# Models a satellite orbiting Earth under Newtonian gravity
# Integration method: Runge-Kutta 4th order (RK4)
# Author: Yashaswi Burugupalli
# Started: March 2026

import os
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

G = 6.674e-11                 # Gravitational constant (m³ kg⁻¹ s⁻²)
M_earth = 5.9722e24           # Mass of Earth (kg)
mu_earth = G * M_earth        # Standard gravitational parameter (m³ s⁻²)
R_earth = 6.371e6             # Radius of Earth (m)
ALTITUDE = 400_000            # Orbital altitude above Earth's surface (m)


# ─────────────────────────────────────────
# INITIAL CONDITIONS
# ISS-like orbit: 400km above Earth's surface
# ─────────────────────────────────────────

def calculate_initial_conditions(altitude):
    # Calculates the initial position and velocity for a circular orbit.
    r_orbit = R_earth + altitude
    # Circular orbit velocity: v = √(μ/r)
    v_circ = np.sqrt(mu_earth / r_orbit)
    r0 = np.array([r_orbit, 0.0])
    v0 = np.array([0.0, v_circ])
    return r0, v0


# ─────────────────────────────────────────
# Gravity Function
# ─────────────────────────────────────────

def acceleration(r):
    # Satellite mass cancels in F=ma — result independent of satellite mass
    # Dividing by r_mag^3 combines magnitude (GM/r^2) and unit direction
    r_mag = np.linalg.norm(r)
    return -(mu_earth / r_mag**3) * r    # Negative sign: acceleration points towards Earth


# ─────────────────────────────────────────
# Forward Euler Integration
# ─────────────────────────────────────────

def simulate_euler(r0, v0, dt, steps):
    # Simulates the orbit using the Euler (first-order) integration method.
    # Returns arrays of positions and velocities over time.
    positions = np.zeros((steps, 2))
    velocities = np.zeros((steps, 2))

    r = r0.copy()
    v = v0.copy()

    for i in range(steps):
        positions[i] = r
        velocities[i] = v
        
        a = acceleration(r)
        
        # Updates position and velocity
        r = r + v * dt
        v = v + a * dt

    return positions, velocities


# ─────────────────────────────────────────
# Specific Orbital Energy Calculation
# ─────────────────────────────────────────

def calculate_orbital_energy(positions, velocities):
    # Calculates the specific orbital energy at the start and end of the simulation.
    # epsilon = v^2/2 - mu/r
    
    # Initial energy
    r0_mag = (positions[0][0]**2 + positions[0][1]**2)**0.5
    v0_mag = (velocities[0][0]**2 + velocities[0][1]**2)**0.5
    initial_energy = (v0_mag**2 / 2) - (mu_earth / r0_mag)

    # Final energy
    rf_mag = (positions[-1][0]**2 + positions[-1][1]**2)**0.5
    vf_mag = (velocities[-1][0]**2 + velocities[-1][1]**2)**0.5
    final_energy = (vf_mag**2 / 2) - (mu_earth / rf_mag)

    # Percentage drift
    energy_drift_pct = abs((final_energy - initial_energy) / initial_energy) * 100
    
    return initial_energy, final_energy, energy_drift_pct


# ─────────────────────────────────────────
# Kepler's Third Law Verification
# ─────────────────────────────────────────

def measure_orbital_period(positions, dt):
    # Measures the orbital period from trajectory data by detecting
    # when the satellite completes a full orbit (y crosses zero from
    # negative to positive, i.e. returns to the +x axis).
    # Uses linear interpolation for sub-timestep accuracy.
    crossing_times = []
    for i in range(1, len(positions)):
        y_prev = positions[i - 1, 1]
        y_curr = positions[i, 1]
        # Detect positive-going zero crossing (y goes from - to +)
        if y_prev < 0 and y_curr >= 0:
            # Linear interpolation to find precise crossing time
            frac = -y_prev / (y_curr - y_prev)
            t_cross = (i - 1 + frac) * dt
            crossing_times.append(t_cross)

    if len(crossing_times) < 2:
        return None, []

    # Period = time between consecutive crossings
    periods = [crossing_times[j] - crossing_times[j - 1]
               for j in range(1, len(crossing_times))]
    avg_period = sum(periods) / len(periods)
    return avg_period, periods


# ─────────────────────────────────────────
# ODE Reduction: The Derivative Function
# ─────────────────────────────────────────

def deriv(state):

    # Extracts position (first two elements) and velocity (last 2 elements)
    r = state[:2]
    v = state[2:]

    # Computes acceleration based on current position
    a = acceleration(r)

    # Returns the derivative of the state: [vx, vy, ax, ay]
    return np.concatenate([v, a])


# ─────────────────────────────────────────
# Runge-Kutta 4th order
# ─────────────────────────────────────────

def rk4step(state, dt):

    # Calculates the four slopes (k1, k2, k3, k4) using the derivative function
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    
    # Returns new state by using weighted average of the slopes
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# ─────────────────────────────────────────
# Plotting the Orbit
# ─────────────────────────────────────────

def plot_results(positions, rk4_positions, dt, steps, altitude):
    # Handles all matplotlib logic for plotting orbit and drift.
    os.makedirs("images", exist_ok=True)
    
    # 1. Plot Forward Euler Orbit
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_coords, y_coords, color="blue", lw=1.2, label="Orbit")

    earth = plt.Circle((0, 0), R_earth, color="dodgerblue", label="Earth")
    ax.add_patch(earth)

    ax.set_aspect("equal")
    ax.set_title(f"Stage 1: 2D Euler Orbit ({altitude/1000:.0f} km altitude)", fontsize=13)
    ax.set_xlabel("x position (m)", fontsize=10)
    ax.set_ylabel("y position (m)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/stage1_Euler_orbit.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Plot Forward Euler Orbital Drift
    # Optimized radii magnitude calculation
    radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    radius_error = radii - radii[0]
    time_array = np.arange(steps) * dt / 60

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_array, radius_error / 1000, color="red", lw=1.2)
    ax.set_xlabel("Time (minutes)", fontsize=10)
    ax.set_ylabel("Radius error from initial orbit (km)", fontsize=10)
    ax.set_title("Stage 1: Euler Integration Drift Over 5 Orbits", fontsize=13)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    plt.tight_layout()
    plt.savefig("images/stage1_Euler_drift.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot RK4 Orbit
    #print(positions[:,0])
    x_coords = rk4_positions[:, 0]
    y_coords = rk4_positions[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_coords, y_coords, color="blue", lw=1.2, label="Orbit")

    # Earth — Radius 6,371,000m
    earth = plt.Circle((0,0), R_earth, color="dodgerblue", label="Earth")
    ax.add_patch(earth)

    ax.set_aspect("equal")
    ax.set_title(f"Stage 2: 2D RK4 Orbit ({ALTITUDE/1000:.0f} km altitude)", fontsize=13)
    ax.set_xlabel("x position (m)", fontsize=10)
    ax.set_ylabel("y position (m)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig("images/stage2_RK4_orbit.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # 1. Initialization
    r0, v0 = calculate_initial_conditions(ALTITUDE)
    
    # Verify gravity at initial position
    a_test = acceleration(r0)
    print(f"Gravity verification at {ALTITUDE/1000:.0f} km altitude:")
    a_test_mag = (a_test[0]**2 + a_test[1]**2)**0.5
    print(f"  Acceleration magnitude: {a_test_mag:.4f} m/s²")
    print(f"  Direction: {'toward Earth ✓' if a_test[0] < 0 else 'ERROR — not toward Earth'}\n")

    # 2. Simulation parameters
    num_orbits = 5
    dt = 10.0
    r_orbit = R_earth + ALTITUDE
    T_kepler = 2 * np.pi * np.sqrt(r_orbit**3 / mu_earth)
    steps = round((num_orbits * T_kepler) / dt)

    # 3. Run Simulation
    positions, velocities = simulate_euler(r0, v0, dt, steps)
    
    # 4. Energy Calculations & Console Output
    initial_energy, final_energy, energy_drift_pct = calculate_orbital_energy(positions, velocities)
    
    print("--- Specific Orbital Energy Check ---")
    print(f"Initial Specific Energy: {initial_energy:.2f} J/kg")
    print(f"Final Specific Energy:   {final_energy:.2f} J/kg")
    print(f"Energy Drift (Euler):    {energy_drift_pct:.4f}%")
    print("Note: The 'phantom energy' drift observed is a direct result of the Taylor series truncation error of the Euler method.\n")
    
    # Kepler period check — measured from trajectory data
    T_measured, all_periods = measure_orbital_period(positions, dt)
    print("--- Kepler Period Verification ---")
    print(f"Kepler period:    {T_kepler:.1f} s ({T_kepler/60:.2f} min)")
    if T_measured is not None:
        period_error_pct = abs(T_measured - T_kepler) / T_kepler * 100
        print(f"Measured period:  {T_measured:.1f} s (avg of {len(all_periods)} orbits)")
        print(f"Period error:     {period_error_pct:.2f}%")
        if len(all_periods) > 1:
            print(f"  Per-orbit periods: {', '.join(f'{p:.1f} s' for p in all_periods)}")
    else:
        print("  Could not measure period (not enough complete orbits).")
    print()

    # 5. RK4 Simulation
    initial_state = np.concatenate([r0, v0])
    
    rk4_positions = np.zeros((steps, 2))
    s = initial_state.copy()

    for i in range(steps):

        # Records the current x,y positions
        rk4_positions[i] = s[:2]

        # Updates the state vector using RK4
        s = rk4step(s, dt)
    
    # 6. Visualizations
    plot_results(positions, rk4_positions, dt, steps, ALTITUDE)
    print("Plots saved to images/ directory.")
