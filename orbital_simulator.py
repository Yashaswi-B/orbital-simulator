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

# Different version of simulate_euler() which takes a state vector whereas the original took separate r, v arrays.
def euler_step(state, dt):
    r = state[:2]
    v = state[2:]
    a = acceleration(r)

    r_new = r + v * dt
    v_new = v + a * dt
    return np.concatenate([r_new, v_new])


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
# Propagating Function
# ─────────────────────────────────────────

# Unified propagator — same initial state, same dt, only the step function changes.
# This makes the Euler vs RK4 comparison as accurate as possible.

def propagate(initial_state, dt, steps, method="rk4"):
    step_func = rk4step if method=="rk4" else euler_step
    states = np.zeros((steps, len(initial_state)))
    s = initial_state.copy()

    for i in range(steps):
        states[i] = s
        s = step_func(s, dt)
    return states

# ─────────────────────────────────────────
# Energy and Angular Momentum Conservation Tracking
# ─────────────────────────────────────────

def specific_energy(r, v):
    KE = 0.5 * np.dot(v, v)                 # Kinetic energy per unit mass: v²/2
    PE = -mu_earth / np.linalg.norm(r)      # Potential energy per unit mass: -μ/r
    return KE + PE                          # This value should remain constant in a correct simulation

def angular_momentum(r, v):
    return r[0]*v[1] - r[1]*v[0]            # This value should remain constant in a correct simulation

# ─────────────────────────────────────────
# Plotting the Orbit
# ─────────────────────────────────────────

def plot_results(dt, steps, altitude):
    # Handles all matplotlib logic for plotting orbit and drift.
    os.makedirs("images", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(euler_x_coords, euler_y_coords, color="blue", lw=1.2, label="Orbit")

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
    radii = np.sqrt(euler_x_coords**2 + euler_y_coords**2)
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
    rk4_x_coords = r_rk4[:, 0]
    rk4_y_coords = r_rk4[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rk4_x_coords, rk4_y_coords, color="blue", lw=1.2, label="Orbit")

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

    # Plotting Energy and Angular Momentum Conservation
    # Calculating initial values
    E0 = specific_energy(initial_state[:2], initial_state[2:])
    h0 = angular_momentum(initial_state[:2], initial_state[2:])

    # Relative errors in energy and angular momentum
    energy_error_rk4 = np.array([(specific_energy(r, v) - E0) / abs(E0) for r, v in zip(r_rk4, v_rk4)])
    energy_error_eul = np.array([(specific_energy(r, v) - E0) / abs(E0) for r, v in zip(r_euler, v_euler)])

    angular_momentum_error_rk4 = np.array([(angular_momentum(r, v) - h0) / abs(h0) for r, v in zip(r_rk4, v_rk4)])
    angular_momentum_error_eul = np.array([(angular_momentum(r, v) - h0) / abs(h0) for r, v in zip(r_euler, v_euler)])

    time_array = np.arange(steps) * dt / 3600

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Specific Energy Relative Error
    ax1.plot(time_array, energy_error_eul, color='crimson', lw=1.5, label='Euler (1st Order)')
    ax1.plot(time_array, energy_error_rk4, color='royalblue', lw=1.5, label='RK4 (4th Order)')
    ax1.set_title('Stage 2: Relative Specific Mechanical Energy Error', fontsize=14)
    ax1.set_ylabel('(E - E0) / |E0|', fontsize=12)
    ax1.axhline(0, color='black', ls='--', lw=0.8)
    ax1.legend(fontsize=12)

    # Panel 2: Angular Momentum Relative Error
    ax2.plot(time_array, angular_momentum_error_eul, color='crimson', lw=1.5, label='Euler')
    ax2.plot(time_array, angular_momentum_error_rk4, color='royalblue', lw=1.5, label='RK4')
    ax2.set_title('Stage 2: Relative Angular Momentum Error', fontsize=14)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('(h - h0) / |h0|', fontsize=12)
    ax2.axhline(0, color='black', ls='--', lw=0.8)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('images/stage2_energy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Orbit Comparison Plot (RK4 vs Euler)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(euler_x_coords, euler_y_coords, color='crimson',   lw=0.8, alpha=0.7, label='Euler')
    ax.plot(rk4_x_coords, rk4_y_coords, color='royalblue', lw=1.2, label='RK4')

    earth = plt.Circle((0, 0), R_earth, color='dodgerblue', label='Earth')
    ax.add_patch(earth)

    ax.set_aspect('equal')
    ax.set_title('Stage 2: Euler vs RK4 — 5 Orbital Periods', fontsize=13)
    ax.set_xlabel('x position (m)', fontsize=10)
    ax.set_ylabel('y position (m)', fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig('images/stage2_orbit_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 1. Initialization
    r0, v0 = calculate_initial_conditions(ALTITUDE)
    initial_state = np.concatenate([r0,v0])
    
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

    # Running both propagators
    states_rk4 = propagate(initial_state, dt, steps, method="rk4")
    states_euler = propagate(initial_state, dt, steps, method="euler")

    # Extracting positions and velocities from both methods
    r_rk4, v_rk4 = states_rk4[:, :2], states_rk4[:, 2:]
    r_euler, v_euler = states_euler[:, :2], states_euler[:, 2:]
    
    # Plot Forward Euler Orbit
    euler_x_coords = r_euler[:, 0]
    euler_y_coords = r_euler[:, 1]
    
    # 5. Visualizations
    plot_results(dt, steps, ALTITUDE)
    print("Plots saved to images/ directory.")
