import numpy as np
from scipy.constants import pi
from scipy.integrate import simps

# Function to update the fields in each time step
def update_fields(φ, A, J, dx, dt):

    # Constants and parameters
    μ = 1.0  # Mass term
    λ = 1.0  # Self-interaction strength
    χ = 1.0  # Interaction strength
    e = 1.0  # Charge of the scalar field (normalized)
    c = 299792458  # Speed of light in m/s
    ℏ = 1.0545718e-34  # Planck's constant in J*s
    ε_0 = 8.854187817e-12  # Permittivity of free space

    # Update J based on φ and A
    J[:] = -2j * e * (φ.conj() * np.gradient(φ, dx) - np.gradient(φ.conj(), dx) * φ) - 2 * e**2 * A * np.abs(φ)**2
    
    # Update A based on J
    A -= dt * np.gradient(J, dx) / ε_0  # Numerical update for A (discretized Maxwell's equation)

    # Update φ based on its equation of motion (Klein-Gordon equation)
    φ += dt * (np.gradient(np.gradient(φ, dx), dx) + (μ**2 - χ) * φ - 2 * λ * np.abs(φ)**2 * φ) / (ℏ * dt)

    
# Function to calculate the refractive index for a given k
def calculate_refractive_index(φ, A, k, t, x, dx):

    # Constants and parameters
    μ = 1.0  # Mass term
    λ = 1.0  # Self-interaction strength
    χ = 1.0  # Interaction strength
    e = 1.0  # Charge of the scalar field (normalized)
    c = 299792458  # Speed of light in m/s
    ℏ = 1.0545718e-34  # Planck's constant in J*s
    ε_0 = 8.854187817e-12  # Permittivity of free space

    k_term = np.exp(-1j * k * x)
    
    A_k = np.fft.fft(A)
    φ_k = np.fft.fft(φ)
    
    lagrangian_density = (
        np.abs(np.gradient(φ, dx))**2
        - μ * np.abs(φ)**2
        - λ * np.abs(φ)**4
        - 0.25 * np.abs(np.gradient(A, dx))**2
        + χ * np.abs(φ)**2
    )
    
    integral_result = simps(lagrangian_density * k_term, dx=dx)
    n = 1.0 + (χ / (pi * integral_result))
    
    return n












