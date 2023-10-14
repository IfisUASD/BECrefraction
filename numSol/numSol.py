import numpy as np
from scipy.integrate import simps
from scipy.constants import pi
import matplotlib.pyplot as plt

# Constants and parameters
μ = 1.0  # Mass term
λ = 1.0  # Self-interaction strength
χ = 1.0  # Interaction strength
e = 1.0  # Charge of the scalar field (normalized)
c = 299792458  # Speed of light in m/s
ℏ = 1.0545718e-34  # Planck's constant in J*s
ε_0 = 8.854187817e-12  # Permittivity of free space

# Discretization and grid
L = 10.0  # Box size
N = 1000  # Number of grid points
dx = L / N  # Grid spacing
x = np.linspace(0, L, N)  # Spatial grid
dt = 0.01  # Time step
T = 10.0  # Total simulation time
Nt = int(T / dt)  # Number of time steps

# Initialize fields
φ = np.zeros(N, dtype=complex)
A = np.zeros(N)
J = np.zeros(N)

# Initialize arrays to store refractive indices
k_values = np.linspace(0.001, 1.0, 100)
refractive_indices = np.zeros((len(k_values), Nt))

# Function to update the fields in each time step
def update_fields():
    # Update J based on φ and A
    J[:] = -2j * e * (φ.conj() * np.gradient(φ, dx) - np.gradient(φ.conj(), dx) * φ) - 2 * e**2 * A * np.abs(φ)**2
    
    # Update A based on J
    A -= dt * np.gradient(J, dx) / ε_0  # Numerical update for A (discretized Maxwell's equation)

    # Update φ based on its equation of motion (Klein-Gordon equation)
    φ += dt * (np.gradient(np.gradient(φ, dx), dx) + (μ**2 - χ) * φ - 2 * λ * np.abs(φ)**2 * φ) / (ℏ * dt)
    
# Function to calculate the refractive index for a given k
def calculate_refractive_index(k, t):
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

# Time evolution and refractive index calculation
for i, k in enumerate(k_values):
    for t in range(Nt):
        update_fields()
        refractive_indices[i, t] = calculate_refractive_index(k, t*dt)

# Plot the refractive index as a function of time
for i, k in enumerate(k_values):
    plt.plot(np.arange(Nt)*dt, refractive_indices[i, :], label=f'k = {k:.3f}')

plt.xlabel('Time')
plt.ylabel('Refractive Index')
plt.legend()
plt.show()
