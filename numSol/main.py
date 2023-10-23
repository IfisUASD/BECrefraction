import numpy as np
import matplotlib.pyplot as plt
from numSol import update_fields, calculate_refractive_index

# Discretization and grid
L = 10.0  # Box size
N = 1000  # Number of grid points
dx = L / N  # Grid spacing
x = np.linspace(0, L, N)  # Spatial grid
dt = 0.01  # Time step
T = 10.0  # Total siμlation time
Nt = int(T / dt)  # Number of time steps

# Initialize fields
#TODO: get the initial values of de φ, A & J
φ = np.zeros(N, dtype=complex)
A = np.zeros(N)
J = np.zeros(N)
def A_0(x):
     return np.cos(x)

i = 0
while i < len(x):
     A[i] = A_0(x[i])
     i = i + 1
# Initialize arrays to store refractive indices
k_values = np.linspace(0.001, 1.0, 100)
refractive_indices = np.zeros((len(k_values), Nt))

def main():
    # Time evolution and refractive index calculation 
    for i, k in enumerate(k_values):
        for t in range(Nt):
            update_fields(φ, A, J, dx, dt)
            refractive_indices[i, t] = calculate_refractive_index(φ, A, k, dt*t, x, dx)

    # Plot the refractive index as a function of time
    for i, k in enumerate(k_values):
        plt.plot(np.arange(Nt)*dt, refractive_indices[i, :], label=f'k = {k:.3f}')

    plt.xlabel('Time')
    plt.ylabel('Refractive Index')
    plt.legend()
    plt.show()

    print("hasta ahora todo jevi")

	#main function

if __name__ == '__main__':
	main()
      
