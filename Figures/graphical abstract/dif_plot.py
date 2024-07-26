import fipy as fp
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the 2D domain
length = 10.0
width = 10.0
nx = 50
ny = 50
dx = length / nx
dy = width / ny
D = 1.0  # Diffusion coefficient
total_time = 10.0  # Total time for simulation
time_step = 0.1  # Time step for the solver

# Create a 2D mesh
mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

# Initialize variables for six cytokines
cytokines = []
initial_conditions = [
    (2.5, 2.5), (7.5, 2.5), (2.5, 7.5), (7.5, 7.5), (5.0, 5.0), (3.0, 7.0)
]

for initial in initial_conditions:
    c = fp.CellVariable(name="cytokine", mesh=mesh, value=0.0)
    c.setValue(1.0, where=(mesh.x > initial[0] - 0.5) & (mesh.x < initial[0] + 0.5) &
                       (mesh.y > initial[1] - 0.5) & (mesh.y < initial[1] + 0.5))
    cytokines.append(c)

# Define the diffusion equation
diffusion_eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D)

# Time-stepping loop
time = 0.0
while time < total_time:
    for c in cytokines:
        diffusion_eq.solve(var=c, dt=time_step)
    time += time_step

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))
X, Y = np.meshgrid(np.linspace(0, length, nx), np.linspace(0, width, ny))

# Define a colormap
colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm']
for idx, c in enumerate(cytokines):
    cs = ax.contourf(X, Y, c.value.reshape((nx, ny)), 20, cmap=colors[idx], alpha=0.5)
    # fig.colorbar(cs, ax=ax, shrink=0.5)

ax.axis('off')  # Remove the axes
# plt.title('2D Diffusion of Six Cytokines')
plt.show()
