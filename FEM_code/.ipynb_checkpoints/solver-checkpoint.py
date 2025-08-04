import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx

import matplotlib.pyplot as plt

import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
comm = MPI.COMM_WORLD
mesh_filename = f"SavedFiles/Mesh/disk_mesh_{MPI.COMM_WORLD.rank}.xdmf"
with XDMFFile(comm, mesh_filename, "r") as xdmf:
    domain = xdmf.read_mesh(name="disk")
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    domain_ct = xdmf.read_meshtags(domain, name="disk_cells")
    domain_ft = xdmf.read_meshtags(domain, name="disk_facets")
V = fem.functionspace(domain, ("CG", 1))
def inner_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 0.5)

def outer_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 2.0)

# Time-dependent concentration at inner boundary
def concentration(t):
    return 1 + 1*(1 + np.sin(np.pi * t))
    # return 5
# Define boundary condition
inner_dofs = fem.locate_dofs_geometrical(V, inner_boundary)
outer_dofs = fem.locate_dofs_geometrical(V, outer_boundary)

# Initial concentration value
initial_concentration = fem.Constant(domain, ScalarType(0))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Previous solution
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.zeros_like(x[0]))
t = 0
dt = 0.1
T = 20.0
D = fem.Constant(domain, ScalarType(0.5))
r = fem.Constant(domain, ScalarType(-0.02))
a = u * v * ufl.dx + dt * D * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = u_n * v * ufl.dx + r * v * ufl.dx 
bc_inner = fem.dirichletbc(ScalarType(concentration(0)), inner_dofs, V)
bc_outer = fem.dirichletbc(ScalarType(0), outer_dofs, V)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_closest_point(given_point, array_of_points):
    array_of_points = np.array(array_of_points)  # Convert list to numpy array if it's not already
    distances = np.linalg.norm(array_of_points - given_point, axis=1)
    closest_index = np.argmin(distances)  # Index of the closest point
    return closest_index

points = [[0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0], [2.5, 0, 0]]
idx = []
for pt in points:
    idx.append(find_closest_point(pt, domain.geometry.x))

sol1 = []
sol2 = []
sol3 = []
sol4 = []

time_values = []
# plt.scatter(domain.geometry.x[:, 0], domain.geometry.x[:, 1])
# plt.scatter(domain.geometry.x[idx, 0], domain.geometry.x[idx, 1])
 
with io.XDMFFile(domain.comm, "diffusion_results.xdmf", "w") as file:
    file.write_mesh(domain)

    # Time-stepping loop
    while t < T:
        # Update time
        t += dt

        # Update boundary condition
        bc_inner = fem.dirichletbc(ScalarType(concentration(t)), inner_dofs, V)
        bc_outer = fem.dirichletbc(ScalarType(1), outer_dofs, V)

        # Set up and solve the problem
        problem = fem.petsc.LinearProblem(a, L, bcs=[bc_inner], u=u_n)
        u_n = problem.solve()

        sol1.append(u_n.x.array[idx[0]])
        sol2.append(u_n.x.array[idx[1]])
        sol3.append(u_n.x.array[idx[2]])
        sol4.append(u_n.x.array[idx[3]])

        time_values.append(t)

        # Write solution to file
        file.write_function(u_n, t)
 
# No reaction : Same fixed points
# 1. variation of concentration at some fixed point over time - constant release vs. Sinusoidal release
# 2. Constant release with and without consumption
# 3. Sinusoidal increasing with and without consumption
 
plt.figure(figsize=(10, 6))
plt.plot(time_values, sol1, linewidth=3, markersize=6, label = "A")
plt.plot(time_values, sol2, linewidth=3, markersize=6, label = "B")
plt.plot(time_values, sol3, linewidth=3, markersize=6, label = "C")
plt.plot(time_values, sol4, linewidth=3, markersize=6, label = "D")

# Add labels and title with increased font size

# Add grid for better visibility and beautification
plt.grid(True, linestyle='--', alpha=0.7)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend()

# Save the figure
plt.savefig('SinInlet5_conc_1_1.pdf', dpi = 600)

# Display the plot
plt.show()