import numpy as np
import matplotlib.pyplot as plt
import ufl
from dolfinx import fem, io, mesh, log
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, meshtags
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import argparse

# -------------------------
# MPI and mesh loading
# -------------------------
mesh_filename = f"SavedFiles/Mesh/square_two_holes_{MPI.COMM_WORLD.rank}.xdmf"

with XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
    domain = xdmf.read_mesh(name="disk")
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    cell_tags = xdmf.read_meshtags(mesh, name="disk_cells")
    facet_tags = xdmf.read_meshtags(mesh, name="disk_facets")


# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Process command-line arguments for Fenics2.py")
parser.add_argument("filename", type=str, help="File name of the results")
parser.add_argument("cmro2_1", type=float, help="CMRO2 for Hole 1 (μmol.cm-3.min-1)")
parser.add_argument("cmro2_2", type=float, help="CMRO2 for Hole 2 (μmol.cm-3.min-1)")
parser.add_argument("marker1", type=int, help="Facet marker for Hole 1")
parser.add_argument("marker2", type=int, help="Facet marker for Hole 2")
parser.add_argument("P_ves", type=float, help="Inner Boundary")
args = parser.parse_args()

# -------------------------
# Constants
# -------------------------
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)

filename = args.filename
M1 = args.cmro2_1 / cmro2_by_M
M2 = args.cmro2_2 / cmro2_by_M

# Overwrite values near Hole 2 (marker2)
marker1 = args.marker1
marker2 = args.marker2

P_ves = args.P_ves
p50_val = 10.
rin = 10.
r0 = 80.
rout = 80.

pixel_size = 10.0

# -------------------------
# Function Space and Constants
# -------------------------
V = fem.functionspace(domain, ("CG", 1))
uh = fem.Function(V)
v = ufl.TestFunction(V)

# Interpolate into Function
M_func = fem.Function(V)

M_func.x.array[:] = M1 # default

# Identify facets and dofs for both holes
holes2_facets = np.where(facet_tags.values == marker2)[0]
hole2_dofs = fem.locate_dofs_topological(V, tdim - 1, facet_tags.indices[hole2_facets])

with M_func.vector.localForm() as loc:
    loc.setValues(hole2_dofs, np.full(len(hole2_dofs), M2))

# -------------------------
# Dirichlet BC on inner boundary
# -------------------------
uD1 = fem.Function(V)
uD2 = fem.Function(V)

def uD_expr_1(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    r = np.maximum(r, 1e-8)  # Avoid log(0)
    return P_ves + (M1 / 4) * ((r**2 - rin**2) - 2 * r0**2 * np.log(r / rin))

def uD_expr_2(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    r = np.maximum(r, 1e-8)
    return P_ves + (M2 / 4) * ((r**2 - rin**2) - 2 * r0**2 * np.log(r / rin))

uD1.interpolate(uD_expr_1)
uD2.interpolate(uD_expr_2)

facets1 = np.where(facet_tags.values == marker1)[0]
facets2 = np.where(facet_tags.values == marker2)[0]

dofs1 = fem.locate_dofs_topological(V, tdim - 1, facet_tags.indices[facets1])
dofs2 = fem.locate_dofs_topological(V, tdim - 1, facet_tags.indices[facets2])

bc1 = fem.dirichletbc(uD1, dofs1)
bc2 = fem.dirichletbc(uD2, dofs2)

# -------------------------
# Neumann BC on outer boundary
# -------------------------
g = fem.Function(V)
g.interpolate(lambda x: 0.0)

def neumann_flux(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.where(np.isclose(r, 60., atol=1.), 1., 0.)

# -------------------------
# Weak form
# -------------------------
# M(p) = M * p / (p50 + p)
M = M_func
f = M
F = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# -------------------------
# Newton solver
# -------------------------
problem = NonlinearProblem(F, uh, bcs=[bc1, bc2])
solver = NewtonSolver(comm, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-6
solver.report = True

n, converged = solver.solve(uh)
uh.x.scatter_forward()

if not converged:
    raise RuntimeError(f"Newton solver failed to converge in {n} iterations.")
else:
    print(f"Newton solver converged in {n} iterations.")

# -------------------------
# Compute gradient field (NEW ADDITION)
# -------------------------
V_vec = fem.functionspace(domain, ("CG", 1), shape=(domain.geometry.dim,))  # Vector function space
grad_proj = fem.Function(V_vec)  # Projected gradient field

# Create expression for gradient and interpolate
grad_expr = ufl.grad(uh)
grad_proj_expr = fem.Expression(grad_expr, V_vec.element.interpolation_points())
grad_proj.interpolate(grad_proj_expr)

# -------------------------
# Save solution
# -------------------------
with XDMFFile(comm, filename, "w") as file:
    file.write_mesh(domain)
    file.write_function(uh)
    file.write_function(grad_proj)

np.save("diffusion_results_multiples_holes.npy", uh.x.array)
np.save("diffusion_results_multiples_holes_axis.npy", domain.geometry.x)
