import numpy as np
import matplotlib.pyplot as plt
import ufl
import dolfinx
from dolfinx import fem, io, mesh, log
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import argparse

# -------------------------
# MPI and mesh loading
# -------------------------
comm = MPI.COMM_WORLD
path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/"        
mesh_filename = path + f"disk_mesh_{comm.rank}.xdmf"

with XDMFFile(comm, mesh_filename, "r") as xdmf:
    domain = xdmf.read_mesh(name="disk")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Process command-line arguments for Fenics2.py")
parser.add_argument("filename", type=str, help="File name of the results")
parser.add_argument("cmro2", type=float, help="Consumption (μmol.cm-3.min-1)")
parser.add_argument("P_ves", type=float, help="Inner Boundary")
parser.add_argument("rin", type=float, help="radius in")
parser.add_argument("r0", type=float, help="radius 0")
parser.add_argument("rout", type=float, help="radius out")
args = parser.parse_args()

# -------------------------
# Constants
# -------------------------
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)

filename = args.filename
M_val = args.cmro2 / cmro2_by_M
P_ves = args.P_ves
p50_val = 10.
rin = args.rin
r0 = args.r0
rout = args.rout

pixel_size = 10.0
beta_val = (M_val / 2) * (rin**2 - r0**2)

# -------------------------
# Function Space and Constants
# -------------------------
V = fem.functionspace(domain, ("CG", 1))
V_vec = fem.functionspace(domain, ("CG", 1), shape=(domain.geometry.dim,)) 

uh = fem.Function(V)
grad_expr = ufl.grad(uh)
grad_proj = fem.Function(V_vec)
grad_problem = fem.petsc.LinearProblem(
    ufl.inner(fem.TestFunction)
)
v = ufl.TestFunction(V)
M = fem.Constant(domain, ScalarType(M_val))
p50 = fem.Constant(domain, ScalarType(p50_val))
beta = fem.Constant(domain, ScalarType(beta_val))

# -------------------------
# Dirichlet BC on inner boundary
# -------------------------
uD = fem.Function(V)

def uD_expr(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    r = np.maximum(r, 1e-8)  # Avoid log(0)
    return P_ves + (M_val / 4) * ((r**2 - rin**2) - 2 * rin**2 * np.log(r / rin)) + beta_val * np.log(r / rin)

uD.interpolate(uD_expr)

def inner_boundary(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.isclose(r, rin, atol=1e-2)

boundary_facets = mesh.locate_entities_boundary(domain, fdim, inner_boundary)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# -------------------------
# Weak form
# -------------------------
# M(p) = M * p / (p50 + p)
f = M * uh / (p50 + uh)
F = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# -------------------------
# Newton solver
# -------------------------
problem = NonlinearProblem(F, uh, bcs=[bc])
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
# Save solution
# -------------------------
with XDMFFile(comm, filename, "w") as file:
    file.write_mesh(domain)
    file.write_function(uh)

np.save("diffusion_results.npy", uh.x.array)
np.save("diffusion_results_axis.npy", domain.geometry.x)
