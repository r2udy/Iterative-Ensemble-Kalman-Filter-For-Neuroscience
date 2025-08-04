import pytest
import numpy as np
from mpi4py import MPI
from generateMesh_Solver import DiffusionSolver, HoleGeometry, SolverParameters

@pytest.fixture
def solver_with_mesh():
    comm = MPI.COMM_WORLD
    solver = DiffusionSolver(comm)
    params = SolverParameters()
    holes = [
        HoleGeometry(center=(-50, 0, 0), radius=10.0, marker=params.marker1),
        HoleGeometry(center=(50, 0, 0), radius=10.0, marker=params.marker2)
    ]
    solver.generate_mesh(holes)
    return solver, params

def test_setup_problem_initializes_function_space(solver_with_mesh):
    solver, params = solver_with_mesh
    solver.setup_problem(params)
    assert solver.V is not None
    assert hasattr(solver, "uh")
    assert hasattr(solver, "v")

def test_setup_problem_M_func_values(solver_with_mesh):
    solver, params = solver_with_mesh
    solver.setup_problem(params)
    # Check default M1 value
    assert np.allclose(solver.M_func.x.array, params.M1, atol=1e-12) or np.any(
        np.isclose(solver.M_func.x.array, params.M2, atol=1e-12)
    )
    # Check that at least some dofs have M2 value
    assert np.any(np.isclose(solver.M_func.x.array, params.M2, atol=1e-12))

def test_setup_problem_sets_boundary_conditions(solver_with_mesh):
    solver, params = solver_with_mesh
    solver.setup_problem(params)
    assert hasattr(solver, "bcs")
    assert isinstance(solver.bcs, list)
    assert len(solver.bcs) == 2

def test_setup_problem_weak_form_is_defined(solver_with_mesh):
    solver, params = solver_with_mesh
    solver.setup_problem(params)
    assert hasattr(solver, "F")
    assert solver.F is not None