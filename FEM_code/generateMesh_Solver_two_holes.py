import numpy as np
import gmsh
import ufl
import basix
import dolfinx.plot
from dolfinx import fem, io, mesh
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from Core.Mesh import plot_mesh_with_physical_groups, create_mesh
import pyvista

class HoleGeometry:
    """Represents a circular hole in the domain"""
    def __init__(self, center, radius_ves, radius_0, marker):
        self.center = center
        self.radius_ves = radius_ves
        self.radius_0 = radius_0
        self.marker = marker



class DiffusionSolver:
    """Solves nonlinear diffusion problem with multiple holes"""
    
    def __init__(self, comm):
        self.comm = comm
        self.domain = None
        self.facet_tags = None
        self.cell_tags = None
        
        # Physical constants
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = (60 * self.D * self.alpha * 1e12)
        
    def generate_mesh(self, holes, domain_size=200.0, element_size=5.0, refined_size=2.0):
        """Generate mesh with refined areas around holes"""
        gmsh.initialize()
        model = gmsh.model()
        model.add("DomainWithHoles")
        
        # Create main domain
        square = model.occ.addRectangle(
            -domain_size/2, -domain_size/2, 0, 
            domain_size, domain_size
        )
        
        # Create holes and subtract from domain
        hole_objects = []
        for hole in holes:
            disk = model.occ.addDisk(*hole.center, hole.radius_ves, hole.radius_ves)
            hole_objects.append((2, disk))
        
        domain_with_holes, _ = model.occ.cut([(2, square)], hole_objects)
        model.occ.synchronize()
        
        # Add physical group for main surface
        model.addPhysicalGroup(2, [domain_with_holes[0][1]], 1)
        
        # Identify boundaries
        curves = model.getBoundary(domain_with_holes, combined=False, oriented=False)
        boundaries = {hole.marker: [] for hole in holes}
        boundaries["outer"] = []
        
        for dim, tag in curves:
            x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            found = False
            for hole in holes:
                if (x - hole.center[0])**2 + (y - hole.center[1])**2 < (hole.radius_ves * 0.5)**2:
                    boundaries[hole.marker].append(tag)
                    found = True
                    break
            if not found:
                boundaries["outer"].append(tag)
        
        # Create physical groups
        for hole in holes:
            model.addPhysicalGroup(1, boundaries[hole.marker], hole.marker)
        model.addPhysicalGroup(1, boundaries["outer"], 99)  # Outer boundary marker
        
        # Mesh refinement near holes
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", 
            [tag for hole in holes for tag in boundaries[hole.marker]])
        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", refined_size)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", element_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", min(h.radius_ves for h in holes)/2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", max(h.radius_ves for h in holes)*3)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        
        # Generate mesh
        model.mesh.generate(2)
        
        # Convert to DOLFINx mesh
        self.domain, self.cell_tags, self.facet_tags = gmshio.model_to_mesh(
            model, self.comm, 0, gdim=2
        )

        # Finalize GMSH
        gmsh.finalize()
    
    def plot_boundaries(self, holes):
        """Plot mesh and boundary markers using PyVista (Dirichlet, Neumann, Ring Neumann)"""

        # Create main mesh grid
        grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(self.domain, self.domain.topology.dim))
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, color="white", show_edges=True, opacity=0.3, label="Mesh")

        # Helper to plot facet markers 
        def plot_facets_by_marker(marker, color, label):
            try:
                facets = self.facet_tags.find(marker)
                if len(facets) == 0:
                    return
                facet_grid = pyvista.UnstructuredGrid(
                    *dolfinx.plot.vtk_mesh(self.domain, self.domain.topology.dim - 1, facets)
                )
                plotter.add_mesh(facet_grid, color=color, label=label, line_width=4)
            except KeyError:
                pass

        # Plot Dirichlet boundaries (hole boundaries)
        plot_facets_by_marker(self.params.marker1, "blue", f"Hole1 Dirichlet (marker {self.params.marker1})")
        plot_facets_by_marker(self.params.marker2, "green", f"Hole2 Dirichlet (marker {self.params.marker2})")

        # Outer boundary
        # plot_facets_by_marker(99, "red", "Outer Neumann (marker 99)")

        # Ring Neumann boundaries (from setup_circular_neumann_bcs)
        for hole in holes:
            marker = 100 + hole.marker
            plot_facets_by_marker(marker, "red", f"Neumann Ring (marker {marker})")

        plotter.add_legend()
        plotter.show()

    def setup_problem(self, params, holes):
        """Set up the finite element problem"""
        # Store parameters
        self.params = params
        
        # Function space
        self.V = fem.functionspace(self.domain, ("CG", 1))
        
        # Solution and test functions
        self.uh = fem.Function(self.V)
        self.v = ufl.TestFunction(self.V)
        
        # Initialize M function
        self.M_func = fem.Function(self.V)

        domain = self.domain
        tdim = domain.topology.dim
        fdim = tdim - 1
        tolerance = 1.0
        domain.topology.create_connectivity(fdim, tdim)
        domain.topology.create_connectivity(fdim, 0)  # Needed to get vertices of each facet
        x = domain.geometry.x  # Nodal coordinates
        
        
        inner_facets = []
        inner_values = []
        num_facets = domain.topology.index_map(fdim).size_local # Get all local facets
        for hole in holes:

            center=hole.center
            rin = hole.radius_ves
            r0 = hole.radius_0
            marker = 200 + hole.marker
            for f in range(num_facets):
                    vertex_ids = domain.topology.connectivity(fdim, 0).links(f)
                    coords = x[vertex_ids]
                    midpoint = np.mean(coords, axis=0)

                    # Compute distance from center in XY plane
                    dx = midpoint[0] - center[0]
                    dy = midpoint[1] - center[1]
                    dist = np.sqrt(dx**2 + dy**2)

                    if r0 > dist > rin:
                        inner_facets.append(f)
                        inner_values.append(marker)

            all_indices = np.array(inner_facets, dtype=np.int32)
            all_values = np.array(inner_values, dtype=np.int32)

            # Set M values for each hole
            for i, hole in enumerate(holes):
                hole_facets = all_indices[np.array(all_values) == (200 + hole.marker)]
                hole_dofs = fem.locate_dofs_topological(
                    self.V, self.domain.topology.dim-1, 
                    hole_facets
                )
                self.M_func.x.array[hole_dofs] = params.__getattribute__(f'M{i+1}')
        
        # Boundary conditions
        self.setup_dirichlet_bcs()
        
        # Define circular Neumann boundaries
        self.setup_neumann_bcs(holes)

        # Weak form
        self.F = ufl.dot(ufl.grad(self.uh), ufl.grad(self.v)) * ufl.dx - \
                self.M_func * self.v * ufl.dx

        # Add Neumann terms to weak form
        self.F += sum(
            ufl.inner(ufl.grad(self.uh), self.v * ufl.FacetNormal(self.domain)) * ufl.ds(
                domain=self.domain, 
                subdomain_data=self.facet_tags, 
                subdomain_id=100+hole.marker)
                for hole in holes
        )
        
    def setup_dirichlet_bcs(self):
        """Configure Dirichlet boundary conditions"""
        # Create boundary functions
        uD1 = fem.Function(self.V)
        uD2 = fem.Function(self.V)
        
        # Boundary expressions
        def uD_expr_1(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            r = np.maximum(r, 1e-8)
            return self.params.P_ves_1 + (self.params.M1 / 4) * (
                (r**2 - self.params.rin_1**2) - 2 * self.params.r0_1**2 * np.log(r / self.params.rin_1))
            
        def uD_expr_2(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            r = np.maximum(r, 1e-8)
            return self.params.P_ves_2 + (self.params.M2 / 4) * (
                (r**2 - self.params.rin_2**2) - 2 * self.params.r0_2**2 * np.log(r / self.params.rin_2))
        
        uD1.interpolate(uD_expr_1)
        uD2.interpolate(uD_expr_2)
        
        # Locate boundary dofs
        facets1 = np.where(self.facet_tags.values == self.params.marker1)[0]
        facets2 = np.where(self.facet_tags.values == self.params.marker2)[0]
        
        dofs1 = fem.locate_dofs_topological(
            self.V, self.domain.topology.dim-1, 
            self.facet_tags.indices[facets1]
        )
        dofs2 = fem.locate_dofs_topological(
            self.V, self.domain.topology.dim-1, 
            self.facet_tags.indices[facets2]
        )
        
        # Create BCs
        self.bcs = [
            fem.dirichletbc(uD1, dofs1),
            fem.dirichletbc(uD2, dofs2)
        ]
        
    def setup_neumann_bcs(self, holes):
        """Set up no-flux boundary conditions on circles around holes"""
        domain = self.domain
        tdim = domain.topology.dim
        fdim = tdim - 1
        tolerance = 1.0
        domain.topology.create_connectivity(fdim, 0)  # Needed to get vertices of each facet

        x = domain.geometry.x  # Nodal coordinates
        all_neumann_facets = []
        all_neumann_values = []

        for hole in holes:
            center = np.array(hole.center[:2])
            r0 = hole.radius_0
            marker = 100 + hole.marker
            num_facets = domain.topology.index_map(fdim).size_local # Get all local facets

            for f in range(num_facets):
                vertex_ids = domain.topology.connectivity(fdim, 0).links(f) # example array([17, 88])
                coords = x[vertex_ids] # dim 2-by-2 array of vertex coordinates
                midpoint = np.mean(coords, axis=0)

                # Compute distance from center in XY plane
                dx = midpoint[0] - center[0]
                dy = midpoint[1] - center[1]
                dist = np.sqrt(dx**2 + dy**2)

                if np.abs(dist - r0) < tolerance:
                    all_neumann_facets.append(f)
                    all_neumann_values.append(marker)

        # Merge with existing facet_tags (if any)
        if self.facet_tags is not None:
            old_facets = self.facet_tags.indices
            old_values = self.facet_tags.values
            all_indices = np.concatenate([old_facets, np.array(all_neumann_facets, dtype=np.int32)])
            all_values = np.concatenate([old_values, np.array(all_neumann_values, dtype=np.int32)])
        else:
            all_indices = np.array(all_neumann_facets, dtype=np.int32)
            all_values = np.array(all_neumann_values, dtype=np.int32)

        self.facet_tags = mesh.meshtags(domain, fdim, all_indices, all_values)

        mask = self.facet_tags.values != 99
        new_indices = self.facet_tags.indices[mask]
        new_values = self.facet_tags.values[mask]
        self.facet_tags = mesh.meshtags(self.domain, self.domain.topology.dim - 1, new_indices, new_values)


    def solve(self):
        """Solve the nonlinear problem"""
        problem = NonlinearProblem(self.F, self.uh, bcs=self.bcs)
        solver = NewtonSolver(self.comm, problem)
        solver.convergence_criterion = "residual"
        solver.rtol = 1e-6
        
        n, converged = solver.solve(self.uh)
        self.uh.x.scatter_forward()
        
        if not converged:
            raise RuntimeError(f"Solver failed to converge in {n} iterations")
        return converged
        
    def save_results(self, filename):
        """Save solution to XDMF file"""
        
        params = self.params
        path = params.path
        
        # Save the solution to XDMF
        with io.XDMFFile(self.comm, path + f"{filename}.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.uh)

        # Save raw data for post-processing
        np.save(path + f"{filename}_solution.npy", self.uh.x.array)
        np.save(path + f"{filename}_coordinates.npy", self.domain.geometry.x)

        

class SolverParameters:
    """Container for solver parameters"""
    def __init__(self):
        self.path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/"
        self.filename = "square_two_holes"
        self.cmro2_1 = -1.  # CMRO2 for Hole 1
        self.cmro2_2 = -1.  # CMRO2 for Hole 2
        self.marker1 = 3
        self.marker2 = 4
        self.P_ves_1 = 70.0  # Pressure at the vessel wall 1
        self.P_ves_2 = 70.0  # Pressure at the vessel wall 2
        
        # Derived quantities
        self.M1 = self.cmro2_1 / (60 * 4.0e3 * 1.39e-15 * 1e12)
        self.M2 = self.cmro2_2 / (60 * 4.0e3 * 1.39e-15 * 1e12)
        
        # Geometry parameters
        self.rin_1 = 10.0
        self.rin_2 = 10.0
        self.r0_1 = 50.0
        self.r0_2 = 50.0



def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    
    # Create solver parameters
    params = SolverParameters()
    
    # Create solver instance
    solver = DiffusionSolver(comm)
    
    # Define holes
    holes = [
        HoleGeometry(center=(-50, 0, 0), radius_ves=params.rin_1, radius_0=params.r0_1, marker=params.marker1),
        HoleGeometry(center=(50, 0, 0), radius_ves=params.rin_2, radius_0=params.r0_2, marker=params.marker2)
    ]
    
    # Generate mesh
    if comm.rank == 0:
        print("Generating mesh...")
    solver.generate_mesh(holes)
    
    # Set up and solve problem
    if comm.rank == 0:
        print("Setting up problem...")
    solver.setup_problem(params, holes)
    solver.plot_boundaries(holes)
    
    if comm.rank == 0:
        print("Solving nonlinear problem...")
    solver.solve()
    
    # Save results
    if comm.rank == 0:
        print("Saving results...")
    solver.save_results(params.filename)
    print(f"Solution saved to {params.filename}")
    print("Simulation completed successfully.")

if __name__ == "__main__":
    main()