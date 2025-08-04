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
from scipy.interpolate import griddata
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
        model.add("DomainWitHole")
        
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

        domain_with_hole, _ = model.occ.cut([(2, square)], hole_objects)
        model.occ.synchronize()

        # Add physical group for main surface
        model.addPhysicalGroup(2, [domain_with_hole[0][1]], 1)

        # Identify boundaries
        curves = model.getBoundary(domain_with_hole, combined=False, oriented=False)
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
        plot_facets_by_marker(self.params.marker, "blue", f"Hole1 Dirichlet (marker {self.params.marker})")

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
        self.V_vec = basix.ufl.element(family="Lagrange", cell="triangle", degree=1, discontinuous=False, shape=(2,))

        
        # Solution and test functions
        self.uh = fem.Function(self.V)
        self.v = ufl.TestFunction(self.V)
        
        # Initialize M function
        self.M_func = fem.Function(self.V)

        domain = self.domain
        tdim = domain.topology.dim
        fdim = tdim - 1
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
                self.M_func.x.array[hole_dofs] = params.__getattribute__(f'M')
        
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
        uD = fem.Function(self.V)
        
        # Boundary expressions
        def uD_expr(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            r = np.maximum(r, 1e-8)
            return self.params.Pves + (self.params.M / 4) * (
                (r**2 - self.params.Rves**2) - 2 * self.params.R0**2 * np.log(r / self.params.Rves))
            
        uD.interpolate(uD_expr)
        
        # Locate boundary dofs
        facets = np.where(self.facet_tags.values == self.params.marker)[0]
        
        dofs = fem.locate_dofs_topological(
            self.V, self.domain.topology.dim-1, 
            self.facet_tags.indices[facets]
        )
        
        # Create BCs
        self.bcs = [
            fem.dirichletbc(uD, dofs),
        ]
        
    def setup_neumann_bcs(self, holes):
        """Set up no-flux boundary conditions on circles around holes"""
        domain = self.domain
        tdim = domain.topology.dim
        fdim = tdim - 1
        tolerance = 1.1
        domain.topology.create_connectivity(fdim, 0)  # Needed to get vertices of each facet

        x = domain.geometry.x  # Nodal coordinates
        all_neumann_facets = []
        all_neumann_values = []

        for hole in holes:
            center = np.array(hole.center[:2])
            R0 = hole.radius_0
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

                if np.abs(dist - R0) < tolerance:
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

    def interpolation_grid(self, grid_refined, grid_coarse):
        def find_closest_point(given_point, array_of_points):
            # Use Euclidean distance for multi-dimensional points
            distances = np.abs(array_of_points - given_point)
            closest_index = np.argmin(distances)
            return closest_index

        idx_list = []
        for point_coarse in grid_coarse:
            closest_idx = find_closest_point(point_coarse, grid_refined)
            idx_list.append(closest_idx)
        
        return np.array(idx_list)
    
    def solve(self):
        """Solve the nonlinear problem"""
        problem = NonlinearProblem(self.F, self.uh, bcs=self.bcs)
        solver = NewtonSolver(self.comm, problem)
        solver.convergence_criterion = "residual"
        solver.rtol = 1e-6
        
        n, converged = solver.solve(self.uh)
        self.uh.x.scatter_forward()

        uh = self.uh.x.array
        domain_coordinate = self.domain.geometry.x
        x = np.array(domain_coordinate[:, 0])
        y = np.array(domain_coordinate[:, 1])
        
        # -------------------------
        # Interpolate to observation grid
        # -------------------------
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Create observation grid points
        x_obs = np.linspace(x_min, x_max, n)
        y_obs = np.linspace(y_min, y_max, n)
        
        # Create simulation grid
        x_idx_domain = self.interpolation_grid(x, x_obs)
        y_idx_domain = self.interpolation_grid(y, y_obs)
        x_domain = x[x_idx_domain]
        y_domain = y[y_idx_domain]
        X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
        points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))

        # Evaluate FEM solution at observation points
        # Interpolate z values at the grid points
        self.uh_nbyn = griddata((x, y), uh, points, method='linear')
        
        if not converged:
            raise RuntimeError(f"Solver failed to converge in {n} iterations")
        return converged
        
    def save_results(self, filename):
        """Save solution to XDMF file"""
        
        params = self.params
        path = params.path

        # Create gradient function to vector space 
        V_vec = fem.functionspace(self.domain, ("CG", 1))
        
        # Save gradient to XDMF
        with io.XDMFFile(self.comm, path + f"{filename}.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.domain)
            xdmf.write_function(self.uh)

        # Save raw data for post-processing
        np.save(path + f"{filename}_solution.npy", self.uh.x.array)
        np.save(path + f"{filename}_coordinates.npy", self.domain.geometry.x)

class SolverParameters:
    """Container for solver parameters"""
    def __init__(self, filename, cmro2, Pves, Rves, R0, marker=3):
        self.path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/"
        self.filename = filename
        self.cmro2 = -cmro2  # CMRO2 for the hole
        self.marker = marker
        self.Pves = Pves  # Pressure at the vessel wall
        
        # Derived quantities
        self.M = self.cmro2 / (60 * 4.0e3 * 1.39e-15 * 1e12)
        
        # Geometry parameters
        self.Rves = Rves
        self.R0 = R0



def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    
    # Create solver parameters
    params = SolverParameters(filename="square_one_hole_id0", cmro2=-1, Pves=75., Rves=10., R0=80.)
    
    # Create solver instance
    solver = DiffusionSolver(comm)
    
    # Define holes
    holes = [
        HoleGeometry(center=(0, 0, 0), radius_ves=params.Rves, radius_0=params.R0, marker=params.marker),
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