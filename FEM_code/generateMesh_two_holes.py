import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import CellType, meshtags
from Core.Mesh import plot_mesh_with_physical_groups, create_mesh
import pyvista

def create_mesh_with_tags():
    
    # -------------------------
    #  Step 1: Initialize Gmsh model
    # -------------------------
    gmsh.initialize()
    model = gmsh.model()
    # Set parameters
    name = "SquareWithTwoHoles"
    hole_radius = 10.
    domain_size = 200.
    element_size = 5  # Default coarse mesh size
    inner_refined_size = 2  # Finer elements near hole boundary
    hole1_center = (-50, 0, 0)
    hole2_center = (50, 0, 0)

    # -------------------------
    # Step 2: Create geometry
    # -------------------------
    model.add(name)
    model.setCurrent(name)

    # -------------------------
    # Step 3: GMSH Geometry Creation
    # -------------------------
    # Square Domain with two holes
    square = model.occ.addRectangle(-domain_size / 2, -domain_size / 2, 0, domain_size, domain_size)
    hole1 = model.occ.addDisk(*hole1_center, hole_radius, hole_radius)
    hole2 = model.occ.addDisk(*hole2_center, hole_radius, hole_radius)
    domain_with_holes, _ = model.occ.cut([(2, square)], [(2, hole1), (2, hole2)])
    model.occ.synchronize()
    # Physical Groups
    main_surface = model.addPhysicalGroup(2, [domain_with_holes[0][1]], 1)
    model.setPhysicalName(2, main_surface, "MainSurface")

    # -------------------------
    # Step 4: Boundary Identification
    # -------------------------
    # Get boundary curves and classify them
    curves = model.getBoundary(domain_with_holes, combined=False, oriented=False)
    boundaries = {"hole1": [], "hole2": [], "outer": []}

    # Boundary identification
    tolerance = hole_radius * 0.5
    for dim, tag in curves:
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if (x - hole1_center[0])**2 + (y - hole1_center[1])**2 < tolerance**2:
            boundaries["hole1"].append(tag)
        elif (x - hole2_center[0])**2 + (y - hole2_center[1])**2 < tolerance**2:
            boundaries["hole2"].append(tag)
        else:
            boundaries["outer"].append(tag)
    
    # Assign Physical Groups for holes and outer boundary
    outer_boundary = model.addPhysicalGroup(1, boundaries["outer"], 2)
    hole1_boundary = model.addPhysicalGroup(1, boundaries["hole1"], 3)
    hole2_boundary = model.addPhysicalGroup(1, boundaries["hole2"], 4)
    model.setPhysicalName(1, outer_boundary, "OuterBoundary")
    model.setPhysicalName(1, hole1_boundary, "Hole1Boundary")
    model.setPhysicalName(1, hole2_boundary, "Hole2Boundary")

    # -------------------------
    # Step 5: Mesh Refinement
    # -------------------------
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", boundaries["hole1"] + boundaries["hole2"])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", inner_refined_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", element_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", hole_radius/2)
    gmsh.model.mesh.field.setNumber(2, "DistMax", hole_radius*3)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # -------------------------
    # Step 6: Generate mesh
    # -------------------------
    model.occ.synchronize()
    model.mesh.generate(2)

    # Convert to DOLFINx mesh
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)

    # -------------------------
    # Step 7: Save mesh
    # -------------------------
    mesh_filename = f"SavedFiles/Mesh/square_two_holes_{MPI.COMM_WORLD.rank}.xdmf"
    create_mesh(MPI.COMM_SELF, model, "disk", mesh_filename, "w")

    # Step 8: Visualize the refined mesh
    comm = MPI.COMM_WORLD

    with XDMFFile(comm, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="disk")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        cell_tags = xdmf.read_meshtags(mesh, name="disk_cells")
        facet_tags = xdmf.read_meshtags(mesh, name="disk_facets")
        
    plot_mesh_with_physical_groups(
        mesh,
        facet_tags,
        inner_marker=3, # Hole1
        outer_marker=2,
        additional_markers=[4], # Hole2
        filename="Mesh_figure"
    )

    gmsh.finalize()
    return mesh, cell_tags, facet_tags
    
if __name__ == "__main__":
    mesh, cell_tags, facet_tags = create_mesh_with_tags()
    print("Mesh with two holes created and saved successfully.")
    print(f"Mesh saved to: SavedFiles/Mesh/square_two_holes_{MPI.COMM_WORLD.rank}.xdmf")
    print(f"Cell tags: {cell_tags}")
    print(f"Facet tags: {facet_tags}")
    print("Mesh visualization saved as Mesh_figure.pdf")
    print("Run the solver with the generated mesh.")