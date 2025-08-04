from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import gmsh

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

def gmsh_annulus(model: gmsh.model, name: str, inner_radius, outer_radius, element_size) -> gmsh.model:
    """Create a Gmsh model of an annulus.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.
        inner_radius: float - inner radius of annulus.
        outer_radius: float - outer radius of annulus.
        element_size: float - size of the mesh elements.

    Returns:
        Gmsh model.

    """

    model.add(name)
    model.setCurrent(name)

    inner_disk = model.occ.addDisk(0, 0, 0, inner_radius, inner_radius)
    outer_disk = model.occ.addDisk(0, 0, 0, outer_radius, outer_radius)

    
    annulus, _ = model.occ.cut([(2, outer_disk)], [(2, inner_disk)])
    model.occ.synchronize()

    annulus_surface = model.addPhysicalGroup(2, [annulus[0][1]], 1)
    model.setPhysicalName(2, annulus_surface, "Annulus")
    
    outer_curve = model.getBoundary(annulus, combined=False, oriented=False)[1][1]
    inner_curve = model.getBoundary(annulus, combined=False, oriented=False)[0][1]
    
    outer_boundary = model.addPhysicalGroup(1, [outer_curve], 3)
    model.setPhysicalName(1, outer_boundary, "outer")
    
    inner_boundary = model.addPhysicalGroup(1, [inner_curve], 2)
    model.setPhysicalName(1, inner_boundary, "inner")
    
    model.mesh.setSize(gmsh.model.getEntities(0), element_size)  # Set size for all points
    model.mesh.setSize(gmsh.model.getEntities(1), element_size)

    model.occ.synchronize()
    model.mesh.generate(dim=2)
    
    return model

def gmsh_disk(model: gmsh.model, name: str, radius, element_size) -> gmsh.model:
    """Create a Gmsh model of an disk.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.
        radius: float - radius of disk.
        element_size: float - size of the mesh elements.

    Returns:
        Gmsh model.

    """
    model.add(name)
    model.setCurrent(name)

    disk = model.occ.addDisk(0, 0, 0, radius, radius)
    model.occ.synchronize()

    disk_surface = model.addPhysicalGroup(2, [disk], 1)
    model.setPhysicalName(2, disk_surface, "Disk")

    outer_curve = model.getBoundary([(2, disk)], combined=False, oriented=False)[0][1]

    outer_boundary = model.addPhysicalGroup(1, [outer_curve], 2)
    model.setPhysicalName(1, outer_boundary, "outer")

    model.mesh.setSize(gmsh.model.getEntities(0), element_size)  # Set size for all points
    model.mesh.setSize(gmsh.model.getEntities(1), element_size)

    model.occ.synchronize()
    model.mesh.generate(dim=2)
    
    return model
    

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm: MPI communicator top create the mesh on.
        model: Gmsh model.
        name: Name (identifier) of the mesh to add.
        filename: XDMF filename.
        mode: XDMF file mode. "w" (write) or "a" (append).

    """
    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    with XDMFFile(msh.comm, filename, mode) as file:
        msh.topology.create_connectivity(1, 2)
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )


def plot_mesh_with_physical_groups(mesh, facet_tags, inner_marker=None, additional_markers=None, outer_marker=None, filename=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    x = mesh.geometry.x
    tdim = mesh.topology.dim

    cells = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 3)

    triang = Triangulation(x[:, 0], x[:, 1], cells)
    ax.triplot(triang, color="gray", alpha=0.7, label="Mesh")
    labels_used = set()
    
    def plot_facets(marker, color,  label):
        if marker is None:
            return 
        try:
            facets = facet_tags.find(marker)
        except:
            return # in case marker not found
        for facet in facets:
            facet_vertices = mesh.topology.connectivity(1, 0).links(facet)
            ax.plot(x[facet_vertices, 0], x[facet_vertices, 1],
                   color=color, lw=2,
                   label=label if label not in labels_used else "")
            labels_used.add(label)
        
    # Plot known mreker not found
    plot_facets(inner_marker, "blue", f"Inner Boundary {inner_marker}")
    plot_facets(outer_marker, "red", f"Outer Boundary {outer_marker}")
    
    if additional_markers:
        inner_facets = facet_tags.find(inner_marker)
        additional_facets = facet_tags.find(additional_markers[0])
        outer_facets = facet_tags.find(outer_marker)
        for i, marker in enumerate(additional_markers):
            plot_facets(marker, "green", f"Inner Boundary {marker}")
            

    ax.set_title("Mesh with Physical Groups")
    ax.legend()
    ax.set_aspect("equal")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(False)
    if filename:
        plt.savefig(filename + ".pdf", dpi=600)    
    plt.show()
     