import gmsh
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from Core.Mesh import plot_mesh_with_physical_groups, create_mesh

gmsh.initialize()
model = gmsh.model()

name = "Disk"
inner_radius = 0.5
outer_radius = 2
element_size = 0.1

model.add(name)
model.setCurrent(name)

disk = model.occ.addDisk(0, 0, 0, outer_radius, outer_radius)

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


model.setCurrent(name)
create_mesh(MPI.COMM_SELF, model, "disk", f"Core/Mesh/disk_mesh_{MPI.COMM_WORLD.rank}.xdmf", "w")
gmsh.finalize()


comm = MPI.COMM_WORLD
mesh_filename = f"Core/Mesh/disk_mesh_{MPI.COMM_WORLD.rank}.xdmf"

with XDMFFile(comm, mesh_filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="disk")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    cell_tags = xdmf.read_meshtags(mesh, name="disk_cells")
    facet_tags = xdmf.read_meshtags(mesh, name="disk_facets")
    
inner_marker = 2
outer_marker = 3 
plot_mesh_with_physical_groups(mesh, facet_tags, inner_marker, outer_marker, filename = "test")


gmsh.initialize()
model = gmsh.model()

name = "Disk"
inner_radius = 0.5
outer_radius = 2
element_size = 0.1


model.add(name)
model.setCurrent(name)

inner_disk = model.occ.addDisk(0, 0, 0, inner_radius, inner_radius)
outer_disk = model.occ.addDisk(0, 0, 0, outer_radius, outer_radius)

annulus, _ = model.occ.cut([(2, outer_disk)], [(2, inner_disk)])
model.occ.synchronize()

annulus_surface = model.addPhysicalGroup(2, [annulus[0][1]], 1)
model.setPhysicalName(2, annulus_surface, "Annulus")

inner_disk_surface = model.addPhysicalGroup(2, [inner_disk], 2)
model.setPhysicalName(2, inner_disk_surface, "InnerDisk")

model.occ.synchronize()

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
model.setCurrent(name)
create_mesh(MPI.COMM_SELF, model, "disk", f"SavedFiles/Mesh/disk_mesh_{MPI.COMM_WORLD.rank}.xdmf", "w")
comm = MPI.COMM_WORLD
mesh_filename = f"SavedFiles/Mesh/disk_mesh_{MPI.COMM_WORLD.rank}.xdmf"

with XDMFFile(comm, mesh_filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="disk")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    cell_tags = xdmf.read_meshtags(mesh, name="disk_cells")
    facet_tags = xdmf.read_meshtags(mesh, name="disk_facets")
    
inner_marker = 2
outer_marker = 3 
plot_mesh_with_physical_groups(mesh, facet_tags, inner_marker, outer_marker, filename = "Mesh_figure")
