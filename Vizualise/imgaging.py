import scipy.io as sio
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage.filters import gaussian, frangi, threshold_otsu
from skimage.filters import threshold_triangle
from skimage.morphology import disk, opening
from skimage.morphology import remove_small_objects, binary_closing
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np


# Load MATLAB file
mat_file = sio.loadmat('/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/TODEsource/dbase/03/po2/pO2-run02.mat', squeeze_me=True, struct_as_record=False)

# Access variables from the MATLAB file
pO2_struct = mat_file['pO2'] # Assuming pO2 is a 2D array

fitc_img = pO2_struct.references[1].image  # Access the fitc_img variable

# Denoise the image using Gaussian filter
fitc_img_smooth = gaussian(fitc_img, sigma=1.5) 
po2_coords = np.vstack((pO2_struct.pointsX, pO2_struct.pointsY)).T  # Access the po2_coords variable



# Apply Frangi filter to enhance vessel-like structures
vesselness = frangi(fitc_img_smooth, scale_range=(1, 6), scale_step=2)

# Threshold the vesselness image to create a binary mask
binary_mask = vesselness > threshold_otsu(vesselness)

# Strong smoothing 
arteriole_img = gaussian(fitc_img, sigma=1.0)

# Threshold bright region
t = threshold_triangle(arteriole_img)
arteriole_mask = arteriole_img > t

# Clean
arteriole_mask = opening(arteriole_mask, disk(3))
arteriole_mask = remove_small_objects(arteriole_mask, min_size=100)
arteriole_mask = binary_closing(arteriole_mask, disk(3))

labels = label(arteriole_mask)
regions = regionprops(labels)

arteriole = max(
    regions,
    key=lambda r: r.equivalent_diameter
)

arteriole_radius = arteriole.equivalent_diameter / 2
capillary_mask = arteriole_mask.copy()
capillary_mask[labels == arteriole.label] = False

capillaries = []
for i, r in enumerate(regions):
    if r.label != arteriole.label and r.equivalent_diameter < arteriole_radius:
        capillaries.append({
            "cap_id": i,
            "center": r.centroid,
            "radius": r.equivalent_diameter / 2
        })

for cap in capillaries:
    cy, cx = cap["center"]
    radius = cap["radius"]
    y, x = np.ogrid[-cy:fitc_img.shape[0]-cy, -cx:fitc_img.shape[1]-cx]
    mask = x*x + y*y <= radius*radius
    capillary_mask[mask] = False

po2_points = []
for i, (x, y) in enumerate(po2_coords):
    po2_points.append({
        "obs_id": i,
        "coord": (x, y),
        "pO2": pO2_struct.pO2Value[i]
    })


# # Display the results
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# axes[0].imshow(fitc_img, cmap='gray')
# axes[0].set_title('Original FITC Image')
# axes[0].axis('off')
# axes[1].imshow(vesselness, cmap='gray')
# axes[1].set_title('Vesselness Image')
# axes[1].axis('off')
# axes[2].imshow(arteriole_mask, cmap='gray')
# axes[2].set_title('Cleaned Binary Mask')
# axes[2].axis('off')
# plt.tight_layout()
# plt.show()


def find_local_maxima(
    po2_map,
    neighborhood=3,
    threshold_rel=0.1,
    smooth_sigma=0.8
):
    """
    Find local maxima in a 2D PO2 map.
    
    Parameters
    ----------
    po2_map : (20, 20) ndarray
    neighborhood : int
        Size of neighborhood for local max (odd number)
    threshold_rel : float
        Relative threshold (fraction of max PO2)
    smooth_sigma : float
        Gaussian smoothing (0 to disable)
    
    Returns
    -------
    peaks : list of dict
        Each dict contains:
        - index: (i, j)
        - value: PO2 value
    """
    
    Z = po2_map.copy()

    # Optional smoothing (recommended for noisy PO2)
    if smooth_sigma > 0:
        Z = gaussian_filter(Z, sigma=smooth_sigma)

    # Local maximum filter
    local_max = maximum_filter(Z, size=neighborhood) == Z

    # Threshold to remove weak peaks
    threshold = threshold_rel * np.nanmax(Z)
    detected = local_max & (Z > threshold)

    # Extract peak locations
    peak_indices = np.argwhere(detected)

    peaks = []
    for i, j in peak_indices:
        peaks.append({
            "index": (i, j),
            "value": po2_map[i, j]
        })

    return peaks

po2_map = pO2_struct.pO2Value.reshape(20, 20)  # Access the po2_map variable
peaks = find_local_maxima(po2_map, neighborhood=3, threshold_rel=0.35, smooth_sigma=1.0)

print(f"Found {len(peaks)} hotspots")
for p in peaks:
    print(p)


# plt.figure(figsize=(6,5))
# plt.imshow(po2_map, cmap="inferno", origin="lower")
# plt.colorbar(label="PO₂ (mmHg)")

# for p in peaks:
#     i, j = p["index"]
#     plt.scatter(j, i, c="cyan", s=80, edgecolors="white")

# plt.title("Local PO₂ Hotspots")
# plt.xlabel("X index")
# plt.ylabel("Y index")
# plt.show()
