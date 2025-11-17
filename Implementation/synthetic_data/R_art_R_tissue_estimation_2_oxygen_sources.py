import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


# Constants and Parameters
D = 4e-5             # Diffusion coefficient in cm²/s
ALPHA = 1.39 / 1000  # Solubility coefficient in mLO₂/mmHg/cm³
SIGMA_SMOOTH = 2.     # Standard deviation for Gaussian smoothing
NOISE_STD = 2.        # Standard deviation of noise added to pO₂ map
SPACING = 1          # Microns per pixel — must stay 1 for this code to work correctly.

# ----------------------------
# Krogh–Erlang 2D Model Function
# ----------------------------
def krogh_2d(x, y, source_params):
    x0, y0, CMRO2, Rt_um, pO2_art, R_art_um = source_params
    CMRO2_sec = CMRO2 / 60.0  # Convert from µmol/min to µmol/s
    r_um = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_cm = r_um * 1e-4        # Convert to cm
    Rt_cm = Rt_um * 1e-4
    R_art_cm = R_art_um * 1e-4

    pO2 = np.full_like(r_cm, np.nan)
    inside = r_cm <= R_art_cm
    outside = r_cm > R_art_cm

    # Inside arteriolar radius: constant value
    pO2[inside] = pO2_art

    # Between R_art and Rt
    pO2[outside] = pO2_art + (CMRO2_sec / (D * ALPHA)) * (
        (r_cm[outside]**2 - R_art_cm**2) / 4 -
        (Rt_cm**2 / 2) * np.log(r_cm[outside] / R_art_cm)
    )

    # Outside Rt: constant boundary value
    boundary_value = pO2_art + (CMRO2_sec / (D * ALPHA)) * (
        (Rt_cm**2 - R_art_cm**2) / 4 -
        (Rt_cm**2 / 2) * np.log(Rt_cm / R_art_cm)
    )
    pO2[r_um > Rt_um] = boundary_value
    return pO2, boundary_value

# Profile Analysis Functions
def circle_profile(P, X, Y, xc, yc, r_values, num_points):
    """Compute mean pO₂ over full circles around (xc, yc)"""
    radii = np.array(r_values)
    means = np.full_like(radii, np.nan, dtype=float)
    for i, r in enumerate(radii):
        theta = np.linspace(0., 2*np.pi, num_points, endpoint=False)
        x_circle = xc + r * np.cos(theta)
        y_circle = yc + r * np.sin(theta)
        x_idx = np.clip(np.round(x_circle).astype(int), 0, X.shape[1] - 1)
        y_idx = np.clip(np.round(y_circle).astype(int), 0, Y.shape[0] - 1)
        means[i] = np.nanmean(P[y_idx, x_idx])
    return radii, means

def half_circle_profile(P, X, Y, xc, yc, xc2, yc2, r_values, num_points):
    """Compute mean pO₂ over half-circles facing away from (xc2, yc2)"""
    dx, dy = xc - xc2, yc - yc2
    theta0 = np.arctan2(dy, dx)
    radii = np.array(r_values)
    means = np.full_like(radii, np.nan, dtype=float)
    for i, r in enumerate(radii):
        theta = np.linspace(0., 2*np.pi, num_points, endpoint=False)
        x_circle = xc + r * np.cos(theta)
        y_circle = yc + r * np.sin(theta)
        angles = np.arctan2(y_circle - yc, x_circle - xc)
        mask = np.cos(angles - theta0) > 0  # select hemisphere
        x_idx = np.clip(np.round(x_circle[mask]).astype(int), 0, X.shape[1] - 1)
        y_idx = np.clip(np.round(y_circle[mask]).astype(int), 0, Y.shape[0] - 1)
        means[i] = np.nanmean(P[y_idx, x_idx])
    return radii, means


# Analysis Function
def run_analysis(P, X, Y, peak_coords):
    r_values_1 = np.arange(1, 50, 1)     # Radii for R_art estimation
    r_values_2 = np.arange(1, 120, 1)    # Radii for R_tissue estimation
    num_points = 1257                    # High enough for 1 point per µm on a 200 µm radius

    for i, (xc, yc) in enumerate(peak_coords):
        # ---- Full Circle Analysis ----
        radii, means = circle_profile(P, X, Y, xc, yc, r_values_1, num_points)
        d1 = np.diff(means)
        min_d1 = np.argmin(d1)  # Raw minimum index
        R_art = min_d1 - 5      # Apply correction for smoothing delay. NOTE: this correction depends on SIGMA_SMOOTH

        # Create figure with 2 subplots: pO₂ profile and its derivative
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Left subplot: pO₂ profile
        axs[0].plot(radii, means, marker='o', markersize=3, label="pO₂ Profile")
        axs[0].axvline(radii[min_d1], color='gray', linestyle='--',
                       label=f"R_art ≈ {radii[R_art]:.0f} µm (corrected for smoothing delay)")
        axs[0].set_title(f"Full Circle pO₂ Profile\nCenter {i + 1} ({xc:.0f}, {yc:.0f})")
        axs[0].set_xlabel("Radius (µm)")
        axs[0].set_ylabel("pO₂ (mmHg)")
        axs[0].grid()
        axs[0].legend()

        # Right subplot: Derivative of pO₂
        axs[1].plot(radii[:-1], d1, marker='x', color='black')
        if min_d1 < len(radii):
            axs[1].axvline(radii[min_d1], color='gray', linestyle='--',
                           label=f"Minimum Derivative ≈ {radii[min_d1]:.0f} µm (uncorrected)")
        else:
            print(f"Warning: min_d1 = {min_d1} is out of bounds for radii (len={len(radii)})")
        axs[1].set_title("Derivative of pO₂ vs Radius")
        axs[1].set_xlabel("Radius (µm)")
        axs[1].set_ylabel("dpO₂/dr")
        axs[1].grid()
        axs[1].legend()

        plt.tight_layout()
        plt.show()

        # ---- Half Circle Analysis for R_tissue ----
        for j, (xc2, yc2) in enumerate(peak_coords):
            if i != j:
                radii, means = half_circle_profile(P, X, Y, xc, yc, xc2, yc2, r_values_2, num_points)
                d1 = np.diff(means)
                try:
                    R_tissue = np.where(d1 > -1e-2)[0][0]  # Threshold-based tissue radius detection. NOTE: this threshold depends on SIGMA_SMOOTH
                except IndexError:
                    print(f"Warning: could not estimate R_tissue for center {i + 1}.")
                    continue

                # Create figure with 2 subplots: pO₂ and its derivative
                fig, axs = plt.subplots(1, 2, figsize=(14, 5))

                # Left subplot: pO₂ profile
                axs[0].plot(radii, means, marker='o', markersize=3, label="pO₂ Profile")
                axs[0].axvline(radii[R_tissue], color='red', linestyle='--',
                               label=f"R_tissue ≈ {radii[R_tissue]:.0f} µm")
                axs[0].set_title(
                    f"Half Circle pO₂ Profile\n"
                    f"Center {i + 1} ({xc:.0f}, {yc:.0f}) away from Center {j + 1} ({xc2:.0f}, {yc2:.0f})"
                )
                axs[0].set_xlabel("Radius (µm)")
                axs[0].set_ylabel("pO₂ (mmHg)")
                axs[0].grid()
                axs[0].legend()

                # Right subplot: Derivative
                axs[1].plot(radii[:-1], d1, marker='x', color='black')
                axs[1].axvline(radii[R_tissue], color='red', linestyle='--',
                               label="Estimated R_tissue")
                axs[1].set_title("Derivative of pO₂ vs Radius")
                axs[1].set_xlabel("Radius (µm)")
                axs[1].set_ylabel("dpO₂/dr")
                axs[1].grid()
                axs[1].legend()

                plt.tight_layout()
                plt.show()


# Main Execution Block
def main():

    # Grid Setup and Source Placement
    grid_size = 400
    x = np.arange(0, grid_size, SPACING)
    y = np.arange(0, grid_size, SPACING)
    X, Y = np.meshgrid(x, y)

    # Define two oxygen sources
    # Parameters: (x0, y0, CMRO2, Rt_um, pO2_art, R_art_um)
    # Example 1
    source1_params = (160, 150, 1.5, 106.0, 70, 7)
    source2_params = (220, 220, 2.1, 70.0, 55, 10)

    # Example 2
    # source1_params = (150, 150, 1.5, 100.0, 70, 15)
    # source2_params = (240, 240, 1.5, 95.0, 75, 7)

    # Example 3
    # source1_params = (140, 150, 1.5, 95.0, 70, 10)
    # source2_params = (240, 230, 1.5, 100.0, 75, 10)

    # Example 4
    # source1_params = (140, 150, 1.5, 95.0, 70, 10)
    # source2_params = (260, 220, 1.5, 95.0, 70, 15)

    # Example 5
    # source1_params = (160, 150, 1.5, 105.0, 72, 8)
    # source2_params = (220, 220, 1.5, 90.0, 72, 12)

    # Add Gaussian noise
    pO2_map_1, boundary_1 = krogh_2d(X, Y, source1_params)
    pO2_map_1 += np.random.normal(0, NOISE_STD, pO2_map_1.shape)
    pO2_map_2, boundary_2 = krogh_2d(X, Y, source2_params)
    pO2_map_2 += np.random.normal(0, NOISE_STD, pO2_map_2.shape)

    # Combine the maps and normalize baseline
    Z_total = pO2_map_1 + pO2_map_2
    baseline = min(boundary_1, boundary_2)
    Z_corrected = Z_total - baseline

    # Apply Gaussian smoothing
    Z_smooth = gaussian_filter(Z_corrected.copy(), sigma=SIGMA_SMOOTH)


    # Peak Detection
    min_distance_px = int(35 / SPACING)
    coordinates = peak_local_max(Z_smooth, min_distance=min_distance_px, threshold_rel=0.5, threshold_abs=50)
    peak_coords = [(X[row, col], Y[row, col]) for row, col in coordinates]


    # 2D visualization: Original and Smoothed
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(Z_corrected, cmap='jet')
    axs[0].set_title("Combined pO₂ Map (Noisy)")
    axs[1].imshow(Z_smooth, cmap='jet')
    axs[1].set_title("Smoothed pO₂ Map")
    plt.tight_layout()
    plt.show()

    # 3D Plot with Local Maxima
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(X, Y, Z_smooth, cmap='jet', edgecolor='none')
    for x_peak, y_peak in peak_coords:
        z_peak = Z_smooth[int(y_peak/SPACING), int(x_peak/SPACING)] + 3
        ax3.scatter(x_peak, y_peak, z_peak, color='yellow', edgecolor='black', s=120, marker='^', linewidth=1.5)
    ax3.set_title("Smoothed pO₂ Profiles with Detected Local Maxima")
    ax3.set_xlabel("x (µm")
    ax3.set_ylabel("y (µm")
    ax3.set_zlabel("pO₂ (mmHg)")
    plt.tight_layout()
    plt.show()

    # Run profile analysis
    print("Detected centers:", peak_coords)
    if len(peak_coords) == 2:
        run_analysis(Z_smooth, X, Y, peak_coords)
    else:
        print(f"Expected exactly 2 peaks for analysis, but found {len(peak_coords)}.")

    # Print ground truth
    print("\nGround Truth:")
    print(f"center 1: {source2_params[0]}, {source2_params[1]}")
    print(f"R_art 1: {source2_params[5]}, R_tissue 1: {source2_params[3]}")
    print(f"center 2: {source1_params[0]}, {source1_params[1]}")
    print(f"R_art 2: {source1_params[5]}, R_tissue 2: {source1_params[3]}")

if __name__ == "__main__":
    main()
