import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

global seed_value
seed_value = 42

def plot_vectors(X_sampled, Y_sampled, U_sampled, V_sampled, title, width, height, arrow_scale=1.0, arrow_width=0.005, grid_size=None, background_image=None, selected_points=None):
    """
    Plot the sampled vectors and optionally draw the grid.

    Args:
    - X_sampled, Y_sampled: Sampled coordinates.
    - U_sampled, V_sampled: Sampled vector components.
    - title: Title of the plot.
    - arrow_scale: Scale factor for arrows.
    - arrow_width: Width of the arrows.
    - grid_size: Tuple of (rows, cols) for grid visualization (optional).
    - background_image: Background image to display (optional).
    - selected_points: Dictionary of selected points to highlight (optional).
    """
    # 이거 비율 다르게 하면 자꾸 gradio에서 깨져서 보임 => 해결할 것
    # max_size = 8  # Maximum size for any dimension
    # if width == height:
    #     figsize = (max_size, max_size)  # Square figure
    # else:
    #     aspect_ratio = width / height
    #     if aspect_ratio > 1:  # Width is greater than height
    #         figsize = (max_size, max_size / aspect_ratio)
    #     else:  # Height is greater than width
    #         figsize = (max_size * aspect_ratio, max_size)

    # plt.figure(figsize=(width / 100, height / 100))
    # NOTE: Do NOT create a new figure here - it's already created in the calling function
    # Clear the current axes to prevent accumulation
    plt.gca().clear()
    np.random.seed(seed_value)

    # Set the background - use original image if available, otherwise white background
    # Match coordinate system from load_and_visualize_vector_field
    if background_image is not None:
        plt.imshow(background_image, extent=(0, width, height, 0), origin='upper', alpha=0.7)
    else:
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        plt.imshow(white_bg, extent=(0, width, height, 0), origin='upper', alpha=0.3)

    # Calculate adaptive scale using the SAME logic as load_and_visualize_vector_field
    magnitude = np.sqrt(U_sampled**2 + V_sampled**2)
    threshold = 0.1
    significant_magnitudes = magnitude[magnitude > threshold]

    if len(significant_magnitudes) > 0:
        scale_factor = np.percentile(significant_magnitudes, 75)
        adaptive_scale = max(1.0, min(10.0, 20.0 / scale_factor)) if scale_factor > 0 else 3.0
    else:
        adaptive_scale = 3.0

    print(f"Sampling stats: max_magnitude={magnitude.max():.3f}, adaptive_scale={adaptive_scale:.3f}")

    # Separate selected_points from grid-sampled points
    num_selected = len(selected_points) // 2 if selected_points is not None and len(selected_points) >= 2 else 0

    if num_selected > 0:
        # Plot grid-sampled vectors (excluding selected_points)
        X_grid = X_sampled[:-num_selected]
        Y_grid = Y_sampled[:-num_selected]
        U_grid = U_sampled[:-num_selected]
        V_grid = V_sampled[:-num_selected]

        if len(X_grid) > 0:
            plt.quiver(X_grid, Y_grid, U_grid, V_grid,
                       angles='xy', scale_units='xy', scale=1.0/adaptive_scale,
                       color='red', width=0.006, linewidth=1.0, alpha=0.6)

        # Plot selected_points with fixed scale (no adaptive scaling)
        X_sel = X_sampled[-num_selected:]
        Y_sel = Y_sampled[-num_selected:]
        U_sel = U_sampled[-num_selected:]
        V_sel = V_sampled[-num_selected:]

        plt.quiver(X_sel, Y_sel, U_sel, V_sel,
                   angles='xy', scale_units='xy', scale=1.0/adaptive_scale,
                   color='blue', width=0.006, linewidth=1.0, alpha=0.6)
        print(f"Plotted {num_selected} selected points separately with scale=1.0")
    else:
        # No selected points, plot all with adaptive scaling
        plt.quiver(X_sampled, Y_sampled, U_sampled, V_sampled,
                   angles='xy', scale_units='xy', scale=1.0/adaptive_scale,
                   color='red', width=0.006, linewidth=1.0, alpha=0.6)

    # Plot grid lines if grid_size is provided
    if grid_size is not None:
        rows, cols = grid_size, grid_size
        cell_height = height // rows
        cell_width = width // cols

        # Draw horizontal lines
        for r in range(1, rows):
            plt.axhline(r * cell_height, color='blue', linestyle='--', linewidth=0.5)

        # Draw vertical lines
        for c in range(1, cols):
            plt.axvline(c * cell_width, color='blue', linestyle='--', linewidth=0.5)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xlim(0, width)
    plt.ylim(height, 0) # Match load_and_visualize_vector_field coordinate system
    plt.grid(True)  # Enable grid lines
    # NOTE: Do NOT call plt.show() - it interferes with plt.savefig() in Gradio
    # plt.show()


def farthest_point_sampling(X, Y, num_samples):
    """
    Farthest Point Sampling (FPS) to select spatially uniformly distributed points.
    
    Args:
    - X, Y: Coordinate arrays of candidate points
    - num_samples: Number of points to sample
    
    Returns:
    - indices: Indices of selected points
    """
    n_points = len(X)
    if num_samples >= n_points:
        return np.arange(n_points)
    
    # Stack coordinates
    points = np.stack([X, Y], axis=1)
    
    # Initialize with a random point (or center point)
    selected_indices = []
    center_idx = np.argmin(np.sum((points - points.mean(axis=0))**2, axis=1))
    selected_indices.append(center_idx)
    
    # Track minimum distances from each point to selected set
    min_distances = np.full(n_points, np.inf)
    
    for _ in range(num_samples - 1):
        # Update distances to the last selected point
        last_selected = points[selected_indices[-1]]
        distances = np.sum((points - last_selected)**2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        
        # Select the farthest point
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
        
    return np.array(selected_indices)


def grid_based_sampling(X, Y, U, V, width, height, grid_size, sampling_num=30, initial_sample_size=200):
    """
    Perform grid-based sampling to uniformly distribute samples across grid cells.
    Ensures each grid cell contributes equally to the final sampling.

    Args:
    - X, Y: Coordinate grids.
    - U, V: Vector components.
    - width, height: Dimensions of the vector field.
    - grid_size: Number of rows and columns for grid division.
    - sampling_num: Final number of vectors to sample.
    - initial_sample_size: Number of initial samples to extract for refinement.

    Returns:
    - X_sampled, Y_sampled, U_sampled, V_sampled: Uniformly distributed sampled vectors.
    """
    np.random.seed(seed_value)

    # Grid dimensions
    grid_rows, grid_cols = grid_size, grid_size
    cell_height = height // grid_rows
    cell_width = width // grid_cols
    total_cells = grid_rows * grid_cols

    # Calculate how many samples per cell for uniform distribution
    samples_per_cell = max(1, sampling_num // total_cells)

    sampled_X, sampled_Y, sampled_U, sampled_V = [], [], [], []

    # Sample uniformly from each grid cell
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Define the boundaries of the current grid cell
            y_min = i * cell_height
            y_max = (i + 1) * cell_height
            x_min = j * cell_width
            x_max = (j + 1) * cell_width

            # Find all vectors within the current cell
            cell_mask = (X >= x_min) & (X < x_max) & (Y >= y_min) & (Y < y_max)
            X_cell = X[cell_mask]
            Y_cell = Y[cell_mask]
            U_cell = U[cell_mask]
            V_cell = V[cell_mask]

            # Skip empty cells
            if len(X_cell) == 0:
                continue

            # Sample exactly samples_per_cell from this cell (or all if less available)
            num_samples = min(len(X_cell), samples_per_cell)
            sampled_indices = np.random.choice(len(X_cell), size=num_samples, replace=False)

            sampled_X.extend(X_cell[sampled_indices])
            sampled_Y.extend(Y_cell[sampled_indices])
            sampled_U.extend(U_cell[sampled_indices])
            sampled_V.extend(V_cell[sampled_indices])

    # Convert to NumPy arrays
    sampled_X = np.array(sampled_X)
    sampled_Y = np.array(sampled_Y)
    sampled_U = np.array(sampled_U)
    sampled_V = np.array(sampled_V)

    # Always use farthest point sampling to ensure exactly sampling_num samples
    if len(sampled_X) > 0:
        # Use FPS to select exactly sampling_num samples (or all if we have fewer)
        num_to_select = min(len(sampled_X), sampling_num)
        indices = farthest_point_sampling(sampled_X, sampled_Y, num_to_select)
        sampled_X = sampled_X[indices]
        sampled_Y = sampled_Y[indices]
        sampled_U = sampled_U[indices]
        sampled_V = sampled_V[indices]

    print(f"Grid-based sampling: {len(sampled_X)} samples from {total_cells} cells ({samples_per_cell} per cell)")

    return sampled_X, sampled_Y, sampled_U, sampled_V


def stratified_importance_sampling(X, Y, U, V, magnitudes, width, height, grid_size, sampling_num=10, initial_sample_size=50, clip_threshold=5.0):
    """
    Perform stratified importance sampling: first grid-based uniform sampling (~50 samples),
    then apply importance weighting based on vector magnitudes.

    Args:
    - X, Y: Coordinate grids.
    - U, V: Vector components.
    - magnitudes: Magnitudes of the vectors.
    - width, height: Dimensions of the vector field.
    - grid_size: Number of rows and columns for grid division.
    - sampling_num: Final number of vectors to sample.
    - initial_sample_size: Number of initial samples to extract uniformly (~50).
    - clip_threshold: Maximum allowed weight for clipping.

    Returns:
    - X_sampled, Y_sampled, U_sampled, V_sampled: Importance-sampled vectors.
    """
    np.random.seed(seed_value)

    # Grid dimensions
    grid_rows, grid_cols = grid_size, grid_size
    cell_height = height // grid_rows
    cell_width = width // grid_cols
    total_cells = grid_rows * grid_cols

    # Calculate samples per cell for initial grid-based sampling
    samples_per_cell = max(1, initial_sample_size // total_cells)

    initial_X, initial_Y, initial_U, initial_V, initial_magnitudes = [], [], [], [], []

    # Step 1: Grid-based uniform sampling (approximately initial_sample_size samples)
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Define the boundaries of the current grid cell
            y_min = i * cell_height
            y_max = (i + 1) * cell_height
            x_min = j * cell_width
            x_max = (j + 1) * cell_width

            # Find vectors within this cell
            cell_mask = (X >= x_min) & (X < x_max) & (Y >= y_min) & (Y < y_max)
            X_cell = X[cell_mask]
            Y_cell = Y[cell_mask]
            U_cell = U[cell_mask]
            V_cell = V[cell_mask]
            magnitudes_cell = magnitudes[cell_mask]

            if len(X_cell) == 0:
                continue  # Skip empty cells

            # Sample uniformly within the cell
            num_samples = min(len(X_cell), samples_per_cell)
            sampled_indices = np.random.choice(len(X_cell), size=num_samples, replace=False)

            initial_X.extend(X_cell[sampled_indices])
            initial_Y.extend(Y_cell[sampled_indices])
            initial_U.extend(U_cell[sampled_indices])
            initial_V.extend(V_cell[sampled_indices])
            initial_magnitudes.extend(magnitudes_cell[sampled_indices])

    # Convert to NumPy arrays
    initial_X = np.array(initial_X)
    initial_Y = np.array(initial_Y)
    initial_U = np.array(initial_U)
    initial_V = np.array(initial_V)
    initial_magnitudes = np.array(initial_magnitudes)

    print(f"Grid-based initial sampling for importance: {len(initial_X)} samples from {total_cells} cells ({samples_per_cell} per cell)")

    # If we have fewer initial samples than requested, return all initial samples
    if len(initial_X) <= sampling_num:
        print(f"Warning: Initial samples ({len(initial_X)}) <= requested samples ({sampling_num}), returning all initial samples")
        return initial_X, initial_Y, initial_U, initial_V

    # Step 2: Importance sampling on the initial set
    # Compute importance weights
    mean_magnitude = np.mean(initial_magnitudes)
    std_dev_magnitude = max(np.std(initial_magnitudes), 1e-6)  # Avoid zero std deviation
    proposal_pdf = lambda x: np.maximum(
        (1 / (std_dev_magnitude * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_magnitude) / std_dev_magnitude) ** 2),
        1e-10
    )
    weights = 1 / proposal_pdf(initial_magnitudes)
    weights = np.clip(weights, a_min=1e-10, a_max=clip_threshold)
    weights /= np.sum(weights)  # Normalize weights

    # Perform final sampling
    sampled_indices = np.random.choice(len(initial_X), size=sampling_num, p=weights, replace=False)

    X_sampled = initial_X[sampled_indices]
    Y_sampled = initial_Y[sampled_indices]
    U_sampled = initial_U[sampled_indices]
    V_sampled = initial_V[sampled_indices]

    return X_sampled, Y_sampled, U_sampled, V_sampled

