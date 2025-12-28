"""
Utility functions for FlowDrag mesh deformation.
Contains helper functions for geometry, projection, and visualization.
"""

import open3d as o3d
import numpy as np
import cv2


# ============================================================================
# Mesh Utilities
# ============================================================================

def clone_triangle_mesh(mesh):
    """Deep copy a TriangleMesh."""
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    
    if mesh.has_vertex_colors():
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    if mesh.has_vertex_normals():
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    
    new_mesh.compute_vertex_normals()
    return new_mesh


def create_sphere(point, color, radius):
    """Create a sphere at given point."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    return sphere


def create_sphere_at_point(point, radius=0.02, color=[1, 0, 0]):
    """Create a sphere at given point with specific parameters."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(point)
    sphere.paint_uniform_color(color)
    return sphere


# ============================================================================
# Projection Functions
# ============================================================================

def project_orthographic_2d(x, y, xmin, ymin, xmax, ymax, scale_factor_dx, scale_factor_dy):
    """Project 3D point to 2D image coordinates."""
    u = (x - xmin) * scale_factor_dx
    v = (ymax - y) * scale_factor_dy
    return (u, v)


def unproject_orthographic_2d(u, v, xmin, ymin, xmax, ymax, scale_factor_dx, scale_factor_dy):
    """Unproject 2D image coordinates to 3D."""
    x = (u / scale_factor_dx) + xmin
    y = ymax - (v / scale_factor_dy)
    return (x, y)


def project_3d_to_2d(point_3d, view_matrix, projection_matrix, width, height):
    """Project a 3D point to 2D pixel coordinates using camera matrices."""
    try:
        # Convert to homogeneous coordinates
        point_3d_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

        # Apply view transformation
        view_coords = view_matrix @ point_3d_h

        # Apply projection transformation
        clip_coords = projection_matrix @ view_coords

        # Perspective divide
        if clip_coords[3] != 0:
            ndc_coords = clip_coords[:3] / clip_coords[3]
        else:
            ndc_coords = clip_coords[:3]

        # Convert NDC to pixel coordinates
        pixel_x = (ndc_coords[0] + 1) * width / 2
        pixel_y = (1 - ndc_coords[1]) * height / 2  # Flip Y coordinate

        return [pixel_x, pixel_y]

    except Exception as e:
        print(f"❌ Error in 3D to 2D projection: {e}")
        return None


# ============================================================================
# Geometric Transformations
# ============================================================================

def compute_rotation_matrix_from_vectors(pivot, old_handle, new_handle):
    """Compute rotation matrix to rotate from old_handle to new_handle around pivot."""
    v1 = old_handle - pivot
    v2 = new_handle - pivot
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 < 1e-8 or norm_v2 < 1e-8:
        return np.eye(4)
    
    v1n = v1 / norm_v1
    v2n = v2 / norm_v2
    axis = np.cross(v1n, v2n)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-8:
        return np.eye(4)
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R_3x3 = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_3x3
    translate1 = np.eye(4)
    translate1[:3, 3] = -pivot
    translate2 = np.eye(4)
    translate2[:3, 3] = pivot
    M = translate2 @ T @ translate1
    return M


# ============================================================================
# Image Processing and Visualization
# ============================================================================

def add_projected_point_markers(img_np, handle_points, target_points, view_matrix, 
                                projection_matrix, temp_handle_2d=None, temp_target_2d=None):
    """Add red markers for projected 3D handle/target points on the captured image."""
    try:
        print("🔴 Adding red markers for projected 3D points...")

        # Check if we have the required data
        if view_matrix is None or projection_matrix is None:
            print("⚠️ No camera matrices available for point projection")
            return img_np

        if not handle_points or len(handle_points) == 0:
            print("⚠️ No 3D handle points to project")
            return img_np

        # Create a copy of the image for editing
        img_with_markers = img_np.copy()
        H, W = img_with_markers.shape[:2]

        print(f"📐 Image dimensions: {W} x {H}")

        # Calculate global offset for alignment (same as in compute_flow_field_projection)
        global_offset = np.array([0.0, 0.0])
        if temp_handle_2d is not None and temp_target_2d is not None and len(handle_points) > 0:
            # Get first 3D handle point and its 2D counterpart
            handle_3d = np.array(handle_points[0])
            handle_2d = np.array(temp_handle_2d)

            # Project 3D handle point without offset
            handle_pixel_raw = project_3d_to_2d(handle_3d, view_matrix, projection_matrix, W, H)

            if handle_pixel_raw is not None:
                # Calculate offset to align with 2D input
                global_offset = handle_2d - np.array(handle_pixel_raw)
                print(f"📍 Using global offset for capture image markers: ({global_offset[0]:.1f}, {global_offset[1]:.1f}) pixels")

        # Project 3D handle points to 2D
        for i, handle_3d in enumerate(handle_points):
            print(f"📍 Projecting 3D handle point {i+1}: {handle_3d}")

            # Project handle point and apply global offset
            handle_pixel_raw = project_3d_to_2d(handle_3d, view_matrix, projection_matrix, W, H)
            handle_pixel = None
            if handle_pixel_raw is not None:
                handle_pixel = np.array(handle_pixel_raw) + global_offset

            if handle_pixel is not None:
                x, y = int(handle_pixel[0]), int(handle_pixel[1])
                if 0 <= x < W and 0 <= y < H:
                    # Draw red circle for handle point
                    cv2.circle(img_with_markers, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
                    cv2.circle(img_with_markers, (x, y), 10, (255, 255, 255), 2)  # White outline
                    print(f"🔴 Added red marker for handle point {i+1} at pixel ({x}, {y})")
                else:
                    print(f"⚠️ Handle point {i+1} projects outside image bounds: ({x}, {y})")

            # Project corresponding target point if available
            if i < len(target_points):
                target_3d = target_points[i]
                print(f"📍 Projecting 3D target point {i+1}: {target_3d}")

                # Project target point and apply global offset
                target_pixel_raw = project_3d_to_2d(target_3d, view_matrix, projection_matrix, W, H)
                target_pixel = None
                if target_pixel_raw is not None:
                    target_pixel = np.array(target_pixel_raw) + global_offset

                if target_pixel is not None:
                    x, y = int(target_pixel[0]), int(target_pixel[1])
                    if 0 <= x < W and 0 <= y < H:
                        # Draw red square for target point (different shape to distinguish)
                        cv2.rectangle(img_with_markers, (x-6, y-6), (x+6, y+6), (0, 0, 255), -1)  # Red filled square
                        cv2.rectangle(img_with_markers, (x-8, y-8), (x+8, y+8), (255, 255, 255), 2)  # White outline
                        print(f"🔴 Added red marker for target point {i+1} at pixel ({x}, {y})")
                    else:
                        print(f"⚠️ Target point {i+1} projects outside image bounds: ({x}, {y})")

        print(f"✅ Projected {len(handle_points)} handle points and {len(target_points)} target points")
        return img_with_markers

    except Exception as e:
        print(f"❌ Error adding projected point markers: {e}")
        return img_np


def add_projected_point_markers_to_flow_field(viz_img, handle_points, target_points, 
                                              view_matrix, projection_matrix,
                                              temp_handle_2d=None, temp_target_2d=None):
    """Add red markers for projected 3D handle/target points to the flow field visualization."""
    try:
        print("🔴 Adding red markers to 2d_vector_flow_field.png...")

        # Check if we have the required data
        if view_matrix is None or projection_matrix is None:
            print("⚠️ No camera matrices available for point projection in flow field")
            return viz_img

        if not handle_points or len(handle_points) == 0:
            print("⚠️ No 3D handle points to project in flow field")
            return viz_img

        # Create a copy of the image for editing
        img_with_markers = viz_img.copy()
        H, W = img_with_markers.shape[:2]

        print(f"📐 Flow field image dimensions: {W} x {H}")

        # Calculate global offset for alignment (same as in compute_flow_field_projection)
        global_offset = np.array([0.0, 0.0])
        if temp_handle_2d is not None and temp_target_2d is not None and len(handle_points) > 0:
            # Get first 3D handle point and its 2D counterpart
            handle_3d = np.array(handle_points[0])
            handle_2d = np.array(temp_handle_2d)

            # Project 3D handle point without offset
            handle_pixel_raw = project_3d_to_2d(handle_3d, view_matrix, projection_matrix, W, H)

            if handle_pixel_raw is not None:
                # Calculate offset to align with 2D input
                global_offset = handle_2d - np.array(handle_pixel_raw)
                print(f"📍 Using global offset for markers: ({global_offset[0]:.1f}, {global_offset[1]:.1f}) pixels")

        # Project 3D handle and target points to 2D for flow field visualization
        for i, handle_3d in enumerate(handle_points):
            print(f"📍 Projecting 3D handle point {i+1} to flow field: {handle_3d}")

            # Project handle point and apply global offset
            handle_pixel_raw = project_3d_to_2d(handle_3d, view_matrix, projection_matrix, W, H)
            handle_pixel = None
            if handle_pixel_raw is not None:
                handle_pixel = np.array(handle_pixel_raw) + global_offset

            # Project corresponding target point if available
            target_pixel = None
            if i < len(target_points):
                target_3d = target_points[i]
                print(f"📍 Projecting 3D target point {i+1} to flow field: {target_3d}")
                target_pixel_raw = project_3d_to_2d(target_3d, view_matrix, projection_matrix, W, H)
                if target_pixel_raw is not None:
                    target_pixel = np.array(target_pixel_raw) + global_offset

            # Draw red arrow from handle to target if both points are valid
            if handle_pixel is not None and target_pixel is not None:
                handle_x, handle_y = int(handle_pixel[0]), int(handle_pixel[1])
                target_x, target_y = int(target_pixel[0]), int(target_pixel[1])

                if (0 <= handle_x < W and 0 <= handle_y < H and
                    0 <= target_x < W and 0 <= target_y < H):

                    # Draw thick red arrow from handle to target
                    cv2.arrowedLine(img_with_markers,
                                  (handle_x, handle_y),
                                  (target_x, target_y),
                                  (0, 0, 255), 8,  # Red color, thick line
                                  tipLength=0.3)

                    # Add white outline for better visibility
                    cv2.arrowedLine(img_with_markers,
                                  (handle_x, handle_y),
                                  (target_x, target_y),
                                  (255, 255, 255), 12,  # White outline, thicker
                                  tipLength=0.3)

                    # Draw the red arrow again on top
                    cv2.arrowedLine(img_with_markers,
                                  (handle_x, handle_y),
                                  (target_x, target_y),
                                  (0, 0, 255), 6,  # Red color, medium thickness
                                  tipLength=0.3)

                    print(f"🔴 Added red arrow for handle-target pair {i+1} from ({handle_x}, {handle_y}) to ({target_x}, {target_y}) in flow field")
                else:
                    print(f"⚠️ Handle-target pair {i+1} projects outside flow field bounds")

            elif handle_pixel is not None:
                # If only handle point is available, draw a small red circle
                handle_x, handle_y = int(handle_pixel[0]), int(handle_pixel[1])
                if 0 <= handle_x < W and 0 <= handle_y < H:
                    cv2.circle(img_with_markers, (handle_x, handle_y), 8, (255, 255, 255), -1)  # White filled circle
                    cv2.circle(img_with_markers, (handle_x, handle_y), 6, (0, 0, 255), -1)     # Red filled circle
                    print(f"🔴 Added red dot for handle point {i+1} at ({handle_x}, {handle_y}) in flow field (no target)")
                else:
                    print(f"⚠️ Handle point {i+1} projects outside flow field bounds: ({handle_x}, {handle_y})")

        print(f"✅ Added projected point markers to flow field visualization")
        return img_with_markers

    except Exception as e:
        print(f"❌ Error adding projected point markers to flow field: {e}")
        return viz_img


# ============================================================================
# Arrow Drawing Functions (from external utils_)
# ============================================================================

def sample_movable_ids(movable_ids, vertices, method='random', interval=10, sample_size=100, distance_threshold=1.0):
    """Sample movable vertex IDs using different strategies."""
    if method == 'random':
        # Randomly sample a fixed number of points
        sampled_ids = np.random.choice(movable_ids, size=min(sample_size, len(movable_ids)), replace=False)
    elif method == 'grid':
        # Sample points based on a grid interval
        sampled_ids = movable_ids[::interval]
    elif method == 'distance':
        # Sample points based on distance threshold
        sampled_ids = []
        last_selected = movable_ids[0]
        sampled_ids.append(last_selected)
        for i in movable_ids[1:]:
            if np.linalg.norm(vertices[last_selected] - vertices[i]) > distance_threshold:
                sampled_ids.append(i)
                last_selected = i
        sampled_ids = np.array(sampled_ids)
    else:
        # Default to using all movable_ids
        sampled_ids = movable_ids
    
    return sampled_ids


def draw_arrow(start_point, end_point, color=[1, 0, 0], arrow_scale_factor=0.01):
    """
    Create an arrow from start_point to end_point.
    
    Args:
        start_point: Starting position
        end_point: Ending position
        color: RGB color [r, g, b] (0-1 range)
        arrow_scale_factor: Scale factor for arrow dimensions
    """
    direction_vec = end_point - start_point
    length = np.linalg.norm(direction_vec)
    if length < 1e-8:
        # start_point and end_point are almost the same, no arrow needed
        return None

    # Create arrow with proportional dimensions
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=arrow_scale_factor * 0.5,
        cone_height=arrow_scale_factor * 1.0,
        cylinder_radius=arrow_scale_factor * 0.25,
        cylinder_height=length,
        resolution=10,
        cylinder_split=4,
        cone_split=1
    )
    arrow.paint_uniform_color(color)
    
    # Align the arrow with the movement vector
    direction = direction_vec / length
    default_direction = np.array([0, 0, 1])
    
    # Calculate rotation axis and angle
    if not np.allclose(direction, default_direction):
        rotation_axis = np.cross(default_direction, direction)
        norm_axis = np.linalg.norm(rotation_axis)
        if norm_axis > 1e-8:
            rotation_axis /= norm_axis
            rotation_angle = np.arccos(np.dot(default_direction, direction))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle
            )
            arrow.rotate(rotation_matrix, center=(0,0,0))

    arrow.translate(start_point)
    return arrow


def draw_arrows_for_movement_kookie(original_mesh, deformed_mesh, movable_ids, arrow_scale, 
                                   method='grid', interval=100):
    """
    Draw arrows showing the movement from original to deformed mesh.
    
    Args:
        original_mesh: Original mesh
        deformed_mesh: Deformed mesh
        movable_ids: IDs of movable vertices
        arrow_scale: Scale factor for arrows
        method: Sampling method ('random', 'grid', 'distance')
        interval: Interval for grid sampling
        
    Returns:
        arrows: List of arrow geometries
        sampled_movable_ids: IDs of sampled vertices
    """
    sampled_movable_ids = sample_movable_ids(movable_ids, original_mesh.vertices, method, interval)
    original_vertices = np.asarray(original_mesh.vertices)
    deformed_vertices = np.asarray(deformed_mesh.vertices)
    
    arrows = []
    for i in sampled_movable_ids:
        start_point = original_vertices[i]
        end_point = deformed_vertices[i]
        arrow = draw_arrow(start_point, end_point, color=[1,0,0], arrow_scale_factor=arrow_scale)
        if arrow is not None:
            arrows.append(arrow)
    
    return arrows, sampled_movable_ids

