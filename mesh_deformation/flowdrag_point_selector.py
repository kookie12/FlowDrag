#!/usr/bin/env python3
"""
Standalone point selector for FlowDrag.
This runs independently from the main GUI to avoid conflicts.
"""

import open3d as o3d
import numpy as np
import pickle
import sys
import os


class PointSelector:
    def __init__(self, mesh_path, existing_points_path=None):
        """Initialize point selector with mesh."""
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()

        self.handle_points = []
        self.target_points = []
        self.current_mode = "handle"  # "handle" or "target"

        # Load existing points if provided
        if existing_points_path and os.path.exists(existing_points_path):
            try:
                with open(existing_points_path, 'rb') as f:
                    data = pickle.load(f)
                    self.existing_handle_points = data.get('handle_points', [])
                    self.existing_target_points = data.get('target_points', [])
                print(f"📌 Loaded {len(self.existing_handle_points)} existing handle points")
                print(f"📌 Loaded {len(self.existing_target_points)} existing target points")
            except Exception as e:
                print(f"⚠️ Could not load existing points: {e}")
                self.existing_handle_points = []
                self.existing_target_points = []
        else:
            self.existing_handle_points = []
            self.existing_target_points = []

        # For visualization
        self.bbox = self.mesh.get_axis_aligned_bounding_box()
        self.bbox_diag = np.linalg.norm(self.bbox.get_max_bound() - self.bbox.get_min_bound())
        self.sphere_radius = self.bbox_diag * 0.01
        
    def create_sphere(self, point, color):
        """Create a sphere at given point."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        return sphere
    
    def select_points(self):
        """Main selection interface."""
        print("\n" + "="*60)
        print("FLOWDRAG POINT SELECTOR - Add New Pair")
        print("="*60)

        if len(self.existing_handle_points) > 0:
            print(f"✨ {len(self.existing_handle_points)} existing pairs will be shown")
            print("   You are adding a NEW pair on top of existing ones")

        # Select handle points
        print("\n--- STEP 1: Select HANDLE point (RED) for NEW pair ---")
        print("Instructions:")
        print("  1. Existing points are already shown (gray)")
        print("  2. Hold Shift and click on mesh to select NEW handle point")
        print("  3. Press Q when done selecting handle point")
        print("  4. Press ESC to cancel\n")

        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window(window_name="Select NEW HANDLE Point (Red) - Shift+Click, then Q", width=1000, height=800)
        vis.add_geometry(self.mesh)

        # Show existing points as gray spheres
        for hp in self.existing_handle_points:
            sphere = self.create_sphere(hp, [0.5, 0.5, 0.5])  # Gray
            vis.add_geometry(sphere)
        for tp in self.existing_target_points:
            sphere = self.create_sphere(tp, [0.3, 0.3, 0.8])  # Light blue
            vis.add_geometry(sphere)

        vis.run()
        
        picked_points = vis.get_picked_points()
        vis.destroy_window()
        
        if not picked_points:
            print("No handle points selected. Exiting.")
            return False
        
        # Process handle points
        for point in picked_points:
            coord = np.array(point.coord, dtype=float)
            self.handle_points.append(coord)
            print(f"  Handle point {len(self.handle_points)}: [{coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}]")
        
        # Select target points
        print("\n--- STEP 2: Select TARGET point (BLUE) for NEW pair ---")
        print("Instructions:")
        print("  1. Existing points are shown (gray/light blue)")
        print("  2. Hold Shift and click on mesh to select NEW target point")
        print(f"  3. Need to select {len(self.handle_points)} target point(s)")
        print("  4. Press Q when done\n")

        vis2 = o3d.visualization.VisualizerWithVertexSelection()
        vis2.create_window(window_name=f"Select NEW TARGET Point (Blue) - Shift+Click, then Q",
                          width=1000, height=800)

        # Create mesh with handle points shown
        mesh_with_handles = o3d.geometry.TriangleMesh(self.mesh)
        vis2.add_geometry(mesh_with_handles)

        # Show existing points as gray/light blue spheres
        for hp in self.existing_handle_points:
            sphere = self.create_sphere(hp, [0.5, 0.5, 0.5])  # Gray
            vis2.add_geometry(sphere)
        for tp in self.existing_target_points:
            sphere = self.create_sphere(tp, [0.3, 0.3, 0.8])  # Light blue
            vis2.add_geometry(sphere)

        # Add NEW handle spheres for reference (bright red)
        for hp in self.handle_points:
            sphere = self.create_sphere(hp, [1, 0, 0])  # Bright red
            vis2.add_geometry(sphere)

        vis2.run()
        picked_points = vis2.get_picked_points()
        vis2.destroy_window()
        
        if not picked_points:
            print("No target points selected. Exiting.")
            return False
        
        # Process target points
        for point in picked_points:
            coord = np.array(point.coord, dtype=float)
            self.target_points.append(coord)
            print(f"  Target point {len(self.target_points)}: [{coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}]")
        
        # Verify and visualize
        print("\n--- STEP 3: Verify Selection ---")
        print(f"Selected {len(self.handle_points)} handle points (red)")
        print(f"Selected {len(self.target_points)} target points (blue)")
        
        if len(self.handle_points) != len(self.target_points):
            print("\nWARNING: Number of handle and target points don't match!")
            print("Using minimum number of pairs.")
        
        # Show final result
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name="Verify Selection - Press ESC to confirm", width=1000, height=800)
        vis3.add_geometry(self.mesh)
        
        # Add all spheres
        for hp in self.handle_points:
            sphere = self.create_sphere(hp, [1, 0, 0])  # Red
            vis3.add_geometry(sphere)
        
        for tp in self.target_points:
            sphere = self.create_sphere(tp, [0, 0, 1])  # Blue
            vis3.add_geometry(sphere)
        
        # Add lines connecting pairs
        min_pairs = min(len(self.handle_points), len(self.target_points))
        for i in range(min_pairs):
            points = [self.handle_points[i], self.target_points[i]]
            lines = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green lines
            vis3.add_geometry(line_set)
        
        vis3.run()
        vis3.destroy_window()
        
        return True
    
    def save_points(self, output_path="selected_points.pkl"):
        """Save selected points to file."""
        data = {
            'handle_points': self.handle_points,
            'target_points': self.target_points
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nPoints saved to {output_path}")
        
        # Also save as text for readability
        txt_path = output_path.replace('.pkl', '.txt')
        with open(txt_path, 'w') as f:
            f.write("Handle Points:\n")
            for i, hp in enumerate(self.handle_points):
                f.write(f"  {i}: [{hp[0]:.6f}, {hp[1]:.6f}, {hp[2]:.6f}]\n")
            f.write("\nTarget Points:\n")
            for i, tp in enumerate(self.target_points):
                f.write(f"  {i}: [{tp[0]:.6f}, {tp[1]:.6f}, {tp[2]:.6f}]\n")
        
        print(f"Points also saved as text to {txt_path}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python flowdrag_point_selector.py <mesh_path> [output_path] [existing_points_path]")
        print(f"Received arguments: {sys.argv}")
        return 1

    mesh_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "selected_points.pkl"
    existing_points_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Absolute path tried: {os.path.abspath(mesh_path)}")
        return 1

    selector = PointSelector(mesh_path, existing_points_path)

    if selector.select_points():
        selector.save_points(output_path)
        print("\nPoint selection completed successfully!")
        return 0
    else:
        print("\nPoint selection cancelled.")
        return 1


if __name__ == "__main__":
    sys.exit(main())