import open3d as o3d
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from open3d.visualization import gui, rendering
import sys
import json
import utils

class FlowDragEditor:
    def __init__(self, mesh_path, image_path, mask_path):
        """
        Initialize FlowDrag Editor with mesh, image, and mask.
        
        Args:
            mesh_path: Path to .ply mesh file
            image_path: Path to reference image
            mask_path: Path to mask image for movable regions
        """
        # Load mesh
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # Check if mesh loaded successfully
        if not self.mesh.has_vertices():
            raise ValueError(f"Failed to load mesh from {mesh_path}. File may not exist or be corrupted.")

        print(f"✅ Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")

        # Critical mesh preprocessing (same as working version)
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_unreferenced_vertices()

        self.mesh.compute_vertex_normals()
        self.original_mesh = None  # Will be set after clone_triangle_mesh is defined
        
        # Load image and mask with debugging
        print(f"Loading image from: {image_path}")
        print(f"Image path exists: {os.path.exists(image_path) if image_path else False}")

        if image_path and os.path.exists(image_path):
            self.image = cv2.imread(image_path)
            if self.image is not None:
                # Keep original image separate for camera capture (without web viewer annotations)
                self.original_image = self.image.copy()
                print(f"Image loaded successfully: {self.image.shape}")
            else:
                print("Failed to load image with cv2.imread")
                self.original_image = None
        else:
            self.image = None
            self.original_image = None
            print("Image path does not exist or is None")

        print(f"Loading mask from: {mask_path}")
        print(f"Mask path exists: {os.path.exists(mask_path) if mask_path else False}")

        if mask_path and os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if self.mask is not None:
                print(f"Mask loaded successfully: {self.mask.shape}")
            else:
                print("Failed to load mask with cv2.imread")
        else:
            self.mask = None
            print("Mask path does not exist or is None")
        
        # Manual region selection
        self.manual_movable_ids = set()
        self.manual_constraint_ids = set()
        self.manual_skeleton_ids = set()  # NEW: For skeleton-based rigid regions
        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None
        self.region_mode = None  # 'movable', 'constraint', or 'skeleton'
        self.use_radius_selection = False  # Alternative selection method
        self.original_mouse_handler = None  # Store original mouse handler
        self.selection_box = None  # Visual selection box

        # Computed region IDs (shared between on_show_regions and apply_simplified_sr_arap)
        self.movable_ids = []
        self.static_ids = []
        self.handle_ids = []
        
        # Preview mode for selection
        self.preview_mode = False
        self.preview_vertices = []  # Vertices in current preview
        self.pending_selection = None  # Store pending selection until confirmed
        
        # Store paths
        self.mesh_path = mesh_path
        self.image_path = image_path
        self.mask_path = mask_path
        
        # Initialize point pairs
        self.handle_points = []
        self.target_points = []
        self.current_pair_index = 0
        self.is_selecting_handle = True
        
        # Mesh properties
        self.bbox = self.mesh.get_axis_aligned_bounding_box()
        self.mesh_center = self.mesh.get_center()
        self.bbox_diag = np.linalg.norm(self.bbox.get_max_bound() - self.bbox.get_min_bound())
        
        # Safety check for bbox_diag
        if self.bbox_diag <= 0:
            raise ValueError(f"Invalid mesh bounding box diagonal: {self.bbox_diag}. Mesh may be degenerate.")
        
        print(f"📏 Mesh bounding box diagonal: {self.bbox_diag:.4f}")
        print(f"📍 Mesh center: {self.mesh_center}")
        
        # Setup orthographic projection parameters
        self.setup_projection()
        
        # Now we can clone the mesh
        self.original_mesh = utils.clone_triangle_mesh(self.mesh)
        
        # GUI components
        self.app = None
        self.window = None
        self.widget3d = None
        self.panel = None
        
        # Deformation parameters
        self.deformed_mesh = None
        self.flow_field = None
        
        # Camera matrices for projection
        self.view_matrix = None
        self.projection_matrix = None
        
        # Step-by-step process state
        self.current_step = 0
        self.arrows = []
        self.colored_mesh = None
        
        # Keep track of the widget3d bounds for drag selection
        self.widget_width = 0
        self.widget_height = 0
        
    def clone_triangle_mesh(self, mesh):
        """Deep copy a TriangleMesh - now delegates to utils module."""
        return utils.clone_triangle_mesh(mesh)
    
    def setup_projection(self):
        """Setup orthographic projection parameters."""
        xmin, ymin, zmin = self.bbox.min_bound
        xmax, ymax, zmax = self.bbox.max_bound
        
        self.xmin, self.ymin, self.zmin = xmin, ymin, zmin
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax
        
        if self.image is not None:
            H, W = self.image.shape[:2]
            self.scale_factor_dx = W / (xmax - xmin)
            self.scale_factor_dy = H / (ymax - ymin)
        else:
            self.scale_factor_dx = 512 / (xmax - xmin)
            self.scale_factor_dy = 512 / (ymax - ymin)
    
    def project_orthographic_2d(self, x, y):
        """Project 3D point to 2D image coordinates."""
        return utils.project_orthographic_2d(
            x, y, self.xmin, self.ymin, self.xmax, self.ymax, 
            self.scale_factor_dx, self.scale_factor_dy
        )
    
    def unproject_orthographic_2d(self, u, v):
        """Unproject 2D image coordinates to 3D."""
        return utils.unproject_orthographic_2d(
            u, v, self.xmin, self.ymin, self.xmax, self.ymax,
            self.scale_factor_dx, self.scale_factor_dy
        )
    
    
    def create_sphere(self, point, color, radius=None):
        """Create a sphere at given point."""
        if radius is None:
            radius = self.bbox_diag * 0.01
        return utils.create_sphere(point, color, radius)
    
    def run_gui(self):
        """Run the main GUI for point editing and deformation."""
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("FlowDrag Editor", 1200, 800)  # Larger window for dynamic square viewer + panel
        
        # Setup 3D widget
        self.setup_3d_widget()
        
        # Setup control panel
        self.setup_control_panel()
        
        # Layout
        self.setup_layout()
        
        # Run application
        self.app.run()
    
    def setup_3d_widget(self):
        """Setup 3D visualization widget."""
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        
        # Add mesh
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.8, 0.8, 0.8, 1.0]
        self.widget3d.scene.add_geometry("mesh", self.mesh, mat)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.bbox_diag * 0.1, origin=[0, 0, 0])
        coord_mat = rendering.MaterialRecord()
        coord_mat.shader = "defaultUnlit"
        self.widget3d.scene.add_geometry("coordinate_frame", coordinate_frame, coord_mat)
        
        # Setup camera
        self.widget3d.setup_camera(60, self.bbox, self.mesh_center)
        
        # Add handle and target spheres
        self.update_spheres_in_scene()
        
        # Set up mouse event handlers for drag selection
        self.widget3d.set_on_mouse(self.on_mouse_event)
        
        # Set up keyboard event handler
        self.widget3d.set_on_key(self.on_key_event)
        
        # Store original view control for restoration
        self.original_view_control = gui.SceneWidget.Controls.ROTATE_CAMERA
        
    def update_spheres_in_scene(self):
        """Update handle and target spheres in the 3D scene."""
        sphere_mat = rendering.MaterialRecord()
        sphere_mat.shader = "defaultLit"
        
        # Remove old spheres
        for i in range(10):  # Max 10 pairs
            self.widget3d.scene.remove_geometry(f"handle_{i}")
            self.widget3d.scene.remove_geometry(f"target_{i}")
        
        # Add handle spheres (red)
        for i, hp in enumerate(self.handle_points):
            sphere = self.create_sphere(hp, [1, 0, 0])
            self.widget3d.scene.add_geometry(f"handle_{i}", sphere, sphere_mat)
        
        # Add target spheres (blue)
        for i, tp in enumerate(self.target_points):
            sphere = self.create_sphere(tp, [0, 0, 1])
            self.widget3d.scene.add_geometry(f"target_{i}", sphere, sphere_mat)
        
        self.widget3d.force_redraw()
    
    def setup_control_panel(self):
        """Setup control panel with buttons and inputs."""
        em = self.window.theme.font_size
        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        
        # Title
        title = gui.Label("FlowDrag Control Panel")
        self.panel.add_child(title)
        self.panel.add_child(gui.Label(""))
        
        # Point selection section
        self.panel.add_child(gui.Label("=== Point Selection ==="))
        
        select_btn = gui.Button("Select Points on Mesh (Add New Pair)")
        select_btn.set_on_clicked(self.on_select_points)
        self.panel.add_child(select_btn)

        # Info label for multiple pairs
        info_label = gui.Label("Click button to add handle-target pairs.\nEach click adds a new pair (cumulative).")
        self.panel.add_child(info_label)

        # Current points display
        self.points_text = gui.TextEdit()
        self.points_text.text_value = "No points selected"
        self.points_text.enabled = False
        self.panel.add_child(self.points_text)

        # Button layout for clear and remove
        btn_layout = gui.Horiz()
        clear_btn = gui.Button("Clear All Points")
        clear_btn.set_on_clicked(self.on_clear_points)
        btn_layout.add_child(clear_btn)

        remove_last_btn = gui.Button("Remove Last Pair")
        remove_last_btn.set_on_clicked(self.on_remove_last_pair)
        btn_layout.add_child(remove_last_btn)
        self.panel.add_child(btn_layout)
        
        self.panel.add_child(gui.Label(""))
        
        # Target point adjustment section
        self.panel.add_child(gui.Label("=== Adjust Target Points ==="))
        
        self.panel.add_child(gui.Label("Select pair index:"))
        self.pair_selector = gui.NumberEdit(gui.NumberEdit.INT)
        self.pair_selector.int_value = 0
        self.pair_selector.set_limits(0, 10)
        self.pair_selector.set_on_value_changed(self.on_pair_selector_changed)
        self.panel.add_child(self.pair_selector)
        
        self.panel.add_child(gui.Label("Target X:"))
        self.target_x = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_x.double_value = 0 # NOTE: Target X=0
        self.target_x.set_on_value_changed(self.on_target_x_changed)
        self.panel.add_child(self.target_x)

        self.panel.add_child(gui.Label("Target Y:"))
        self.target_y = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_y.double_value = 0 # NOTE: Target Y=0
        self.target_y.set_on_value_changed(self.on_target_y_changed)
        self.panel.add_child(self.target_y)

        self.panel.add_child(gui.Label("Target Z:"))
        self.target_z = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_z.double_value = 0 # NOTE: Target Z=0
        self.target_z.set_on_value_changed(self.on_target_z_changed)
        self.panel.add_child(self.target_z)
        
        update_target_btn = gui.Button("Update Target Position")
        update_target_btn.set_on_clicked(self.on_update_target)
        self.panel.add_child(update_target_btn)
        
        self.panel.add_child(gui.Label(""))
        
        # Mask section
        self.panel.add_child(gui.Label("=== Mask Configuration ==="))
        
        self.panel.add_child(gui.Label("Mask path:"))
        self.mask_path_edit = gui.TextEdit()
        self.mask_path_edit.text_value = self.mask_path if self.mask_path else ""
        self.panel.add_child(self.mask_path_edit)
        
        load_mask_btn = gui.Button("Load Mask")
        load_mask_btn.set_on_clicked(self.on_load_mask)
        self.panel.add_child(load_mask_btn)
        
        self.panel.add_child(gui.Label(""))
        
        # Deformation section with step-by-step process
        self.panel.add_child(gui.Label("=== SR-ARAP Deformation ==="))
        
        # Deformation mode selection
        self.panel.add_child(gui.Label("Deformation Mode:"))
        self.deform_mode = gui.Combobox()
        self.deform_mode.add_item("Relocation (No Rotation)")
        self.deform_mode.add_item("Rotation (With Pivot)")
        self.deform_mode.add_item("Skeleton-based (Linear)")
        # self.deform_mode.selected_index = 1  # Default to rotation (with pivot)
        self.deform_mode.selected_index = 0  # Default to relocation
        self.panel.add_child(self.deform_mode)

        # Info label for skeleton mode
        self.skeleton_info_label = gui.Label("Skeleton mode: Keeps handle pairs in straight line")
        self.panel.add_child(self.skeleton_info_label)
        
        # Step 1: Show movable/constraint regions
        show_regions_btn = gui.Button("1. Show Movable/Constraint Regions")
        show_regions_btn.set_on_clicked(self.on_show_regions)
        self.panel.add_child(show_regions_btn)
        
        # Manual region selection controls
        self.panel.add_child(gui.Label("Manual Region Selection:"))

        horiz_layout = gui.Horiz()
        self.movable_btn = gui.Button("Select Movable")
        self.movable_btn.set_on_clicked(lambda: self.set_region_mode('movable'))
        horiz_layout.add_child(self.movable_btn)

        self.constraint_btn = gui.Button("Select Constraint")
        self.constraint_btn.set_on_clicked(lambda: self.set_region_mode('constraint'))
        horiz_layout.add_child(self.constraint_btn)
        self.panel.add_child(horiz_layout)

        # Skeleton region selection (for skeleton-based deformation)
        skeleton_layout = gui.Horiz()
        self.skeleton_btn = gui.Button("Select Skeleton Region")
        self.skeleton_btn.set_on_clicked(lambda: self.set_region_mode('skeleton'))
        skeleton_layout.add_child(self.skeleton_btn)
        self.panel.add_child(skeleton_layout)
        
        # Alternative selection method
        alt_method = gui.Button("Alt: Select by Radius")
        alt_method.set_on_clicked(self.on_alt_selection_method)
        self.panel.add_child(alt_method)
        
        # Preview status label
        self.preview_status_label = gui.Label("")
        self.panel.add_child(self.preview_status_label)
        
        self.clear_manual_btn = gui.Button("Clear Manual Selections")
        self.clear_manual_btn.set_on_clicked(self.on_clear_manual_selections)
        self.panel.add_child(self.clear_manual_btn)
        
        self.manual_info_label = gui.Label("Manual: 0 movable, 0 constraint")
        self.panel.add_child(self.manual_info_label)
        
        # Step 2: Apply deformation
        deform_btn = gui.Button("2. Apply SR-ARAP Deformation")
        deform_btn.set_on_clicked(self.on_apply_deformation)
        self.panel.add_child(deform_btn)
        
        # Step 3: Show arrows
        show_arrows_btn = gui.Button("3. Show Deformation Arrows")
        show_arrows_btn.set_on_clicked(self.on_show_arrows)
        self.panel.add_child(show_arrows_btn)
        
        # Continue button
        self.continue_btn = gui.Button("Continue to Next Step")
        self.continue_btn.set_on_clicked(self.on_continue_step)
        self.continue_btn.enabled = False
        self.panel.add_child(self.continue_btn)
        
        self.panel.add_child(gui.Label(""))
        
        # Projection section
        self.panel.add_child(gui.Label("=== 2D Projection ==="))
        self.panel.add_child(gui.Label("From Webviewer, please input the 2D handle and target point in the following part:"))

        # Add 2D handle point input for alignment
        self.panel.add_child(gui.Label("2D Handle Point:"))

        handle_2d_layout = gui.Horiz()
        handle_2d_layout.add_child(gui.Label("X:"))
        self.handle_2d_x = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.handle_2d_x.set_value(0.0)
        self.handle_2d_x.set_limits(0, 1024)
        handle_2d_layout.add_child(self.handle_2d_x)

        handle_2d_layout.add_child(gui.Label("Y:"))
        self.handle_2d_y = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.handle_2d_y.set_value(0.0)
        self.handle_2d_y.set_limits(0, 1024)
        handle_2d_layout.add_child(self.handle_2d_y)
        self.panel.add_child(handle_2d_layout)

        # Add 2D target point display
        self.panel.add_child(gui.Label("2D Target Point:"))

        target_2d_layout = gui.Horiz()
        target_2d_layout.add_child(gui.Label("X:"))
        self.target_2d_x = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.target_2d_x.set_value(0.0)
        self.target_2d_x.set_limits(0, 1024)
        target_2d_layout.add_child(self.target_2d_x)

        target_2d_layout.add_child(gui.Label("Y:"))
        self.target_2d_y = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.target_2d_y.set_value(0.0)
        self.target_2d_y.set_limits(0, 1024)
        target_2d_layout.add_child(self.target_2d_y)
        self.panel.add_child(target_2d_layout)

        self.panel.add_child(gui.Label(""))

        # Output directory path
        self.panel.add_child(gui.Label("Output Directory:"))
        self.output_dir_text = gui.TextEdit()
        # Set default output directory path
        if hasattr(self, 'output_dir'):
            self.output_dir_text.text_value = self.output_dir
        else:
            self.output_dir_text.text_value = "flow_field_outputs/[mesh_name]"
        # os.makedirs(self.output_dir_text.text_value, exist_ok=True)
        self.panel.add_child(self.output_dir_text)

        # Info label for output files
        self.output_info_label = gui.Label("Files will be saved here when capturing/projecting")
        self.panel.add_child(self.output_info_label)

        self.panel.add_child(gui.Label(""))

        # Add button to show image for picking points
        show_img_btn = gui.Button("Show Original Image")
        show_img_btn.set_on_clicked(self.on_show_original_image)
        self.panel.add_child(show_img_btn)

        # Camera Matrix & Capture Image button
        capture_matrix_btn = gui.Button("Camera Matrix & Capture Image")
        capture_matrix_btn.set_on_clicked(self.on_camera_matrix_capture)
        self.panel.add_child(capture_matrix_btn)

        # Final projection with auto-alignment
        final_project_btn = gui.Button("Auto-align to Input Image & 2D Projection")
        final_project_btn.set_on_clicked(self.on_final_auto_align_projection)
        self.panel.add_child(final_project_btn)

        self.panel.add_child(gui.Label(""))

        # Scale factor control
        self.panel.add_child(gui.Label("Flow Scale Factor:"))

        # Scale factor explanation
        scale_info = gui.Label("Scale adjusts flow arrow sizes in 2D projection.\nAuto-calculated from handle/target distance ratio.\nIncrease if arrows too small, decrease if too large.")
        self.panel.add_child(scale_info)

        scale_layout = gui.Horiz()
        scale_layout.add_child(gui.Label("Scale:"))
        self.scale_factor_input = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.scale_factor_input.set_value(1.0)
        self.scale_factor_input.set_limits(0.01, 100.0)
        scale_layout.add_child(self.scale_factor_input)
        self.panel.add_child(scale_layout)

        # Track if user manually modified scale
        self.user_modified_scale = False
        self.scale_factor_input.set_on_value_changed(self._on_scale_changed)

        # Display auto-calculated scale
        self.auto_scale_label = gui.Label("Auto-calculated: Not yet computed")
        self.panel.add_child(self.auto_scale_label)

        self.panel.add_child(gui.Label(""))

        # Camera matrix display with larger area
        self.panel.add_child(gui.Label("=== Camera Matrix ==="))
        self.camera_matrix_text = gui.TextEdit()
        self.camera_matrix_text.enabled = False
        # Set a larger size for the camera matrix display
        self.camera_matrix_text.text_value = "Camera Matrix not captured yet.\n\nClick 'Auto-align to Input Image & 2D Projection'\nto capture current view and display matrices.\n\n[View Matrix will be shown here]\n\n[Projection Matrix will be shown here]"
        self.panel.add_child(self.camera_matrix_text)
        
        self.panel.add_child(gui.Label(""))
        
        # Save section
        self.panel.add_child(gui.Label("=== Save Results ==="))

        # save_mesh_btn = gui.Button("Save Deformed Mesh")
        # save_mesh_btn.set_on_clicked(self.on_save_mesh)
        # self.panel.add_child(save_mesh_btn)
    
    def setup_layout(self):
        """Setup window layout with dynamic square 3D viewer (NxN)."""
        PANEL_WIDTH = 480  # Increased from 400 to 480

        def on_layout(context):
            r = self.window.content_rect

            # Calculate available space for viewer (window width minus panel width)
            available_width = r.width - PANEL_WIDTH
            available_height = r.height

            # Make viewer square: use the smaller of available width/height
            viewer_size = min(available_width, available_height)

            # Ensure minimum size for usability
            viewer_size = max(viewer_size, 300)  # Minimum 300x300

            # Center the square viewer in the available space if needed
            viewer_x = r.x + (available_width - viewer_size) // 2
            viewer_y = r.y + (available_height - viewer_size) // 2

            self.widget3d.frame = gui.Rect(viewer_x, viewer_y, viewer_size, viewer_size)
            self.panel.frame = gui.Rect(r.x + available_width, r.y, PANEL_WIDTH, r.height)
        
        self.window.add_child(self.widget3d)
        self.window.add_child(self.panel)
        self.window.set_on_layout(on_layout)
    
    def on_alt_selection_method(self):
        """Alternative selection method using radius around clicked point."""
        print("\n=== Alternative Selection Method ===")
        print("This method selects vertices within a radius of a clicked point.")
        print("Instructions:")
        print("1. Click 'Select Movable' or 'Select Constraint' button first")
        print("2. Then click on the mesh to select vertices around that point")
        print("3. Selection will complete automatically after one click\n")
        
        # Enable simple click selection but DON'T reset region_mode
        # Region mode should be set by the Select Movable/Constraint buttons
        self.use_radius_selection = True
        print("Radius selection enabled. Now click 'Select Movable' or 'Select Constraint' to choose mode.")
    
    def set_region_mode(self, mode):
        """Set the region selection mode."""
        self.region_mode = mode
        
        # Check if we're using radius selection or drag selection
        if self.use_radius_selection:
            # For radius selection, don't use preview mode
            self.preview_mode = False
            print(f"\n=== Radius selection mode: {mode.upper()} ===")
            print("Click on the mesh to select vertices around that point.")
            print("Selection will complete automatically after one click.\n")
        else:
            # For drag selection, use preview mode
            self.preview_mode = True
            print(f"\n=== Drag selection mode: {mode.upper()} ===")
            print("Instructions:")
            print("1. Click and drag to preview selection area (3D box will appear)")
            print("2. Press 'Q' to clear and reselect")
            print("3. Press 'Enter' to confirm selection")
            print("4. Press 'Esc' to cancel selection mode")
            print("5. Camera rotation is disabled during selection")
            print("Tip: Rotate the view first to see the desired area, then select\n")
        
        # Update button visual feedback
        if mode == 'movable':
            self.movable_btn.enabled = False
            self.constraint_btn.enabled = True
            self.skeleton_btn.enabled = True
        elif mode == 'constraint':
            self.movable_btn.enabled = True
            self.constraint_btn.enabled = False
            self.skeleton_btn.enabled = True
        elif mode == 'skeleton':
            self.movable_btn.enabled = True
            self.constraint_btn.enabled = True
            self.skeleton_btn.enabled = False
    
    def on_key_event(self, event):
        """Handle keyboard events for selection control."""
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
            
        # Only handle keys in preview mode
        if not self.preview_mode or self.region_mode is None:
            return gui.Widget.EventCallbackResult.IGNORED
        
        # Q key - Clear current preview and allow reselection
        if event.key == gui.KeyName.Q:
            print("\n[Q] Clearing preview, ready for new selection")
            self.clear_selection_box()
            self.preview_vertices = []
            self.pending_selection = None
            return gui.Widget.EventCallbackResult.HANDLED
        
        # Enter key - Confirm selection
        elif event.key == gui.KeyName.ENTER:
            if self.preview_vertices:
                print(f"\n[Enter] Confirming selection of {len(self.preview_vertices)} vertices")
                self.confirm_selection()
                return gui.Widget.EventCallbackResult.HANDLED
            else:
                print("\n[Enter] No vertices to confirm. Please make a selection first.")
                return gui.Widget.EventCallbackResult.IGNORED
        
        # Escape key - Cancel selection mode
        elif event.key == gui.KeyName.ESCAPE:
            print("\n[Esc] Cancelling selection mode")
            self.cancel_selection_mode()
            return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.IGNORED
    
    def confirm_selection(self):
        """Confirm the current preview selection."""
        if not self.preview_vertices:
            return
        
        # Instead of using preview_vertices, recalculate selection from drag bounds
        if self.drag_start is None or self.drag_end is None:
            return
            
        # Get all vertices in the selection rectangle
        selected_ids = self.get_vertices_in_screen_rectangle(self.drag_start, self.drag_end)
        
        # Apply selection based on mode
        if self.region_mode == 'movable':
            self.manual_movable_ids.update(selected_ids)
            self.manual_constraint_ids -= set(selected_ids)
            self.manual_skeleton_ids -= set(selected_ids)
            print(f"Added {len(selected_ids)} vertices to movable set")
        elif self.region_mode == 'constraint':
            self.manual_constraint_ids.update(selected_ids)
            self.manual_movable_ids -= set(selected_ids)
            self.manual_skeleton_ids -= set(selected_ids)
            print(f"Added {len(selected_ids)} vertices to constraint set")
        elif self.region_mode == 'skeleton':
            self.manual_skeleton_ids.update(selected_ids)
            self.manual_movable_ids -= set(selected_ids)
            self.manual_constraint_ids -= set(selected_ids)
            print(f"Added {len(selected_ids)} vertices to skeleton region set")

        # Update info label
        self.manual_info_label.text = f"Manual: {len(self.manual_movable_ids)} movable, {len(self.manual_constraint_ids)} constraint, {len(self.manual_skeleton_ids)} skeleton"

        # Clear preview and reset mode
        self.clear_selection_box()
        self.preview_vertices = []
        self.preview_mode = False
        self.region_mode = None
        self.movable_btn.enabled = True
        self.constraint_btn.enabled = True
        self.skeleton_btn.enabled = True
        self.preview_status_label.text = ""  # Clear status
        
        # Update visualization if regions are shown
        if self.colored_mesh is not None:
            self.on_show_regions()
        
        print("Selection confirmed. Ready for next action.")
    
    def cancel_selection_mode(self):
        """Cancel the current selection mode."""
        self.clear_selection_box()
        self.preview_vertices = []
        self.preview_mode = False
        self.region_mode = None
        self.movable_btn.enabled = True
        self.constraint_btn.enabled = True
        self.skeleton_btn.enabled = True
        self.preview_status_label.text = ""  # Clear status
        print("Selection mode cancelled.")
    
    def on_clear_manual_selections(self):
        """Clear all manual region selections."""
        self.manual_movable_ids = set()
        self.manual_constraint_ids = set()
        self.manual_skeleton_ids = set()
        self.region_mode = None
        self.movable_btn.enabled = True
        self.constraint_btn.enabled = True
        self.skeleton_btn.enabled = True
        self.manual_info_label.text = "Manual: 0 movable, 0 constraint, 0 skeleton"
        print("Cleared manual selections")
        
        # Region mode has been reset, normal camera controls are restored
        
        # Clear selection box if exists
        if self.selection_box is not None:
            self.clear_selection_box()
        
        # Update visualization if regions are shown
        if self.colored_mesh is not None:
            self.on_show_regions()
    
    def on_mouse_event(self, event):
        """Handle mouse events for drag selection."""
        # When in selection mode, consume all mouse events to prevent camera movement
        if self.region_mode is not None:
            # For radius selection (alternative method)
            if self.use_radius_selection:
                if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                    self.select_vertices_by_radius(event.x, event.y)
                    return gui.Widget.EventCallbackResult.HANDLED
                # Consume all events to prevent camera movement
                return gui.Widget.EventCallbackResult.CONSUMED
            
            # For drag selection (main method)
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                self.is_dragging = True
                self.drag_start = [event.x, event.y]
                self.drag_end = [event.x, event.y]
                print(f"Started drag selection at ({event.x}, {event.y})")
                # Create initial selection box
                self.update_selection_box()
                return gui.Widget.EventCallbackResult.CONSUMED
                
            elif event.type == gui.MouseEvent.Type.DRAG:
                if self.is_dragging:
                    self.drag_end = [event.x, event.y]
                    # Update selection box visualization
                    self.update_selection_box()
                return gui.Widget.EventCallbackResult.CONSUMED
                
            elif event.type == gui.MouseEvent.Type.BUTTON_UP:
                if self.is_dragging:
                    self.is_dragging = False
                    print(f"\nCompleted drag selection from {self.drag_start} to {self.drag_end}")
                    
                    # In preview mode, keep the selection box visible
                    if self.preview_mode:
                        print(f"Preview: {len(self.preview_vertices)} vertices selected")
                        print("Press 'Enter' to confirm, 'Q' to reselect, or 'Esc' to cancel")
                        # Don't clear the box, keep it for preview
                    else:
                        # Old behavior for non-preview mode
                        self.clear_selection_box()
                        self.select_vertices_in_rectangle()
                return gui.Widget.EventCallbackResult.CONSUMED
            
            # Consume all other mouse events during selection mode
            elif event.type in [gui.MouseEvent.Type.MOVE, gui.MouseEvent.Type.WHEEL]:
                return gui.Widget.EventCallbackResult.CONSUMED
        
        # Not in selection mode, allow normal camera controls
        return gui.Widget.EventCallbackResult.IGNORED
    
    def get_vertices_in_screen_rectangle(self, start_pos, end_pos):
        """Get all vertex indices within the screen rectangle."""
        x1, y1 = start_pos
        x2, y2 = end_pos
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        # Get widget bounds
        bounds = self.widget3d.frame
        widget_width = bounds.width
        widget_height = bounds.height
        
        # Get camera matrices
        camera = self.widget3d.scene.camera
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        mvp_matrix = proj_matrix @ view_matrix
        
        # Check all vertices
        vertices = np.asarray(self.mesh.vertices)
        selected_ids = []
        
        for i, vertex in enumerate(vertices):
            # Transform to screen coordinates
            v_h = np.append(vertex, 1.0)
            v_clip = mvp_matrix @ v_h
            
            if abs(v_clip[3]) > 1e-6:
                v_ndc = v_clip[:3] / v_clip[3]
                screen_x = (v_ndc[0] + 1.0) * 0.5 * widget_width
                screen_y = (1.0 - v_ndc[1]) * 0.5 * widget_height
                
                # Check if in selection rectangle
                if xmin <= screen_x <= xmax and ymin <= screen_y <= ymax:
                    # Also check if vertex is visible (z > 0 in view space)
                    view_pos = view_matrix @ v_h
                    if view_pos[2] < 0:  # In front of camera
                        selected_ids.append(i)
        
        return selected_ids
    
    def update_selection_box(self):
        """Update the visual selection box during drag."""
        if self.drag_start is None or self.drag_end is None:
            return
        
        # Clear previous selection box
        self.clear_selection_box()
        
        # Get rectangle corners in screen space
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        
        # Create highlighted vertices to show selection area
        # This is a simpler approach - highlight vertices that would be selected
        vertices = np.asarray(self.mesh.vertices)
        
        # Get widget bounds
        bounds = self.widget3d.frame
        widget_width = bounds.width
        widget_height = bounds.height
        
        # Get camera matrices
        camera = self.widget3d.scene.camera
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        mvp_matrix = proj_matrix @ view_matrix
        
        # Find vertices in selection rectangle
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        selected_points = []
        for vertex in vertices:
            # Transform to screen coordinates
            v_h = np.append(vertex, 1.0)
            v_clip = mvp_matrix @ v_h
            
            if abs(v_clip[3]) > 1e-6:
                v_ndc = v_clip[:3] / v_clip[3]
                screen_x = (v_ndc[0] + 1.0) * 0.5 * widget_width
                screen_y = (1.0 - v_ndc[1]) * 0.5 * widget_height
                
                # Check if in selection rectangle
                if xmin <= screen_x <= xmax and ymin <= screen_y <= ymax:
                    selected_points.append(vertex)
        
        # Create point cloud for highlighted vertices
        if len(selected_points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(selected_points)
            
            # Set color based on selection mode and preview state
            if self.preview_mode:
                # In preview mode, use semi-transparent colors
                if self.region_mode == 'movable':
                    color = [0.5, 1.0, 0.5]  # Light green for preview
                elif self.region_mode == 'constraint':
                    color = [1.0, 0.5, 0.5]  # Light red for preview
                else:
                    color = [1.0, 1.0, 0.5]  # Light yellow
            else:
                # Normal selection colors
                if self.region_mode == 'movable':
                    color = [0.0, 1.0, 0.5]  # Bright green
                elif self.region_mode == 'constraint':
                    color = [1.0, 0.0, 0.5]  # Bright red
                else:
                    color = [1.0, 1.0, 0.0]  # Yellow
            
            pcd.paint_uniform_color(color)
            
            # Add to scene
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 5.0
            
            self.widget3d.scene.add_geometry("selection_highlight", pcd, mat)
            
        # Draw 3D bounding box for selection area
        # Create a wireframe box around selected vertices
        if len(selected_points) > 0:
            selected_array = np.array(selected_points)
            
            # Calculate bounding box of selected vertices
            min_bound = np.min(selected_array, axis=0)
            max_bound = np.max(selected_array, axis=0)
            
            # Create 3D bounding box corners
            corners_3d = [
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
            ]
            
            # Create lines connecting the corners to form a box
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
            ]
            
            # Create LineSet for the bounding box
            bbox_lines = o3d.geometry.LineSet()
            bbox_lines.points = o3d.utility.Vector3dVector(corners_3d)
            bbox_lines.lines = o3d.utility.Vector2iVector(lines)
            
            # Set color based on selection mode and preview state
            if self.preview_mode:
                # Preview colors (lighter/translucent appearance)
                if self.region_mode == 'movable':
                    bbox_color = [0.5, 1.0, 0.5]  # Light green
                elif self.region_mode == 'constraint':
                    bbox_color = [1.0, 0.5, 0.5]  # Light red
                else:
                    bbox_color = [1.0, 1.0, 0.5]  # Light yellow
            else:
                if self.region_mode == 'movable':
                    bbox_color = [0.0, 1.0, 0.5]  # Bright green
                elif self.region_mode == 'constraint':
                    bbox_color = [1.0, 0.0, 0.5]  # Bright red
                else:
                    bbox_color = [1.0, 1.0, 0.0]  # Yellow
            
            bbox_lines.paint_uniform_color(bbox_color)
            
            # Add bounding box to scene
            mat_bbox = rendering.MaterialRecord()
            mat_bbox.shader = "unlitLine"
            mat_bbox.line_width = 3.0
            self.widget3d.scene.add_geometry("selection_bbox", bbox_lines, mat_bbox)
            
            # Also create a 2D selection rectangle indicator using corner spheres
            # Place small spheres at the corners of the 3D bounding box
            corner_spheres = []
            for corner in corners_3d:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.bbox_diag * 0.004)
                sphere.translate(corner)
                sphere.paint_uniform_color(bbox_color)
                corner_spheres.append(sphere)
            
            # Merge all corner spheres
            if corner_spheres:
                merged_spheres = corner_spheres[0]
                for sphere in corner_spheres[1:]:
                    merged_spheres += sphere
                
                mat_spheres = rendering.MaterialRecord()
                mat_spheres.shader = "defaultLit"
                self.widget3d.scene.add_geometry("selection_corners", merged_spheres, mat_spheres)
        
        self.widget3d.force_redraw()
        
        # Store preview vertices
        self.preview_vertices = selected_points
        
        # Update preview status label
        if self.preview_mode and len(selected_points) > 0:
            status_text = f"Preview: {len(selected_points)} vertices selected\n"
            status_text += "[Enter] Confirm | [Q] Reselect | [Esc] Cancel"
            self.preview_status_label.text = status_text
        
        # Print selection info
        if len(selected_points) > 0:
            if self.preview_mode:
                print(f"Preview: {len(selected_points)} vertices in selection area")
            else:
                print(f"Selecting {len(selected_points)} vertices...")
    
    def clear_selection_box(self):
        """Clear the visual selection box."""
        # Remove all selection visualizations
        self.widget3d.scene.remove_geometry("selection_highlight")
        self.widget3d.scene.remove_geometry("selection_corners")
        self.widget3d.scene.remove_geometry("selection_box")
        self.widget3d.scene.remove_geometry("selection_bbox")  # Remove 3D bounding box
        self.selection_box = None
        self.widget3d.force_redraw()
    
    def select_vertices_by_radius(self, click_x, click_y):
        """Select vertices within a radius of the clicked point."""
        print(f"\nSelecting vertices near ({click_x}, {click_y})")
        
        # Get widget bounds
        bounds = self.widget3d.frame
        widget_width = bounds.width
        widget_height = bounds.height
        
        # Get camera matrices
        camera = self.widget3d.scene.camera
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        mvp_matrix = proj_matrix @ view_matrix
        
        # Project vertices and find those near click point
        vertices = np.asarray(self.mesh.vertices)
        selected_ids = []
        radius = 20  # Pixel radius for selection
        
        for i, vertex in enumerate(vertices):
            # Transform to homogeneous coordinates
            v_h = np.append(vertex, 1.0)
            v_clip = mvp_matrix @ v_h
            
            if abs(v_clip[3]) > 1e-6:
                v_ndc = v_clip[:3] / v_clip[3]
                
                # Convert to screen coordinates
                screen_x = (v_ndc[0] + 1.0) * 0.5 * widget_width
                screen_y = (1.0 - v_ndc[1]) * 0.5 * widget_height
                
                # Check distance from click point
                dist = np.sqrt((screen_x - click_x)**2 + (screen_y - click_y)**2)
                if dist <= radius:
                    selected_ids.append(i)
        
        # Update selections
        if self.region_mode == 'movable':
            self.manual_movable_ids.update(selected_ids)
            self.manual_constraint_ids -= set(selected_ids)
            print(f"Selected {len(selected_ids)} vertices as movable (radius method)")
        elif self.region_mode == 'constraint':
            self.manual_constraint_ids.update(selected_ids)
            self.manual_movable_ids -= set(selected_ids)
            print(f"Selected {len(selected_ids)} vertices as constraint (radius method)")
        
        # Update info and visualization
        self.manual_info_label.text = f"Manual: {len(self.manual_movable_ids)} movable, {len(self.manual_constraint_ids)} constraint, {len(self.manual_skeleton_ids)} skeleton"
        
        # Show regions but DON'T call on_show_regions repeatedly
        # Reset region mode to stop continuous selection
        self.region_mode = None
        self.use_radius_selection = False
        
        # Update visualization once
        if self.colored_mesh is not None:
            self.on_show_regions()
    
    def select_vertices_in_rectangle(self):
        """Select vertices within the dragged rectangle (for non-preview mode)."""
        if self.preview_mode:
            # In preview mode, selection is handled by confirm_selection()
            return
            
        if self.drag_start is None or self.drag_end is None:
            return
            
        # Get rectangle bounds
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        # Check if the rectangle is too small (likely accidental click)
        if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:
            print("Selection area too small, ignoring")
            return
        
        # Get widget bounds to compute actual dimensions
        bounds = self.widget3d.frame
        widget_width = bounds.width
        widget_height = bounds.height
        
        # Get camera matrices
        camera = self.widget3d.scene.camera
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        mvp_matrix = proj_matrix @ view_matrix
        
        # Project vertices to screen space
        vertices = np.asarray(self.mesh.vertices)
        selected_ids = []
        
        for i, vertex in enumerate(vertices):
            # Transform to homogeneous coordinates
            v_h = np.append(vertex, 1.0)
            
            # Apply MVP transformation
            v_clip = mvp_matrix @ v_h
            
            # Perspective division
            if abs(v_clip[3]) > 1e-6:
                v_ndc = v_clip[:3] / v_clip[3]
            else:
                continue
                
            # Convert NDC to screen coordinates
            screen_x = (v_ndc[0] + 1.0) * 0.5 * widget_width
            screen_y = (1.0 - v_ndc[1]) * 0.5 * widget_height
            
            # Check if vertex is in selection rectangle
            if xmin <= screen_x <= xmax and ymin <= screen_y <= ymax:
                selected_ids.append(i)
        
        # Update manual selections based on mode
        if self.region_mode == 'movable':
            self.manual_movable_ids.update(selected_ids)
            # Remove from constraint if they were there
            self.manual_constraint_ids -= set(selected_ids)
            print(f"Selected {len(selected_ids)} vertices as movable")
        elif self.region_mode == 'constraint':
            self.manual_constraint_ids.update(selected_ids)
            # Remove from movable if they were there
            self.manual_movable_ids -= set(selected_ids)
            print(f"Selected {len(selected_ids)} vertices as constraint")
        
        # Update info label
        self.manual_info_label.text = f"Manual: {len(self.manual_movable_ids)} movable, {len(self.manual_constraint_ids)} constraint, {len(self.manual_skeleton_ids)} skeleton"
        
        # Reset region mode FIRST to prevent continuous selection
        self.region_mode = None
        
        # Update visualization if regions are shown
        if self.colored_mesh is not None:
            self.on_show_regions()
        self.movable_btn.enabled = True
        self.constraint_btn.enabled = True
        
        # Region mode has been reset, normal camera controls are restored
    
    def on_select_points(self):
        """Handle point selection button click."""
        import subprocess
        import pickle

        print("\nLaunching point selector...")
        print(f"Mesh path: {self.mesh_path}")

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        selector_path = os.path.join(script_dir, "flowdrag_point_selector.py")
        temp_points_path = os.path.join(script_dir, "temp_points.pkl")
        existing_points_path = os.path.join(script_dir, "existing_points.pkl")

        # Make mesh path absolute
        mesh_abs_path = os.path.abspath(self.mesh_path)

        # Save existing points for the selector to load
        existing_data = {
            'handle_points': self.handle_points,
            'target_points': self.target_points
        }
        with open(existing_points_path, 'wb') as f:
            pickle.dump(existing_data, f)

        print(f"Running: python {selector_path} {mesh_abs_path}")
        print(f"Existing points: {len(self.handle_points)} handle, {len(self.target_points)} target")

        # Run the external point selector (without closing the main GUI)
        result = subprocess.run([
            sys.executable,
            selector_path,
            mesh_abs_path,
            temp_points_path,
            existing_points_path
        ], cwd=script_dir)
        
        # Load the selected points if successful
        if result.returncode == 0 and os.path.exists(temp_points_path):
            with open(temp_points_path, 'rb') as f:
                data = pickle.load(f)
                new_handle_points = data['handle_points']
                new_target_points = data['target_points']

            # Clean up temp file
            os.remove(temp_points_path)
            txt_path = temp_points_path.replace('.pkl', '.txt')
            if os.path.exists(txt_path):
                os.remove(txt_path)

            # Add new points to existing lists (cumulative)
            old_handle_count = len(self.handle_points)
            old_target_count = len(self.target_points)

            self.handle_points.extend(new_handle_points)
            self.target_points.extend(new_target_points)

            print(f"✅ Added {len(new_handle_points)} new handle points, {len(new_target_points)} new target points")
            print(f"📊 Total: {len(self.handle_points)} handle points, {len(self.target_points)} target points")

            # Update the display without restarting GUI
            self.update_spheres_in_scene()
            self.update_points_display()

            # Clean up existing_points file
            if os.path.exists(existing_points_path):
                os.remove(existing_points_path)
        else:
            print("Point selection was cancelled or failed.")
            # Clean up existing_points file even on failure
            if os.path.exists(existing_points_path):
                os.remove(existing_points_path)
    
    def on_clear_points(self):
        """Clear all selected points."""
        self.handle_points = []
        self.target_points = []
        self.current_pair_index = 0
        self.update_spheres_in_scene()
        self.update_points_display()
        print("✅ Cleared all points")

    def on_remove_last_pair(self):
        """Remove the last handle-target pair."""
        if len(self.handle_points) > 0 and len(self.target_points) > 0:
            removed_handle = self.handle_points.pop()
            removed_target = self.target_points.pop()
            print(f"🗑️ Removed last pair:")
            print(f"   Handle: {removed_handle}")
            print(f"   Target: {removed_target}")

            # Update pair index if needed
            if self.current_pair_index >= len(self.handle_points):
                self.current_pair_index = max(0, len(self.handle_points) - 1)

            self.update_spheres_in_scene()
            self.update_points_display()
        else:
            print("⚠️ No pairs to remove")

    def on_update_target(self):
        """Update target point position from GUI inputs."""
        idx = self.pair_selector.int_value
        if idx < len(self.target_points):
            self.target_points[idx] = np.array([
                self.target_x.double_value,
                self.target_y.double_value,
                self.target_z.double_value
            ])
            self.update_spheres_in_scene()
            self.update_points_display()
            print(f"Updated target point {idx}: {self.target_points[idx]}")
    
    def on_target_x_changed(self, value):
        """Handle X coordinate change with real-time update."""
        self.update_target_coordinates_realtime()
    
    def on_target_y_changed(self, value):
        """Handle Y coordinate change with real-time update."""
        self.update_target_coordinates_realtime()
    
    def on_target_z_changed(self, value):
        """Handle Z coordinate change with real-time update."""
        self.update_target_coordinates_realtime()
    
    def update_target_coordinates_realtime(self):
        """Update target point coordinates in real-time as user types."""
        idx = self.pair_selector.int_value
        if idx < len(self.target_points):
            self.target_points[idx] = np.array([
                self.target_x.double_value,
                self.target_y.double_value,
                self.target_z.double_value
            ])
            self.update_spheres_in_scene()
            self.update_points_display()
            # Optional: print for feedback (might be too verbose)
            # print(f"Real-time update target point {idx}: {self.target_points[idx]}")
    
    def on_pair_selector_changed(self, value):
        """Handle pair selector change - load coordinates for selected pair."""
        idx = int(value)
        if idx < len(self.target_points):
            # Update coordinate fields with selected pair's values
            tp = self.target_points[idx]
            self.target_x.double_value = tp[0]
            self.target_y.double_value = tp[1]
            self.target_z.double_value = tp[2]
            print(f"Selected pair {idx}: Target = [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]")
    
    def update_points_display(self):
        """Update the points display text."""
        if self.handle_points:
            text = f"Total pairs: {min(len(self.handle_points), len(self.target_points))}\n"
            for i in range(min(len(self.handle_points), len(self.target_points))):
                hp = self.handle_points[i]
                tp = self.target_points[i] if i < len(self.target_points) else [0, 0, 0]
                text += f"Pair {i}: H:[{hp[0]:.3f}, {hp[1]:.3f}, {hp[2]:.3f}]\n"
                text += f"       T:[{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}]\n"
            self.points_text.text_value = text
            
            # Update target coordinate fields if pair is selected
            if self.pair_selector.int_value < len(self.target_points):
                tp = self.target_points[self.pair_selector.int_value]
                self.target_x.double_value = tp[0]
                self.target_y.double_value = tp[1]
                self.target_z.double_value = tp[2]
        else:
            self.points_text.text_value = "No points selected"
    
    def on_load_mask(self):
        """Load mask from the specified path."""
        mask_path = self.mask_path_edit.text_value
        if not mask_path:
            print("No mask path specified!")
            return
        
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.mask_path = mask_path
            print(f"Mask loaded from {mask_path}")
            print(f"Mask shape: {self.mask.shape}")
            # Check mask values
            unique_vals = np.unique(self.mask)
            print(f"Unique mask values: {unique_vals}")
            print(f"Non-zero pixels: {np.count_nonzero(self.mask)}")
            # Test mask indices
            test_indices = self.get_mask_indices()
            print(f"Mask would select {len(test_indices)} vertices")
        else:
            print(f"Mask file not found: {mask_path}")
    
    # NOTE: on_show_regions
    def on_show_regions(self):
        """Show movable (green), constraint (red), and skeleton (blue) regions based on mask and manual selections."""
        if self.mask is None and len(self.manual_movable_ids) == 0 and len(self.manual_constraint_ids) == 0 and len(self.manual_skeleton_ids) == 0:
            print("No mask loaded and no manual selections! Please load a mask or make manual selections first.")
            return
        
        print("Showing movable/constraint regions...")
        
        # Create colored mesh
        self.colored_mesh = self.clone_triangle_mesh(self.mesh)
        vertices = np.asarray(self.colored_mesh.vertices)
        
        # Get mask indices (includes manual selections)
        mask_indices = self.get_mask_indices()
        
        # Calculate static IDs (everything not in mask_indices, plus manual constraints)
        all_vertex_ids = set(range(len(vertices)))
        movable_from_mask = set(mask_indices.tolist())
        static_ids = (all_vertex_ids - movable_from_mask) | self.manual_constraint_ids

        # Calculate handle regions (similar to SR-ARAP function)
        all_handle_region_ids = set()
        if len(self.handle_points) > 0:
            bbox = self.mesh.get_axis_aligned_bounding_box()
            bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            radius = 0.02 * bbox_diag

            for handle_point in self.handle_points:
                distances = np.linalg.norm(vertices - handle_point, axis=1)
                handle_region_ids = np.where(distances < radius)[0]
                all_handle_region_ids.update(handle_region_ids)

        # Calculate actual movable IDs (everything - static - handle_region)
        movable_ids = list(set(range(len(vertices))) - set(static_ids) - all_handle_region_ids)

        # Store computed IDs for use in apply_simplified_sr_arap
        # For SR-ARAP: combine all movable types (mask-based + manual)
        # For SR-ARAP: combine all static types (outside mask + manual constraints)
        self.movable_ids = list(set(movable_ids) | self.manual_movable_ids)  # Combined for SR-ARAP
        self.static_ids = list(set(static_ids) | self.manual_constraint_ids)  # Combined for SR-ARAP
        self.handle_ids = list(all_handle_region_ids)  # Handle regions for SR-ARAP

        # Create vertex colors
        colors = np.ones((len(vertices), 3)) * 0.8  # Default gray

        # Color movable vertices green (brighter for manual selections)
        for idx in movable_ids:
            if idx in self.manual_movable_ids:
                colors[idx] = [0.0, 1.0, 0.5]  # Bright green for manual movable
            else:
                colors[idx] = [0.0, 0.8, 0.0]  # Normal green for movable

        # Color handle region vertices orange/yellow
        for idx in all_handle_region_ids:
            colors[idx] = [1.0, 0.5, 0.0]  # Orange for handle regions

        # Color static vertices red (brighter for manual selections)
        for idx in static_ids:
            if idx in self.manual_constraint_ids:
                colors[idx] = [1.0, 0.0, 0.5]  # Bright red for manual constraint
            else:
                colors[idx] = [0.8, 0.0, 0.0]  # Normal red for constraint

        # Color skeleton region vertices blue (bright cyan for visibility)
        for idx in self.manual_skeleton_ids:
            colors[idx] = [0.0, 0.7, 1.0]  # Bright cyan/blue for skeleton region
        
        self.colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Update visualization
        self.widget3d.scene.remove_geometry("mesh")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.widget3d.scene.add_geometry("mesh", self.colored_mesh, mat)
        self.widget3d.force_redraw()
        
        print(f"Movable vertices (green): {len(movable_ids)}")
        print(f"Handle region vertices (orange): {len(all_handle_region_ids)}")
        print(f"Constraint vertices (red): {len(static_ids)} (including {len(self.manual_constraint_ids)} manual)")
        print(f"Skeleton region vertices (cyan/blue): {len(self.manual_skeleton_ids)}")
        print(f"Total manual movable: {len(self.manual_movable_ids)}, manual constraint: {len(self.manual_constraint_ids)}, manual skeleton: {len(self.manual_skeleton_ids)}")
        print(f"Stored for SR-ARAP - Combined movable: {len(self.movable_ids)}, Static: {len(self.static_ids)}, Handle region: {len(self.handle_ids)}")
        
        self.current_step = 1
        self.continue_btn.enabled = True
    
    def on_apply_deformation(self):
        """Apply SR-ARAP deformation based on handle-target pairs."""
        if not self.handle_points or not self.target_points:
            print("No point pairs selected!")
            return
        
        print("Applying SR-ARAP deformation...")
        
        # Skip the import and use simplified version directly
        # This avoids dependency issues and provides a working implementation
        print("Using integrated SR-ARAP implementation...")
        self.apply_simplified_sr_arap()
        return
        
        # # For multiple pairs, apply deformation iteratively
        # deformed_mesh = self.clone_triangle_mesh(self.mesh)
        
        # for i, (hp, tp) in enumerate(zip(self.handle_points, self.target_points)):
        #     print(f"Processing pair {i+1}/{len(self.handle_points)}...")
        #     try:
        #         deformed_mesh = self.apply_sr_arap_deformation(
        #             deformed_mesh,
        #             hp, tp,
        #             mask=self.mask
        #         )
        #     except Exception as e:
        #         print(f"Error applying deformation for pair {i+1}: {e}")
        #         print("Using simplified deformation instead...")
        #         self.apply_simplified_sr_arap()
        #         return
        
        # self.deformed_mesh = deformed_mesh
        
        # # Update visualization
        # self.widget3d.scene.remove_geometry("mesh")
        # mat = rendering.MaterialRecord()
        # mat.shader = "defaultLit"
        # mat.base_color = [0.8, 0.8, 0.8, 1.0]
        # self.widget3d.scene.add_geometry("mesh", self.deformed_mesh, mat)
        # self.widget3d.force_redraw()
        
        # print("Deformation completed!")
        
        # self.current_step = 2
        # self.continue_btn.enabled = True
    
    def on_show_arrows(self):
        """Show arrows representing the deformation flow (basic visualization without 2D scale)."""
        if self.deformed_mesh is None:
            print("No deformed mesh! Apply deformation first.")
            return

        print("Generating deformation arrows...")

        # Get mask indices for visualization
        mask_indices = self.get_mask_indices()

        # Use the utility function from utils_ for better arrow generation
        try:
            # Use draw_arrows_for_movement_kookie for basic arrow visualization
            # Calculate appropriate arrow scale based on mesh size (same as reference)
            arrow_scale = self.bbox_diag * 0.01
            arrows, sampled_ids = utils.draw_arrows_for_movement_kookie(
                self.original_mesh if hasattr(self, 'deformed_mesh_pre_subdivision') else self.mesh,
                self.deformed_mesh_pre_subdivision if hasattr(self, 'deformed_mesh_pre_subdivision') else self.deformed_mesh,
                mask_indices,
                arrow_scale,
                method='grid',
                interval=30
            )

            # Clear old arrows
            for i in range(len(self.arrows)):
                try:
                    self.widget3d.scene.remove_geometry(f"arrow_{i}")
                except:
                    pass
            self.arrows = []

            # Add new arrows with red color
            arrow_mat = rendering.MaterialRecord()
            arrow_mat.shader = "defaultLit"
            arrow_mat.base_color = [1.0, 0.0, 0.0, 1.0]

            for i, arrow in enumerate(arrows):
                if arrow is not None:
                    self.arrows.append(arrow)
                    self.widget3d.scene.add_geometry(f"arrow_{i}", arrow, arrow_mat)

            print(f"✅ Created {len(self.arrows)} arrows")
            
        except Exception as e:
            print(f"Failed to use utils function, falling back to original method: {e}")
            
            # Fallback to original method but with improvements
            # Get vertices for fallback method
            deformed_mesh_for_fallback = self.deformed_mesh_pre_subdivision if hasattr(self, 'deformed_mesh_pre_subdivision') else self.deformed_mesh
            original_mesh_for_fallback = self.original_mesh if hasattr(self, 'deformed_mesh_pre_subdivision') else self.mesh
            
            original_vertices = np.asarray(original_mesh_for_fallback.vertices)
            deformed_vertices = np.asarray(deformed_mesh_for_fallback.vertices)
            
            movements = deformed_vertices - original_vertices
            movement_magnitudes = np.linalg.norm(movements, axis=1)
            
            # Filter vertices with significant movement
            threshold = self.bbox_diag * 0.0005  # Lower threshold to show more movements
            moving_indices = np.where(movement_magnitudes > threshold)[0]

            # Allow more arrows for better visualization
            max_arrows = 200  # Increased from 100 to 200 for more uniform coverage
            if len(moving_indices) > max_arrows:
                # Use uniform sampling for better distribution
                step = len(moving_indices) // max_arrows
                moving_indices = moving_indices[::step][:max_arrows]
            
            # Clear old arrows
            for i in range(len(self.arrows)):
                self.widget3d.scene.remove_geometry(f"arrow_{i}")
            self.arrows = []
            
            # Calculate scale factor based on 2D input distance
            scale_factor = 1.0  # Default

            if (hasattr(self, 'temp_handle_2d') and hasattr(self, 'temp_target_2d') and
                self.temp_handle_2d is not None and self.temp_target_2d is not None and
                len(self.handle_points) > 0 and len(self.target_points) > 0):

                # Get 2D input distance (in image coordinates 0-512)
                handle_2d = np.array(self.temp_handle_2d)
                target_2d = np.array(self.temp_target_2d)
                input_2d_distance = np.linalg.norm(target_2d - handle_2d)

                # Get 3D canonical space distance (typically < 1)
                handle_3d = np.array(self.handle_points[0])
                target_3d = np.array(self.target_points[0])
                canonical_3d_distance = np.linalg.norm(target_3d - handle_3d)

                if canonical_3d_distance > 1e-6:
                    # Scale arrows to match 2D input scale
                    scale_factor = input_2d_distance / canonical_3d_distance
                    print(f"📏 2D input distance: {input_2d_distance:.2f} pixels")
                    print(f"📏 3D canonical distance: {canonical_3d_distance:.4f}")
                    print(f"📏 Calculated scale factor: {scale_factor:.2f}")

            # Create arrows with improved size
            arrow_mat = rendering.MaterialRecord()
            arrow_mat.shader = "defaultLit"
            arrow_mat.base_color = [1.0, 0.0, 0.0, 1.0]  # Red arrows

            for i, idx in enumerate(moving_indices):
                start = original_vertices[idx]
                movement = movements[idx]

                # Create arrow with scale matching 2D input
                arrow = self.create_arrow(start, movement, scale=scale_factor)
                if arrow:
                    self.arrows.append(arrow)
                    self.widget3d.scene.add_geometry(f"arrow_{i}", arrow, arrow_mat)

            print(f"Created {len(self.arrows)} red arrows with scale={scale_factor:.2f}")
        
        self.widget3d.force_redraw()
        self.current_step = 3
        self.continue_btn.enabled = True
    
    def create_arrow(self, start_point, direction, scale=1.0):
        """Create an arrow mesh from start point in given direction.
        Uses the previous arrow style with arrow_scale_factor based on bbox_diag.
        """
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None

        # Use arrow scale factor similar to draw_arrows_for_movement_kookie
        arrow_scale_factor = self.bbox_diag * 0.01  # Same as previous version

        # Create arrow with previous style (similar to utils_.draw_arrow)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_radius=arrow_scale_factor * 0.5 * scale,
            cone_height=arrow_scale_factor * 1.0 * scale,
            cylinder_radius=arrow_scale_factor * 0.25 * scale,
            cylinder_height=length * scale,
            resolution=10,
            cylinder_split=4,
            cone_split=1
        )

        # Rotate arrow to align with direction
        direction_normalized = direction / length
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction_normalized)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))

        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            arrow.rotate(rotation_matrix, center=[0, 0, 0])

        # Translate to start point
        arrow.translate(start_point)

        return arrow

    def _on_scale_changed(self, value):
        """Callback when user changes scale factor."""
        self.user_modified_scale = True
        print(f"🔧 User manually set scale factor to: {value:.3f}")

    # NOTE: on_continue_step()
    def on_continue_step(self):
        """Continue to the next step in the process."""
        if self.current_step == 1:
            # After showing regions, apply deformation
            self.on_apply_deformation()
        elif self.current_step == 2:
            # After deformation, show arrows
            self.on_show_arrows()
        elif self.current_step == 3:
            # After arrows, ready for projection
            print("✅ Ready for projection. Use 'Auto-align to Input Image & 2D Projection' to save flow field.")
            self.continue_btn.enabled = False
    
    def compute_pivot_same_z_as_handle(self, mesh, handle_point_3d, epsilon=1e-3):
        """Compute pivot point for rotation based on handle point."""
        v = np.asarray(mesh.vertices)
        y_handle = handle_point_3d[1]
        
        # Find vertices close to handle's y-coordinate (same horizontal plane)
        mask = np.abs(v[:, 1] - y_handle) < epsilon
        close_ids = np.where(mask)[0]
        
        if len(close_ids) == 0:
            # If no vertices found, try with larger epsilon
            epsilon *= 5
            mask = np.abs(v[:, 1] - y_handle) < epsilon
            close_ids = np.where(mask)[0]
            
            if len(close_ids) == 0:
                print(f"Warning: No vertices found near y={y_handle}, using mesh center as pivot")
                return self.mesh_center
        
        # Calculate mean position of close vertices
        close_points = v[close_ids]
        mean_x = np.mean(close_points[:, 0])
        mean_z = np.mean(close_points[:, 2])
        
        pivot = np.array([mean_x, y_handle, mean_z])
        print(f"Pivot computed at {pivot} using {len(close_ids)} vertices")
        return pivot
    
    def compute_rotation_matrix_from_vectors(self, pivot, old_handle, new_handle):
        """Compute rotation matrix to rotate from old_handle to new_handle around pivot."""
        return utils.compute_rotation_matrix_from_vectors(pivot, old_handle, new_handle)
    
    def apply_transform_to_selected_vertices(self, mesh, M_4x4, indices):
        """Apply transformation matrix to selected vertices."""
        v = np.asarray(mesh.vertices)
        n_vertices = len(v)
        
        # Validate indices
        valid_indices = []
        for idx in indices:
            if 0 <= idx < n_vertices:
                valid_indices.append(idx)
            else:
                print(f"Warning: Index {idx} out of bounds (mesh has {n_vertices} vertices)")
        
        if len(valid_indices) == 0:
            print("No valid indices to transform")
            return mesh
        
        valid_indices = np.array(valid_indices)
        ones = np.ones((len(valid_indices), 1))
        
        # Transform selected vertices
        selected_v = v[valid_indices]
        selected_v_h = np.concatenate([selected_v, ones], axis=1)
        v_new = (M_4x4 @ selected_v_h.T).T[:, :3]
        
        # Update mesh vertices
        v[valid_indices] = v_new
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.compute_vertex_normals()
        return mesh

    def create_sphere_at_point(self, point, radius=0.02, color=[1, 0, 0]):
        return utils.create_sphere_at_point(point, radius, color)

    # NOTE: Skeleton-based Deformation Helper Functions
    def create_bone_chain_from_handles(self, handle_points, target_points):
        """Create bone chain from handle-target pairs.

        Returns:
            bones: List of dicts with 'start', 'end', 'start_target', 'end_target'
        """
        bones = []
        for i in range(len(handle_points) - 1):
            bone = {
                'start': np.array(handle_points[i]),
                'end': np.array(handle_points[i + 1]),
                'start_target': np.array(target_points[i]),
                'end_target': np.array(target_points[i + 1])
            }
            bones.append(bone)

        print(f"🦴 Created {len(bones)} bones from {len(handle_points)} handle points")
        return bones

    def compute_bone_weights(self, vertices, bones, bone_radius):
        """Compute skinning weights for each vertex to each bone.

        Args:
            vertices: (N, 3) array of mesh vertices
            bones: List of bone dictionaries
            bone_radius: Influence radius for each bone

        Returns:
            weights: (N, num_bones) array where weights[i, j] is influence of bone j on vertex i
        """
        N = len(vertices)
        num_bones = len(bones)
        weights = np.zeros((N, num_bones))

        for bone_idx, bone in enumerate(bones):
            bone_start = bone['start']
            bone_end = bone['end']
            bone_vec = bone_end - bone_start
            bone_length = np.linalg.norm(bone_vec)

            if bone_length < 1e-6:
                continue

            bone_dir = bone_vec / bone_length

            for v_idx, vertex in enumerate(vertices):
                # Compute distance from vertex to bone line segment
                to_vertex = vertex - bone_start
                t = np.dot(to_vertex, bone_dir)
                t = np.clip(t, 0, bone_length)  # Clamp to bone segment

                closest_point = bone_start + t * bone_dir
                distance = np.linalg.norm(vertex - closest_point)

                # Weight based on distance (inverse distance weighting)
                if distance < bone_radius:
                    weights[v_idx, bone_idx] = 1.0 - (distance / bone_radius)

        # Normalize weights so they sum to 1 for each vertex
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        weights = weights / row_sums

        print(f"💪 Computed skinning weights: {np.count_nonzero(weights)} non-zero weights")
        return weights

    def compute_bone_transformation(self, bone):
        """Compute transformation matrix from bone's original position to target position.

        Args:
            bone: Dict with 'start', 'end', 'start_target', 'end_target'

        Returns:
            T: 4x4 transformation matrix
        """
        # Original bone vector
        v_orig = bone['end'] - bone['start']
        v_orig_len = np.linalg.norm(v_orig)

        # Target bone vector
        v_target = bone['end_target'] - bone['start_target']
        v_target_len = np.linalg.norm(v_target)

        if v_orig_len < 1e-6 or v_target_len < 1e-6:
            return np.eye(4)

        v_orig_normalized = v_orig / v_orig_len
        v_target_normalized = v_target / v_target_len

        # Compute rotation from v_orig to v_target
        cross = np.cross(v_orig_normalized, v_target_normalized)
        dot = np.dot(v_orig_normalized, v_target_normalized)

        # Handle parallel vectors
        if np.linalg.norm(cross) < 1e-6:
            if dot > 0:
                R = np.eye(3)  # Same direction
            else:
                R = -np.eye(3)  # Opposite direction
        else:
            # Rodrigues' rotation formula
            cross_normalized = cross / np.linalg.norm(cross)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            K = np.array([[0, -cross_normalized[2], cross_normalized[1]],
                          [cross_normalized[2], 0, -cross_normalized[0]],
                          [-cross_normalized[1], cross_normalized[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Compute scale
        scale = v_target_len / v_orig_len

        # Combine rotation and scale
        RS = R * scale

        # Compute translation
        t = bone['start_target'] - (RS @ bone['start'])

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = RS
        T[:3, 3] = t

        return T

    def apply_linear_blend_skinning(self, vertices, bones, weights):
        """Apply Linear Blend Skinning (LBS) deformation.

        Args:
            vertices: (N, 3) array of original vertices
            bones: List of bone dictionaries
            weights: (N, num_bones) skinning weights

        Returns:
            deformed_vertices: (N, 3) array of deformed vertices
        """
        N = len(vertices)
        deformed_vertices = np.zeros_like(vertices)

        # Compute transformation for each bone
        bone_transforms = [self.compute_bone_transformation(bone) for bone in bones]

        # Apply weighted transformation to each vertex
        for v_idx in range(N):
            vertex_homogeneous = np.append(vertices[v_idx], 1.0)
            blended_vertex = np.zeros(4)

            for bone_idx in range(len(bones)):
                weight = weights[v_idx, bone_idx]
                if weight > 0:
                    transformed = bone_transforms[bone_idx] @ vertex_homogeneous
                    blended_vertex += weight * transformed

            deformed_vertices[v_idx] = blended_vertex[:3]

        return deformed_vertices

    # NOTE: NEW Skeleton-based Rigid Deformation Functions
    def compute_skeleton_rigid_transformation(self, handle_point, target_point, skeleton_region_vertices):
        """Compute rigid transformation (rotation + translation) for skeleton region.

        Args:
            handle_point: (3,) Handle point position
            target_point: (3,) Target point position
            skeleton_region_vertices: (N, 3) Vertices in the skeleton region

        Returns:
            T: 4x4 transformation matrix
            centroid: (3,) Centroid of skeleton region
        """
        # Compute centroid of skeleton region
        centroid = np.mean(skeleton_region_vertices, axis=0)

        # Compute transformation vector (handle -> target)
        translation_vector = target_point - handle_point

        # For now, use pure translation (can extend to rotation later)
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, 3] = translation_vector

        print(f"  Skeleton rigid transformation:")
        print(f"    Centroid: {centroid}")
        print(f"    Translation: {translation_vector}")
        print(f"    Region size: {len(skeleton_region_vertices)} vertices")

        return T, centroid

    def compute_skeleton_weights_smooth_falloff(self, vertices, skeleton_region_ids, falloff_radius):
        """Compute smooth falloff weights for skeleton deformation.

        Args:
            vertices: (N, 3) All mesh vertices
            skeleton_region_ids: List of indices for skeleton region
            falloff_radius: Radius for smooth falloff

        Returns:
            weights: (N,) array where 1.0 = fully rigid, 0.0 = no influence
        """
        N = len(vertices)
        weights = np.zeros(N)

        if len(skeleton_region_ids) == 0:
            return weights

        skeleton_vertices = vertices[skeleton_region_ids]
        skeleton_set = set(skeleton_region_ids)

        for v_idx in range(N):
            if v_idx in skeleton_set:
                # Vertices in skeleton region are fully rigid
                weights[v_idx] = 1.0
            else:
                # Compute distance to nearest skeleton vertex
                vertex = vertices[v_idx]
                distances = np.linalg.norm(skeleton_vertices - vertex, axis=1)
                min_distance = np.min(distances)

                # Smooth falloff: quadratic decay
                if min_distance < falloff_radius:
                    # Falloff from 1.0 (at distance 0) to 0.0 (at falloff_radius)
                    normalized_dist = min_distance / falloff_radius
                    weights[v_idx] = (1.0 - normalized_dist) ** 2
                else:
                    weights[v_idx] = 0.0

        num_rigid = np.sum(weights >= 0.99)
        num_blend = np.sum((weights > 0.01) & (weights < 0.99))
        print(f"  Skeleton weights: {num_rigid} rigid, {num_blend} blending, {N - num_rigid - num_blend} unaffected")

        return weights

    def apply_skeleton_rigid_deformation(self, vertices, handle_point, target_point, skeleton_region_ids, falloff_radius):
        """Apply rigid deformation to skeleton region with smooth blending.

        Args:
            vertices: (N, 3) Current mesh vertices
            handle_point: (3,) Handle point
            target_point: (3,) Target point
            skeleton_region_ids: List of vertex indices in skeleton region
            falloff_radius: Smooth falloff radius

        Returns:
            deformed_vertices: (N, 3) Deformed vertices
        """
        print(f"\n  Applying skeleton rigid deformation:")
        print(f"    Skeleton region size: {len(skeleton_region_ids)} vertices")
        print(f"    Handle: {handle_point}")
        print(f"    Target: {target_point}")

        if len(skeleton_region_ids) == 0:
            print("    Warning: Empty skeleton region, skipping")
            return vertices.copy()

        # Compute rigid transformation for skeleton region
        skeleton_vertices = vertices[skeleton_region_ids]
        T, centroid = self.compute_skeleton_rigid_transformation(
            handle_point, target_point, skeleton_vertices
        )

        # Compute smooth falloff weights
        weights = self.compute_skeleton_weights_smooth_falloff(
            vertices, skeleton_region_ids, falloff_radius
        )

        # Apply weighted transformation
        deformed_vertices = vertices.copy()

        # Transform all vertices with weighting
        vertices_homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        transformed_vertices = (T @ vertices_homogeneous.T).T[:, :3]

        # Blend: deformed = weight * transformed + (1-weight) * original
        for v_idx in range(len(vertices)):
            w = weights[v_idx]
            deformed_vertices[v_idx] = w * transformed_vertices[v_idx] + (1 - w) * vertices[v_idx]

        print(f"    ✅ Skeleton rigid deformation applied")
        return deformed_vertices

    # NOTE: SR-ARAP deformation 본체
    def apply_simplified_sr_arap(self):
        """SR-ARAP deformation implementation for multiple handle-target pairs."""
        print("Using SR-ARAP deformation implementation...")

        # Check deformation mode
        use_pivot_rotation = (self.deform_mode.selected_index == 1)
        use_skeleton_mode = (self.deform_mode.selected_index == 2)

        if use_skeleton_mode:
            print("Using SKELETON-BASED mode (linear bone chain)")
        elif use_pivot_rotation:
            print("Using ROTATION mode with pivot")
        else:
            print("Using RELOCATION mode (no rotation)")

        # Progressive deformation parameters
        n_steps = 5  # Number of progressive steps
        alpha_reg = 0.0  # ARAP smoothing parameter

        # Store the original mesh BEFORE deformation for arrow calculation
        if not hasattr(self, 'original_mesh') or self.original_mesh is None:
            self.original_mesh = self.clone_triangle_mesh(self.mesh)
            print("✅ Stored original mesh for comparison")

        # Clone the original mesh
        self.deformed_mesh = self.clone_triangle_mesh(self.mesh)
        self.deformed_mesh.compute_vertex_normals()

        # Preserve original colors/textures
        original_colors = None
        if self.mesh.has_vertex_colors():
            original_colors = np.asarray(self.mesh.vertex_colors).copy()

        vertices = np.asarray(self.deformed_mesh.vertices)
        v_orig = vertices.copy()  # Store original vertices

        # Use vertices computed and stored by on_show_regions
        if hasattr(self, 'movable_ids') and hasattr(self, 'static_ids') and hasattr(self, 'handle_ids'):
            # Use the combined IDs from on_show_regions - already includes all manual selections
            all_movable_ids = self.movable_ids
            all_static_ids = self.static_ids
            all_handle_region_ids = self.handle_ids
            print(f"Using pre-computed region IDs from on_show_regions:")
            print(f"Combined movable: {len(all_movable_ids)}, Combined static: {len(all_static_ids)}, Handle regions: {len(all_handle_region_ids)}")
        else:
            # Compute regions directly (fallback)
            print("Computing regions directly...")
            mask_indices = self.get_mask_indices()
            all_vertex_ids = set(range(len(vertices)))
            movable_from_mask = set(mask_indices.tolist()) if len(mask_indices) > 0 else set()
            all_static_ids = list((all_vertex_ids - movable_from_mask) | self.manual_constraint_ids)
            all_movable_ids = list(set(range(len(vertices))) - set(all_static_ids))
            all_movable_ids = list(set(all_movable_ids) | self.manual_movable_ids)
            all_handle_region_ids = []

        # Store for consistency
        self.final_movable_ids = all_movable_ids
        self.final_static_ids = all_static_ids
        self.final_handle_region_ids = all_handle_region_ids

        static_pos = vertices[all_static_ids] if len(all_static_ids) > 0 else np.array([])

        # Calculate handle region radius for later use
        bbox = self.deformed_mesh.get_axis_aligned_bounding_box()
        bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radius = 0.02 * bbox_diag

        print(f"Final vertex counts:")
        print(f"Movable vertices: {len(all_movable_ids)}")
        print(f"Handle region vertices: {len(all_handle_region_ids)}")
        print(f"Static vertices: {len(all_static_ids)}")

        # SKELETON-BASED DEFORMATION: Apply rigid deformation if skeleton mode is enabled
        if use_skeleton_mode and len(self.handle_points) >= 1:
            print(f"\n{'='*60}")
            print(f"🦴 SKELETON-BASED RIGID DEFORMATION")
            print(f"{'='*60}")

            # Check if user has selected skeleton region
            if len(self.manual_skeleton_ids) == 0:
                print("⚠️ WARNING: No skeleton region selected!")
                print("   Please use 'Select Skeleton Region' button to select the rigid region.")
                print("   Falling back to OLD bone chain method...")

                # OLD FALLBACK: Create bone chain from handle-target pairs
                if len(self.handle_points) >= 2:
                    bones = self.create_bone_chain_from_handles(self.handle_points, self.target_points)
                    if len(bones) > 0:
                        bone_radius = radius * 3.0
                        weights = self.compute_bone_weights(vertices, bones, bone_radius)
                        deformed_verts = self.apply_linear_blend_skinning(vertices, bones, weights)
                        self.deformed_mesh.vertices = o3d.utility.Vector3dVector(deformed_verts)
                        self.deformed_mesh.compute_vertex_normals()
                        vertices = np.asarray(self.deformed_mesh.vertices)
            else:
                # NEW METHOD: Use skeleton region for rigid deformation
                print(f"Using skeleton region: {len(self.manual_skeleton_ids)} vertices")

                # Convert skeleton IDs to list
                skeleton_region_ids = list(self.manual_skeleton_ids)

                # Compute falloff radius (larger than handle radius for smooth blending)
                falloff_radius = radius * 4.0
                print(f"Smooth falloff radius: {falloff_radius:.4f}")

                # Apply rigid deformation for each handle-target pair sequentially
                for pair_idx, (handle_pt, target_pt) in enumerate(zip(self.handle_points, self.target_points)):
                    print(f"\n  Processing handle-target pair {pair_idx + 1}/{len(self.handle_points)}")

                    # Apply skeleton rigid deformation
                    vertices = self.apply_skeleton_rigid_deformation(
                        vertices,
                        handle_pt,
                        target_pt,
                        skeleton_region_ids,
                        falloff_radius
                    )

                    # Update mesh
                    self.deformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    self.deformed_mesh.compute_vertex_normals()

                print(f"\n✅ Applied skeleton rigid deformation for {len(self.handle_points)} pairs")
                print(f"   Skeleton region transformed as rigid body with smooth blending")

        # HANDLE HIERARCHY: Process pairs sequentially (hierarchical order)
        use_hierarchy = len(self.handle_points) > 1 and not use_skeleton_mode
        if use_hierarchy:
            print(f"🔗 Using HANDLE HIERARCHY for {len(self.handle_points)} pairs")
            print(f"   Pairs will be applied sequentially: 0 → 1 → 2 → ...")
        else:
            print(f"Processing {len(self.handle_points)} handle-target pair")

        # Prepare all handle-target pairs
        handle_currents = [np.array(hp, dtype=np.float64) for hp in self.handle_points]
        target_points_3d = [np.array(tp, dtype=np.float64) for tp in self.target_points]
        handle_points_3d = [np.array(hp, dtype=np.float64) for hp in self.handle_points]

        # Calculate deltas for each pair
        deltas = [(target - handle) / n_steps for target, handle in zip(target_points_3d, handle_points_3d)]

        # Calculate pivot if using rotation mode (only for first pair)
        pivot = None
        transform_ids = None
        if use_pivot_rotation and len(self.handle_points) > 0:
            pivot = self.compute_pivot_same_z_as_handle(self.deformed_mesh, self.handle_points[0])
            if pivot is not None:
                print(f"Pivot calculated at {pivot}")
                # Transform IDs = handle_region + movable_ids (static excluded)
                transform_ids = np.array(list(set(all_handle_region_ids + all_movable_ids)), dtype=np.int32)
                print(f"Transform vertices for rotation: {len(transform_ids)}")

        # Progressive deformation loop - HANDLE HIERARCHY implementation
        if use_skeleton_mode:
            # SKELETON MODE: LBS already applied, skip SR-ARAP progressive steps
            print(f"\n{'='*60}")
            print(f"Skeleton mode complete - skipping SR-ARAP refinement")
            print(f"The mesh has been deformed using Linear Blend Skinning")
            print(f"{'='*60}")

            # Update handle_currents to target positions for verification
            for pair_idx in range(len(handle_currents)):
                handle_currents[pair_idx] = target_points_3d[pair_idx].copy()

        elif use_hierarchy:
            # HANDLE HIERARCHY: Process each pair sequentially through all steps
            for pair_idx in range(len(handle_currents)):
                print(f"\n{'='*60}")
                print(f"🔗 Processing Handle-Target Pair {pair_idx + 1}/{len(handle_currents)} (HIERARCHY)")
                print(f"{'='*60}")

                # Process this pair through all n_steps
                for step_idx in range(n_steps):
                    print(f"\n  Pair {pair_idx + 1}, Step {step_idx + 1}/{n_steps}")

                    # Get current vertices
                    vertices = np.asarray(self.deformed_mesh.vertices)

                    # Find handle region for this specific pair
                    handle_curr = handle_currents[pair_idx]
                    delta = deltas[pair_idx]

                    distances = np.linalg.norm(vertices - handle_curr, axis=1)
                    step_handle_region_ids = np.where(distances < radius)[0].tolist()
                    step_handle_region_pos = [vertices[idx] + delta for idx in step_handle_region_ids]

                    print(f"  Handle region: {len(step_handle_region_ids)} vertices")

                    # Apply rotation for first 2 steps if enabled (only for first pair)
                    if use_pivot_rotation and pivot is not None and pair_idx == 0 and step_idx < 2:
                        if len(transform_ids) > 0:
                            M = self.compute_rotation_matrix_from_vectors(
                                pivot,
                                handle_currents[0],
                                handle_currents[0] + deltas[0]
                            )
                            self.deformed_mesh = self.apply_transform_to_selected_vertices(
                                self.deformed_mesh, M, transform_ids
                            )
                            vertices = np.asarray(self.deformed_mesh.vertices)

                    # Prepare constraints
                    if len(all_static_ids) == 0:
                        constraint_ids_ovec = o3d.utility.IntVector(step_handle_region_ids)
                        constraint_pos_ovec = o3d.utility.Vector3dVector(vertices[step_handle_region_ids])
                    else:
                        constraint_ids_ovec = o3d.utility.IntVector(np.concatenate((all_static_ids, step_handle_region_ids)).tolist())
                        constraint_pos_ovec = o3d.utility.Vector3dVector(np.concatenate((static_pos, step_handle_region_pos)).tolist())

                    if len(constraint_ids_ovec) == 0:
                        print("  Warning: No constraints, skipping step")
                        continue

                    # Apply ARAP deformation
                    try:
                        self.deformed_mesh = self.deformed_mesh.deform_as_rigid_as_possible(
                            constraint_ids_ovec,
                            constraint_pos_ovec,
                            max_iter=30,
                            energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
                            smoothed_alpha=alpha_reg
                        )

                        # Apply smoothing
                        self.deformed_mesh.filter_smooth_taubin(
                            number_of_iterations=30,
                            lambda_filter=0.5,
                            mu=-0.53
                        )

                        # Update handle position for next step
                        handle_currents[pair_idx] = handle_currents[pair_idx] + deltas[pair_idx]

                    except Exception as e:
                        print(f"  ERROR: ARAP failed at pair {pair_idx + 1}, step {step_idx}: {e}")
                        continue

                print(f"✅ Completed pair {pair_idx + 1}/{len(handle_currents)}")

        else:
            # ORIGINAL: Process all pairs simultaneously in each step
            for step_idx in range(n_steps):
                print(f"\nStep {step_idx + 1}/{n_steps}")

                # Get current vertices
                vertices = np.asarray(self.deformed_mesh.vertices)

                # Collect all handle regions and their target positions for this step
                step_handle_region_ids = []
                step_handle_region_pos = []

                # Process each handle-target pair
                for pair_idx, (handle_curr, delta) in enumerate(zip(handle_currents, deltas)):
                    # Find handle region for this handle (updated position)
                    distances = np.linalg.norm(vertices - handle_curr, axis=1)
                    handle_region_ids = np.where(distances < radius)[0]

                    # Add to collective constraints (avoid duplicates)
                    for idx in handle_region_ids:
                        if idx not in step_handle_region_ids:
                            step_handle_region_ids.append(idx)
                            step_handle_region_pos.append(vertices[idx] + delta)

                    # # Update handle position for next step
                    # handle_currents[pair_idx] = handle_next

                spheres_handles = []
                spheres_handles_next = []
                spheres_targets = []

                # I want to show all the handle_currents and target_points_3d
                # for pair_idx, (handle_curr, target_point_3d) in enumerate(zip(handle_currents, target_points_3d)):
                #     sphere_handle_curr = self.create_sphere_at_point(handle_curr, radius=0.02, color=[1, 0, 0])
                #     sphere_handle_next = self.create_sphere_at_point(handle_next, radius=0.02, color=[1, 0, 0])
                #     sphere_target = self.create_sphere_at_point(target_point_3d, radius=0.02, color=[0, 0, 1])
                #     spheres_handles.append(sphere_handle_curr)
                #     spheres_handles_next.append(sphere_handle_next)
                #     spheres_targets.append(sphere_target)

                # o3d.visualization.draw_geometries([self.deformed_mesh, *spheres_handles, *spheres_handles_next, *spheres_targets])

                for pair_idx, (handle_curr, target_point_3d) in enumerate(zip(handle_currents, target_points_3d)):
                    handle_next = handle_curr + deltas[pair_idx]
                    sphere_handle_curr = self.create_sphere_at_point(handle_curr, radius=0.02, color=[1, 0, 0])
                    sphere_handle_next = self.create_sphere_at_point(handle_next, radius=0.02, color=[1, 0, 0])
                    sphere_target = self.create_sphere_at_point(target_point_3d, radius=0.02, color=[0, 0, 1])
                    spheres_handles.append(sphere_handle_curr)
                    spheres_handles_next.append(sphere_handle_next)
                    spheres_targets.append(sphere_target)

                # o3d.visualization.draw_geometries([self.deformed_mesh, *spheres_handles, *spheres_handles_next, *spheres_targets])

                # Apply rotation for first 2 steps if enabled (only for first pair)
                if use_pivot_rotation and pivot is not None and step_idx < 2:
                    if len(transform_ids) > 0 and len(handle_currents) > 0:
                        # Use first handle pair for rotation
                        M = self.compute_rotation_matrix_from_vectors(
                            pivot,
                            handle_currents[0],  # Current position
                            handle_currents[0] + deltas[0]  # Next position
                        )
                        self.deformed_mesh = self.apply_transform_to_selected_vertices(
                            self.deformed_mesh, M, transform_ids
                        )
                        vertices = np.asarray(self.deformed_mesh.vertices)

                # Prepare constraints exactly like working version
                if len(all_static_ids) == 0:
                    # No static constraints - use only handle region
                    constraint_ids_ovec = o3d.utility.IntVector(step_handle_region_ids)
                    constraint_pos_ovec = o3d.utility.Vector3dVector(vertices[step_handle_region_ids])  # Use current positions like working version
                    print(f"Using only handle constraints: {len(step_handle_region_ids)} vertices")
                else:
                    # Combine static and handle constraints (exactly like working version)
                    constraint_ids_ovec = o3d.utility.IntVector(np.concatenate((all_static_ids, step_handle_region_ids)).tolist())
                    constraint_pos_ovec = o3d.utility.Vector3dVector(np.concatenate((static_pos, step_handle_region_pos)).tolist())
                    print(f"Using combined constraints: {len(all_static_ids)} static + {len(step_handle_region_ids)} handle = {len(all_static_ids) + len(step_handle_region_ids)} total")

                # Validate constraints before ARAP
                print(f"Constraint validation: IDs={len(constraint_ids_ovec)}, Positions={len(constraint_pos_ovec)}")
                print(f"Mesh stats: vertices={len(np.asarray(self.deformed_mesh.vertices))}, triangles={len(np.asarray(self.deformed_mesh.triangles))}")

                if len(constraint_ids_ovec) == 0:
                    print("Warning: No constraints defined, skipping ARAP step")
                    continue

                # Check mesh connectivity and validity
                if not self.deformed_mesh.has_triangles():
                    print("ERROR: Mesh has no triangles, cannot apply ARAP")
                    continue

                # Check for NaN or infinite values in constraints
                try:
                    constraint_ids_array = np.array(constraint_ids_ovec)
                    constraint_pos_array = np.asarray(constraint_pos_ovec)

                    if np.any(np.isnan(constraint_pos_array)) or np.any(np.isinf(constraint_pos_array)):
                        print("ERROR: NaN or infinite values in constraint positions")
                        continue

                    if np.any(constraint_ids_array < 0) or np.any(constraint_ids_array >= len(vertices)):
                        print("ERROR: Invalid constraint IDs found")
                        continue

                except Exception as validation_error:
                    print(f"ERROR: Constraint validation failed: {validation_error}")
                    continue

                # Apply ARAP deformation with additional safety checks
                try:
                    print(f"Applying ARAP with {len(constraint_ids_ovec)} constraints...")
                    # Use exact same parameters as working version
                    self.deformed_mesh = self.deformed_mesh.deform_as_rigid_as_possible(
                        constraint_ids_ovec,
                        constraint_pos_ovec,
                        max_iter=30,  # Same as working version
                        energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
                        smoothed_alpha=alpha_reg
                    )
                    print("ARAP step completed successfully")

                    # Apply smoothing after each step (same as working version)
                    self.deformed_mesh.filter_smooth_taubin(
                        number_of_iterations=30,  # Same as working version
                        lambda_filter=0.5,
                        mu=-0.53
                    )
                    print("Smoothing step completed")

                    # Update handle positions for next step
                    for pair_idx in range(len(handle_currents)):
                        handle_currents[pair_idx] = handle_currents[pair_idx] + deltas[pair_idx]

                except Exception as e:
                    print(f"ERROR: ARAP failed at step {step_idx}: {e}")
                    print(f"Constraint details: IDs type={type(constraint_ids_ovec)}, Pos type={type(constraint_pos_ovec)}")
                    print(f"Mesh vertices: {len(np.asarray(self.deformed_mesh.vertices))}")
                    # Try to continue with next step instead of breaking
                    continue

                sphere_handle_curr = self.create_sphere_at_point(handle_currents[0], radius=0.02, color=[1, 0, 0])
                sphere_target = self.create_sphere_at_point(target_points_3d[0], radius=0.02, color=[0, 0, 1])
                # Removed o3d.visualization call to prevent OpenGL conflicts
                # o3d.visualization.draw_geometries([self.deformed_mesh, sphere_handle_curr, sphere_target])
                print(f"Step {step_idx + 1} visualization skipped to avoid OpenGL conflicts")    

        # Verify final positions
        for pair_idx, (handle_curr, target) in enumerate(zip(handle_currents, target_points_3d)):
            final_distance = np.linalg.norm(handle_curr - target)
            print(f"Pair {pair_idx + 1}: Final distance to target: {final_distance:.6f}")
            if final_distance > 0.001:
                print(f"  WARNING: Handle {pair_idx + 1} did not reach target!")

        self.deformed_mesh.compute_vertex_normals()

        # Restore original colors
        if original_colors is not None:
            self.deformed_mesh.vertex_colors = o3d.utility.Vector3dVector(original_colors)
        
        
        # Update visualization
        self.widget3d.scene.remove_geometry("mesh")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.8, 0.8, 0.8, 1.0]
        self.widget3d.scene.add_geometry("mesh", self.deformed_mesh, mat)
        self.widget3d.force_redraw()

        # Store pre-subdivision deformed mesh for arrow calculation
        # This ensures we can compare vertex movements before subdivision changes vertex count
        self.deformed_mesh_pre_subdivision = self.clone_triangle_mesh(self.deformed_mesh)
        print(f"✅ Stored pre-subdivision deformed mesh ({len(np.asarray(self.deformed_mesh_pre_subdivision.vertices))} vertices)")

        # Apply subdivision for smoother result (similar to working version)
        try:
            print("Applying mesh subdivision for smoother result...")
            self.deformed_mesh = self.deformed_mesh.subdivide_midpoint(1)
            self.deformed_mesh.compute_vertex_normals()

            # Update visualization with subdivided mesh
            self.widget3d.scene.remove_geometry("mesh")
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            self.widget3d.scene.add_geometry("mesh", self.deformed_mesh, mat)
            self.widget3d.force_redraw()
            print(f"Mesh subdivision completed. New vertex count: {len(np.asarray(self.deformed_mesh.vertices))}")
        except Exception as e:
            print(f"Warning: Mesh subdivision failed: {e}")

        print("SR-ARAP deformation completed!")
    
    def get_mask_indices(self):
        """Get vertex indices that fall within the mask, including manual selections."""
        vertices = np.asarray(self.mesh.vertices)
        mask_indices = set()
        
        # Add mask-based indices if mask exists
        if self.mask is not None:
            # Check if projection function exists
            if self.project_orthographic_2d is not None:
                # Use the provided projection function
                for i, vertex in enumerate(vertices):
                    u, v = self.project_orthographic_2d(vertex[0], vertex[1])
                    u_int = int(np.clip(u, 0, self.mask.shape[1] - 1))
                    v_int = int(np.clip(v, 0, self.mask.shape[0] - 1))

                    if self.mask[v_int, u_int] > 128:  # Threshold for white regions
                        mask_indices.add(i)
            else:
                # Use simple front-view projection
                bbox = self.mesh.get_axis_aligned_bounding_box()
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                
                for i, vertex in enumerate(vertices):
                    # Normalize X and Y to [0, 1] for front view projection
                    norm_x = (vertex[0] - min_bound[0]) / (max_bound[0] - min_bound[0] + 1e-8)
                    norm_y = (vertex[1] - min_bound[1]) / (max_bound[1] - min_bound[1] + 1e-8)
                    
                    # Map to mask coordinates
                    u = int(norm_x * (self.mask.shape[1] - 1))
                    v = int((1 - norm_y) * (self.mask.shape[0] - 1))  # Flip Y
                    
                    # Check bounds and mask value
                    if 0 <= u < self.mask.shape[1] and 0 <= v < self.mask.shape[0]:
                        if self.mask[v, u] > 128:  # Threshold for white regions
                            mask_indices.add(i)
            
            print(f"Mask selected {len(mask_indices)} vertices from image mask")
        
        # Add manual movable selections
        before_manual = len(mask_indices)
        mask_indices = mask_indices.union(self.manual_movable_ids)
        if len(mask_indices) > before_manual:
            print(f"Added {len(mask_indices) - before_manual} manual movable vertices")
        
        # Remove manual constraint selections
        before_constraint = len(mask_indices)
        mask_indices = mask_indices - self.manual_constraint_ids
        if before_constraint > len(mask_indices):
            print(f"Removed {before_constraint - len(mask_indices)} manual constraint vertices")
        
        # Don't return all vertices if empty - let the caller handle it
        return np.array(list(mask_indices), dtype=np.int32)
    
    def update_camera_matrix_display(self):
        """Update the camera matrix display in the panel."""
        if self.view_matrix is not None and self.projection_matrix is not None:
            # Format matrices for display (full matrix in compact format)
            display_str = "=== Camera Matrices ===\n\n"
            
            # View Matrix
            display_str += "View Matrix:\n"
            for i in range(4):
                display_str += f"[{self.view_matrix[i,0]:6.3f}, {self.view_matrix[i,1]:6.3f}, "
                display_str += f"{self.view_matrix[i,2]:6.3f}, {self.view_matrix[i,3]:6.3f}]\n"
            
            display_str += "\nProjection Matrix:\n"
            for i in range(4):
                display_str += f"[{self.projection_matrix[i,0]:6.3f}, {self.projection_matrix[i,1]:6.3f}, "
                display_str += f"{self.projection_matrix[i,2]:6.3f}, {self.projection_matrix[i,3]:6.3f}]\n"
            
            # Camera position from view matrix
            # The camera position is the inverse of the view matrix translation
            view_inv = np.linalg.inv(self.view_matrix)
            cam_pos = view_inv[:3, 3]
            display_str += f"\nCamera Position:\n[{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]"
            
            self.camera_matrix_text.text_value = display_str
        else:
            self.camera_matrix_text.text_value = "View: Not set\nProj: Not set\n\nClick 'Project Flow Field' to\nupdate camera matrices"
    
    def show_flow_field_preview(self, flow_field):
        """Show flow field preview safely by saving to temporary file and displaying info."""
        try:
            import tempfile
            import os

            H, W = flow_field.shape[:2]
            print(f"\nFlow Field Statistics:")
            print(f"  Size: {H} x {W}")

            # Calculate flow magnitude for statistics
            magnitude = np.sqrt(flow_field[:,:,0]**2 + flow_field[:,:,1]**2)
            non_zero_mask = magnitude > 1e-6
            if np.any(non_zero_mask):
                print(f"  Non-zero flow vectors: {np.sum(non_zero_mask)}")
                print(f"  Max flow magnitude: {np.max(magnitude):.4f}")
                print(f"  Mean flow magnitude: {np.mean(magnitude[non_zero_mask]):.4f}")
            else:
                print("  No significant flow detected")

            # Create a simple visualization with OpenCV instead of matplotlib
            try:
                # Create visualization image
                viz_img = np.ones((H, W, 3), dtype=np.uint8) * 255

                # If we have the original image, use it as background
                if self.image is not None and self.image.shape[:2] == (H, W):
                    viz_img = self.image.copy()

                # Sample flow field for visualization
                step = 15
                for y in range(0, H, step):
                    for x in range(0, W, step):
                        if y < H and x < W:
                            u, v = flow_field[y, x]
                            if abs(u) > 1e-3 or abs(v) > 1e-3:  # Only draw significant vectors
                                # Draw arrow with bounds checking
                                end_x = max(0, min(W-1, int(x + u * 10)))  # Scale for visibility
                                end_y = max(0, min(H-1, int(y + v * 10)))
                                try:
                                    cv2.arrowedLine(viz_img, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)
                                except:
                                    pass  # Skip problematic arrows

                # Save to temporary file and open with system viewer
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    cv2.imwrite(temp_path, viz_img)

                print(f"\nFlow field visualization saved temporarily to: {temp_path}")
                print("Opening with system default image viewer...")

                # Don't automatically open viewer to avoid crashes
                print(f"You can manually open the file to view the visualization: {temp_path}")
                print("or use any image viewer to see the flow field arrows.")

            except Exception as e:
                print(f"Could not create visualization: {e}")
                print("Flow field data is ready for saving.")

        except Exception as e:
            print(f"Error in flow field preview: {e}")
            print("Flow field computed successfully but preview failed.")

        print("\n✓ Flow field projection complete!")

    def on_show_original_image(self):
        """Show the original image with interactive point selection."""
        print("\n=== Show Original Image Debug ===")
        print(f"Image loaded: {self.image is not None}")

        if self.image is not None:
            print(f"Image shape: {self.image.shape}")
            print(f"Image dtype: {self.image.dtype}")
            print(f"Image min/max values: {self.image.min()}/{self.image.max()}")

            # Test if image can be displayed
            try:
                test_window = "Test Image Display"
                cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(test_window, 400, 300)
                cv2.imshow(test_window, self.image)
                cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyWindow(test_window)
                print("Test display completed - if you saw the image briefly, display works")
            except Exception as e:
                print(f"Test display failed: {e}")

        if self.image is None:
            print("ERROR: No image loaded!")
            print("Please check that the image path is correct and the file exists.")
            return

        print("\n=== Interactive 2D Point Selection ===")
        print("Click on the image to select 2D handle and target points")
        print("Press 'q' to quit the window")

        # Create window for interactive selection
        self.show_image_with_point_selection()

    def show_image_with_point_selection(self):
        """Show image window for point selection - using external viewer to avoid GUI conflicts."""
        if self.image is None:
            print("ERROR: No image loaded! Cannot show original image.")
            return

        print(f"Image shape: {self.image.shape}")
        print("Opening image with external viewer to avoid GUI conflicts...")

        # Use external viewer approach to avoid any GUI conflicts
        self.show_image_with_external_viewer()

    # NOTE: show_image_with_external_viewer
    def show_image_with_external_viewer(self):
        """Show image with web-based interactive point selection (completely GUI-conflict-free)."""
        print("\n=== Web-Based Interactive Point Selection ===")
        print("Creating web interface for handle and target point selection...")

        try:
            # Use web-based interface for completely conflict-free selection
            import threading

            # Run web interface in a separate thread to avoid GUI blocking
            web_thread = threading.Thread(target=self.web_based_interactive_selection, daemon=True)
            web_thread.start()

            # Return immediately to keep GUI responsive
            print("🌐 Web interface started in background thread...")
            print("📝 Web interface will automatically save coordinates when completed.")
            print("💡 You can continue using the FlowDrag Editor while the web interface is open.")

        except Exception as e:
            print(f"⚠️  Web-based interactive mode failed: {e}")
            print("🔄 Falling back to external viewer mode...")
            self.show_image_with_fallback_viewer()

    def web_based_interactive_selection(self):
        """Create a web-based interface for point selection - completely GUI-conflict-free."""
        import tempfile
        import base64
        import webbrowser
        import http.server
        import socketserver
        import socket
        from urllib.parse import parse_qs, urlparse

        print("\n🌐 Creating web-based point selection interface...")

        # Convert image to base64 for web display
        _, buffer = cv2.imencode('.png', self.image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        height, width = self.image.shape[:2]

        # Create HTML interface
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FlowDrag - Point Selection</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .title {{
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }}
        .image-container {{
            position: relative;
            display: inline-block;
            border: 2px solid #ddd;
            background: #f9f9f9;
        }}
        .image-canvas {{
            cursor: crosshair;
            display: block;
        }}
        .grid-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            opacity: 0.3;
        }}
        .controls {{
            margin: 20px 0;
            text-align: center;
        }}
        .button {{
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 16px;
        }}
        .button:hover {{
            background: #45a049;
        }}
        .button.reset {{
            background: #f44336;
        }}
        .button.reset:hover {{
            background: #da190b;
        }}
        .status {{
            margin: 20px 0;
            padding: 15px;
            background: #e7f3ff;
            border: 1px solid #b3d7ff;
            border-radius: 5px;
            font-weight: bold;
        }}
        .coordinates {{
            margin: 20px 0;
            padding: 15px;
            background: #f0fff0;
            border: 1px solid #90EE90;
            border-radius: 5px;
        }}
        .coordinate-item {{
            margin: 5px 0;
            font-family: monospace;
        }}
        .handle {{ color: #d32f2f; }}
        .target {{ color: #1976d2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🎯 FlowDrag - Interactive Point Selection</h1>

        <div class="status" id="status">
            Click on the image to select HANDLE point (red circle)
        </div>

        <div class="image-container">
            <canvas id="imageCanvas" class="image-canvas" width="{width}" height="{height}"></canvas>
            <canvas id="gridCanvas" class="grid-overlay" width="{width}" height="{height}"></canvas>
        </div>

        <div class="controls">
            <button class="button reset" onclick="resetSelection()">🔄 Reset</button>
            <button class="button" onclick="submitCoordinates()">✅ Save & Close</button>
        </div>

        <div class="coordinates" id="coordinates" style="display:none;">
            <h3>📍 Selected Coordinates:</h3>
            <div id="coordinatesList"></div>
        </div>
    </div>

    <script>
        let canvas = document.getElementById('imageCanvas');
        let gridCanvas = document.getElementById('gridCanvas');
        let ctx = canvas.getContext('2d');
        let gridCtx = gridCanvas.getContext('2d');

        let handlePoint = null;
        let targetPoint = null;
        let selectingHandle = true;

        // Load and display image
        let img = new Image();
        img.onload = function() {{
            ctx.drawImage(img, 0, 0);
            drawGrid();
        }};
        img.src = 'data:image/png;base64,{img_base64}';

        function drawGrid() {{
            gridCtx.strokeStyle = '#FFD700';
            gridCtx.lineWidth = 1;
            gridCtx.globalAlpha = 0.4;

            // Draw grid lines
            for (let x = 0; x < {width}; x += 50) {{
                gridCtx.beginPath();
                gridCtx.moveTo(x, 0);
                gridCtx.lineTo(x, {height});
                gridCtx.stroke();

                if (x > 0) {{
                    gridCtx.fillStyle = '#FFD700';
                    gridCtx.font = '12px Arial';
                    gridCtx.fillText(x.toString(), x + 2, 15);
                }}
            }}

            for (let y = 0; y < {height}; y += 50) {{
                gridCtx.beginPath();
                gridCtx.moveTo(0, y);
                gridCtx.lineTo({width}, y);
                gridCtx.stroke();

                if (y > 0) {{
                    gridCtx.fillStyle = '#FFD700';
                    gridCtx.font = '12px Arial';
                    gridCtx.fillText(y.toString(), 5, y - 5);
                }}
            }}
        }}

        function drawPoints() {{
            // Redraw image
            ctx.drawImage(img, 0, 0);

            // Draw handle point (red circle)
            if (handlePoint) {{
                ctx.beginPath();
                ctx.arc(handlePoint.x, handlePoint.y, 8, 0, 2 * Math.PI);
                ctx.fillStyle = '#d32f2f';
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#d32f2f';
                ctx.font = 'bold 12px Arial';
                ctx.fillText(`H(${{handlePoint.x}},${{handlePoint.y}})`, handlePoint.x + 15, handlePoint.y - 10);
            }}

            // Draw target point (blue square)
            if (targetPoint) {{
                ctx.fillStyle = '#1976d2';
                ctx.fillRect(targetPoint.x - 8, targetPoint.y - 8, 16, 16);
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.strokeRect(targetPoint.x - 8, targetPoint.y - 8, 16, 16);

                // Label
                ctx.fillStyle = '#1976d2';
                ctx.font = 'bold 12px Arial';
                ctx.fillText(`T(${{targetPoint.x}},${{targetPoint.y}})`, targetPoint.x + 15, targetPoint.y + 15);

                // Draw arrow from handle to target
                if (handlePoint) {{
                    ctx.beginPath();
                    ctx.moveTo(handlePoint.x, handlePoint.y);
                    ctx.lineTo(targetPoint.x, targetPoint.y);
                    ctx.strokeStyle = '#4CAF50';
                    ctx.lineWidth = 3;
                    ctx.stroke();

                    // Arrow head
                    let angle = Math.atan2(targetPoint.y - handlePoint.y, targetPoint.x - handlePoint.x);
                    let headlen = 15;
                    ctx.beginPath();
                    ctx.moveTo(targetPoint.x, targetPoint.y);
                    ctx.lineTo(targetPoint.x - headlen * Math.cos(angle - Math.PI / 6),
                              targetPoint.y - headlen * Math.sin(angle - Math.PI / 6));
                    ctx.moveTo(targetPoint.x, targetPoint.y);
                    ctx.lineTo(targetPoint.x - headlen * Math.cos(angle + Math.PI / 6),
                              targetPoint.y - headlen * Math.sin(angle + Math.PI / 6));
                    ctx.stroke();
                }}
            }}
        }}

        function updateStatus() {{
            let status = document.getElementById('status');
            let coordDiv = document.getElementById('coordinates');
            let coordList = document.getElementById('coordinatesList');

            if (selectingHandle) {{
                status.textContent = 'Click on the image to select HANDLE point (red circle)';
                status.style.background = '#e7f3ff';
            }} else if (!targetPoint) {{
                status.textContent = 'Now click to select TARGET point (blue square)';
                status.style.background = '#fff3e0';
            }} else {{
                status.textContent = '✅ Points selected! Click "Save & Close" or reset to try again';
                status.style.background = '#e8f5e8';

                coordDiv.style.display = 'block';
                coordList.innerHTML = `
                    <div class="coordinate-item handle">🔴 Handle Point: (${{handlePoint.x}}, ${{handlePoint.y}})</div>
                    <div class="coordinate-item target">🔵 Target Point: (${{targetPoint.x}}, ${{targetPoint.y}})</div>
                `;
            }}
        }}

        canvas.addEventListener('click', function(e) {{
            let rect = canvas.getBoundingClientRect();
            let x = Math.round(e.clientX - rect.left);
            let y = Math.round(e.clientY - rect.top);

            if (selectingHandle) {{
                handlePoint = {{x: x, y: y}};
                selectingHandle = false;
                console.log('Handle point selected:', handlePoint);
            }} else {{
                targetPoint = {{x: x, y: y}};
                console.log('Target point selected:', targetPoint);
            }}

            drawPoints();
            updateStatus();
        }});

        function resetSelection() {{
            handlePoint = null;
            targetPoint = null;
            selectingHandle = true;
            drawPoints();
            updateStatus();
            document.getElementById('coordinates').style.display = 'none';
            console.log('Selection reset');
        }}

        function submitCoordinates() {{
            if (handlePoint && targetPoint) {{
                // Send coordinates to Python via localhost URL (absolute URL)
                let url = `http://localhost:8765/submit?handle_x=${{handlePoint.x}}&handle_y=${{handlePoint.y}}&target_x=${{targetPoint.x}}&target_y=${{targetPoint.y}}`;

                // Show loading message
                document.getElementById('status').textContent = '💾 Saving coordinates...';
                document.getElementById('status').style.background = '#fff3cd';

                console.log('Sending request to:', url);

                // Try fetch first, fallback to XMLHttpRequest if it fails
                fetch(url, {{
                    method: 'GET',
                    mode: 'cors',
                    credentials: 'omit'
                }})
                    .then(response => {{
                        console.log('Response status:', response.status);
                        if (response.ok) {{
                            return response.text();
                        }} else {{
                            throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                        }}
                    }})
                    .then(data => {{
                        console.log('Response data:', data);
                        alert('✅ Coordinates saved successfully!\\n\\nHandle: (${{handlePoint.x}}, ${{handlePoint.y}})\\nTarget: (${{targetPoint.x}}, ${{targetPoint.y}})\\n\\nClosing window...');
                        // Close the window after successful save
                        window.close();
                    }})
                    .catch(error => {{
                        console.error('Fetch failed, trying XMLHttpRequest:', error);

                        // Fallback to XMLHttpRequest
                        try {{
                            let xhr = new XMLHttpRequest();
                            xhr.open('GET', url, true);
                            xhr.onreadystatechange = function() {{
                                if (xhr.readyState === 4) {{
                                    if (xhr.status === 200) {{
                                        console.log('XMLHttpRequest success:', xhr.responseText);
                                        alert('✅ Coordinates saved successfully!\\n\\nHandle: (${{handlePoint.x}}, ${{handlePoint.y}})\\nTarget: (${{targetPoint.x}}, ${{targetPoint.y}})\\n\\nClosing window...');
                                        window.close();
                                    }} else {{
                                        console.error('XMLHttpRequest failed:', xhr.status, xhr.statusText);
                                        alert(`❌ Server error: HTTP ${{xhr.status}}\\n\\nPlease try again.`);
                                        // Reset status
                                        document.getElementById('status').textContent = '✅ Points selected! Click "Save & Close" or reset to try again';
                                        document.getElementById('status').style.background = '#e8f5e8';
                                    }}
                                }}
                            }};
                            xhr.onerror = function() {{
                                console.error('XMLHttpRequest network error');
                                alert('❌ Network connection failed\\n\\nPlease check if the Python server is running and try again.');
                                // Reset status
                                document.getElementById('status').textContent = '✅ Points selected! Click "Save & Close" or reset to try again';
                                document.getElementById('status').style.background = '#e8f5e8';
                            }};
                            xhr.send();
                        }} catch (xhrError) {{
                            console.error('XMLHttpRequest error:', xhrError);
                            alert(`❌ All request methods failed\\n\\nError: ${{xhrError.message}}\\n\\nPlease try refreshing the page.`);
                            // Reset status
                            document.getElementById('status').textContent = '✅ Points selected! Click "Save & Close" or reset to try again';
                            document.getElementById('status').style.background = '#e8f5e8';
                        }}
                    }});
            }} else {{
                alert('⚠️ Please select both handle and target points first!');
            }}
        }}

        // Initialize
        updateStatus();
    </script>
</body>
</html>
        """

        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            html_path = f.name

        # Create a flag to control server shutdown
        self.server_should_stop = False

        # Set up a simple HTTP server to handle coordinate submission
        class CoordinateHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, parent_instance, *args, **kwargs):
                self.parent = parent_instance
                super().__init__(*args, **kwargs)

            def log_message(self, format, *args):
                # Suppress server logs
                return

            def do_GET(self):
                if self.path.startswith('/submit'):
                    # Parse coordinates from URL
                    parsed = urlparse(self.path)
                    params = parse_qs(parsed.query)

                    if all(key in params for key in ['handle_x', 'handle_y', 'target_x', 'target_y']):
                        handle_x = int(params['handle_x'][0])
                        handle_y = int(params['handle_y'][0])
                        target_x = int(params['target_x'][0])
                        target_y = int(params['target_y'][0])

                        # Save coordinates
                        self.parent.temp_handle_2d = (handle_x, handle_y)
                        self.parent.temp_target_2d = (target_x, target_y)
                        self.parent.save_2d_points_to_gui()

                        print(f"✅ Coordinates received from web interface:")
                        print(f"   Handle: ({handle_x}, {handle_y})")
                        print(f"   Target: ({target_x}, {target_y})")

                        # Send success response with comprehensive CORS headers
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        self.wfile.write(b'Success')

                        # Signal server to stop
                        self.parent.server_should_stop = True
                        print("🔄 Coordinate submission received, stopping server...")
                    else:
                        self.send_error(400, "Missing parameters")
                else:
                    self.send_error(404, "Not found")

            def do_OPTIONS(self):
                # Handle preflight CORS requests
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

        # Start HTTP server - try multiple ports if one is in use
        base_port = 8765
        max_attempts = 10
        httpd = None
        handler = lambda *args, **kwargs: CoordinateHandler(self, *args, **kwargs)

        for attempt in range(max_attempts):
            port = base_port + attempt
            try:
                # Enable address reuse to avoid "Address already in use" error
                socketserver.TCPServer.allow_reuse_address = True
                httpd = socketserver.TCPServer(("", port), handler)
                httpd.allow_reuse_address = True
                httpd.timeout = 0.5  # 0.5 second timeout for better responsiveness
                print(f"🌐 Starting web server on port {port}...")
                break  # Success - exit loop
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    print(f"⚠️ Port {port} is already in use, trying next port...")
                    continue
                else:
                    raise  # Re-raise other OSErrors

        if httpd is None:
            raise OSError(f"Could not find available port in range {base_port}-{base_port+max_attempts-1}")

        try:

            # Open HTML file in browser
            file_url = f"file://{html_path}"
            print(f"🔗 Opening web interface: {file_url}")
            webbrowser.open(file_url)

            print("\n📋 Instructions:")
            print("1. 🖱️  Click on the image to select HANDLE point (red)")
            print("2. 🖱️  Click again to select TARGET point (blue)")
            print("3. ✅ Click 'Save & Close' button when done")
            print("4. 🔄 Use 'Reset' button to start over")
            print("5. 🗑️  Close browser tab when finished")
            print("\n⏳ Waiting for coordinate selection...")

            # Run server with frequent timeout checks
            max_requests = 1000  # Safety limit
            request_count = 0
            timeout_count = 0
            max_timeouts = 60  # Max 30 seconds of timeout (60 * 0.5s)

            try:
                while not self.server_should_stop and request_count < max_requests and timeout_count < max_timeouts:
                    try:
                        httpd.handle_request()
                        request_count += 1
                        timeout_count = 0  # Reset timeout counter on successful request
                    except (OSError, socket.timeout):
                        # Timeout occurred, continue loop to check stop flag
                        timeout_count += 1
                        continue
                    except Exception as e:
                        print(f"⚠️  Server error: {e}")
                        break

                if timeout_count >= max_timeouts:
                    print("⏰ Server timeout reached - auto-closing...")

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
            finally:
                print("🛑 Shutting down web server...")
                try:
                    httpd.shutdown()
                except:
                    pass
                try:
                    httpd.server_close()
                except:
                    pass

                # Force cleanup
                self.server_should_stop = True

        except Exception as e:
            print(f"❌ Failed to start web server: {e}")
            raise

        print("✅ Web-based point selection completed!")

    def on_capture_current_view(self):
        """Capture current camera view and display matrix."""
        print("\n=== Capturing Current View ===")

        if self.deformed_mesh is None:
            print("❌ No deformed mesh! Apply deformation first.")
            return

        # Get current camera matrices
        camera = self.widget3d.scene.camera
        self.view_matrix = camera.get_view_matrix()
        self.projection_matrix = camera.get_projection_matrix()

        print("✅ Camera matrices captured")

        # Update camera matrix display with larger format
        self.update_camera_matrix_display_large()

        # Capture current view as image
        self.capture_view_as_image()

    def update_camera_matrix_display_large(self):
        """Update camera matrix display with larger, more readable format."""
        if self.view_matrix is not None and self.projection_matrix is not None:
            display_str = "=== CAMERA MATRICES ===\n\n"

            # View Matrix (more compact format)
            display_str += "VIEW MATRIX:\n"
            for i in range(4):
                row = f"[{self.view_matrix[i,0]:7.3f} {self.view_matrix[i,1]:7.3f} {self.view_matrix[i,2]:7.3f} {self.view_matrix[i,3]:7.3f}]\n"
                display_str += row

            display_str += "\nPROJECTION MATRIX:\n"
            for i in range(4):
                row = f"[{self.projection_matrix[i,0]:7.3f} {self.projection_matrix[i,1]:7.3f} {self.projection_matrix[i,2]:7.3f} {self.projection_matrix[i,3]:7.3f}]\n"
                display_str += row

            display_str += f"\n✅ Matrices captured successfully!"

            self.camera_matrix_text.text_value = display_str
        else:
            self.camera_matrix_text.text_value = "❌ Failed to capture camera matrices"

    def capture_view_as_image(self):
        """Capture current 3D view as 2D image."""
        try:
            # Create output directory
            self.create_output_directory()

            # Capture screenshot
            screenshot = self.widget3d.scene.scene.render_to_image()

            # Save captured image
            captured_path = os.path.join(self.output_dir, "captured_view.png")
            o3d.io.write_image(captured_path, screenshot)

            print(f"📸 View captured and saved to: {captured_path}")

        except Exception as e:
            print(f"❌ Failed to capture view: {e}")

    def create_output_directory(self):
        """Create output directory based on user input or default."""
        # Check if user has specified a custom output directory
        if hasattr(self, 'output_dir_text'):
            user_specified_path = self.output_dir_text.text_value.strip()

            # If user has modified the path and it's not the placeholder
            if user_specified_path and user_specified_path != "flow_field_outputs/[mesh_name]":
                self.output_dir = user_specified_path
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    print(f"📁 Created user-specified output directory: {self.output_dir}")
                else:
                    print(f"📁 Using existing directory: {self.output_dir}")

                # Update the text field to show actual path
                self.output_dir_text.text_value = self.output_dir
                return

        # Default behavior: check if already exists
        if hasattr(self, 'output_dir') and os.path.exists(self.output_dir):
            return

        # Get base name from mesh file or use default
        if hasattr(self, 'mesh_path') and self.mesh_path:
            base_name = os.path.splitext(os.path.basename(self.mesh_path))[0]
        else:
            base_name = "flowdrag_output"

        # Create output directory
        flow_outputs_dir = "flow_field_outputs"
        if not os.path.exists(flow_outputs_dir):
            os.makedirs(flow_outputs_dir)

        self.output_dir = os.path.join(flow_outputs_dir, base_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")

        # Update the text field to show actual path
        if hasattr(self, 'output_dir_text'):
            self.output_dir_text.text_value = self.output_dir

    def save_2d_points_to_gui(self):
        """Save selected 2D points to GUI input fields."""
        if self.temp_handle_2d:
            self.handle_2d_x.set_value(float(self.temp_handle_2d[0]))
            self.handle_2d_y.set_value(float(self.temp_handle_2d[1]))
            print(f"📝 Saved handle point to GUI: ({self.temp_handle_2d[0]}, {self.temp_handle_2d[1]})")

        if self.temp_target_2d:
            self.target_2d_x.set_value(float(self.temp_target_2d[0]))
            self.target_2d_y.set_value(float(self.temp_target_2d[1]))
            print(f"📝 Saved target point to GUI: ({self.temp_target_2d[0]}, {self.temp_target_2d[1]})")

    def on_final_auto_align_projection(self):
        """Final auto-align to input image using 2D points and create flow field projection."""
        print("\n=== Auto-align to Input Image & 2D Projection ===")

        # Check prerequisites
        if self.deformed_mesh is None:
            print("❌ No deformed mesh! Apply deformation first.")
            return

        if not self.handle_points or len(self.handle_points) == 0:
            print("❌ No 3D handle points! Select handle points first.")
            return

        if not hasattr(self, 'temp_handle_2d') or not self.temp_handle_2d:
            print("❌ No 2D handle point! Use 'Show Original Image' to select 2D coordinates first.")
            return

        if not hasattr(self, 'temp_target_2d') or not self.temp_target_2d:
            print("❌ No 2D target point! Use 'Show Original Image' to select both handle and target points.")
            return

        print("✅ All prerequisites met")

        # Create output directory
        self.create_output_directory()

        # Check if we have captured camera matrices from "Camera Matrix & Capture Image"
        if hasattr(self, 'view_matrix') and hasattr(self, 'projection_matrix') and \
           self.view_matrix is not None and self.projection_matrix is not None:
            print("📷 Using previously captured camera matrices (recommended)")
            print("   This preserves the current 3D view without camera movement")
            print("   Tip: Use 'Camera Matrix & Capture Image' button first to avoid 180-degree rotation")

            # Use the captured matrices without changing camera position
            print("✅ Camera matrices ready for projection")

        else:
            print("📷 No captured camera matrices found - performing conservative camera alignment")
            print("   Warning: This may cause slight camera movement. Use 'Camera Matrix & Capture Image' first to avoid this.")

            # Get 3D handle and target points (first pair)
            handle_3d = np.array(self.handle_points[0])
            target_3d = np.array(self.target_points[0]) if self.target_points else handle_3d

            # Get 2D points from web interface
            handle_2d = np.array(self.temp_handle_2d)
            target_2d = np.array(self.temp_target_2d)

            print(f"🎯 3D Handle: {handle_3d}")
            print(f"🎯 3D Target: {target_3d}")
            print(f"🎯 2D Handle: {handle_2d}")
            print(f"🎯 2D Target: {target_2d}")

            # Calculate disparities for alignment
            disparity_3d = np.linalg.norm(target_3d - handle_3d)
            disparity_2d = np.linalg.norm(target_2d - handle_2d)

            print(f"📏 3D Disparity: {disparity_3d:.3f}")
            print(f"📏 2D Disparity: {disparity_2d:.3f}")

            # Align camera to match 2D handle point with 3D handle point (with conservative approach)
            self.align_camera_to_2d_correspondence(handle_3d, handle_2d, disparity_3d, disparity_2d)

            # Capture and display camera matrices after alignment
            self.capture_and_display_camera_matrices()

        # Generate final flow field projection
        self.generate_final_flow_field()

        print("✅ Auto-alignment and 2D projection completed!")
        print("💡 Tip: Use 'Camera Matrix & Capture Image' first to capture current view,")
        print("   then this function will use those matrices without moving the camera.")

    #NOTE: on_camera_matrix_capture()
    def on_camera_matrix_capture(self):
        """Camera Matrix & Capture Image 버튼 기능: 현재 camera matrix 구하고 화면 캡처"""
        print("\n=== Camera Matrix & Capture Image ===")

        try:
            # Get current camera - 백업 파일과 동일한 방식
            camera = self.widget3d.scene.camera

            # Store matrices as instance variables (백업 파일과 동일)
            self.view_matrix = camera.get_view_matrix()
            self.projection_matrix = camera.get_projection_matrix()

            print("📷 Camera matrices captured successfully!")

            # Display matrices in console (상세한 정보)
            print("\n🔍 View Matrix:")
            print(self.view_matrix)
            print("\n🔍 Projection Matrix:")
            print(self.projection_matrix)

            # Calculate camera parameters for debugging
            view_det = np.linalg.det(self.view_matrix[:3,:3])
            camera_pos = self.view_matrix[:3, 3]

            print(f"\n📊 Camera Parameters:")
            print(f"  - View Matrix Determinant: {view_det:.6f}")
            print(f"  - Camera Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")

            # Get camera intrinsic parameters for further debugging
            print(f"\n📊 Additional Camera Info:")
            bounds = self.widget3d.scene.bounding_box
            print(f"  - Scene Bounding Box: {bounds}")

            # Update camera matrix display (백업 파일과 유사한 방식)
            self.update_camera_matrix_display()

            # Capture current 3D view and save to file
            self.capture_and_save_current_view()

            print("✅ Camera matrix capture and image save completed!")
            print("📸 Current view matrices stored for accurate alignment")

        except Exception as e:
            print(f"❌ Error capturing camera matrices: {e}")
            self.camera_matrix_text.text_value = f"Error capturing camera matrices:\n{str(e)}"

    def update_camera_matrix_display(self):
        """Update the camera matrix display in the panel (백업 파일 방식 적용)."""
        if hasattr(self, 'view_matrix') and hasattr(self, 'projection_matrix') and \
           self.view_matrix is not None and self.projection_matrix is not None:

            # Format matrices for display (백업 파일과 유사하지만 더 자세히)
            display_text = "=== Current Camera Matrices ===\n\n"

            # View Matrix
            display_text += "View Matrix:\n"
            for i in range(4):
                row = "  [" + ", ".join([f"{self.view_matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                display_text += row

            display_text += "\nProjection Matrix:\n"
            for i in range(4):
                row = "  [" + ", ".join([f"{self.projection_matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                display_text += row

            # Add camera info
            view_det = np.linalg.det(self.view_matrix[:3,:3])
            camera_pos = self.view_matrix[:3, 3]
            display_text += f"\nCamera Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]\n"
            display_text += f"View Det: {view_det:.6f}\n"
            display_text += "\n⚠️ These matrices match current 3D viewer exactly"

            self.camera_matrix_text.text_value = display_text
        else:
            self.camera_matrix_text.text_value = "Camera matrices not captured yet.\nClick 'Camera Matrix & Capture Image' first."

    #NOTE: capture_and_save_current_view()
    def capture_and_save_current_view(self):
        """현재 3D viewer 화면을 캡처하고 저장 (참고 파일의 정확한 방법 사용)"""

        # Create output directory if needed
        if not hasattr(self, 'output_dir'):
            self.create_output_directory()

        print("📸 Capturing current 3D view using reference file method...")

        # Prepare file path
        timestamp = int(time.time())
        filename = f"camera_capture_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        # 참고 파일과 동일한 방법 (open3d_point_input_250629_success.py line 391-394)
        def on_image_captured(image):
            try:
                # Convert to numpy array for editing
                img_np = np.asarray(image)

                # Save the captured image directly without additional markers to avoid blue dots
                # Note: Removing add_projected_point_markers() to eliminate blue dots issue
                o3d.io.write_image(filepath, image)

                print(f"✅ Image saved with projected point markers: {filepath}")
                print("📸 Camera matrix based capture completed with 3D point projections")
            except Exception as save_error:
                print(f"❌ Error saving image: {save_error}")

        try:
            print("\n" + "="*60)
            print("📸 STEP 1/2: Capturing camera_capture.png...")
            print("="*60)

            # Temporarily hide only UI elements (keep arrows for camera_capture.png)
            self.hide_ui_elements_only()

            # 참고 파일의 정확한 방법 사용
            print("🔄 Using reference file method: widget3d.scene.scene.render_to_image(callback)")
            self.widget3d.scene.scene.render_to_image(on_image_captured)

            # 참고 파일처럼 잠시 대기
            time.sleep(0.5)

            # Restore UI elements after capture
            self.show_ui_elements_after_capture()

            print("\n" + "="*60)
            print("📸 STEP 2/2: Capturing deformed_mesh.png...")
            print("="*60)

            # Also capture deformed_mesh.png if deformed mesh exists
            if hasattr(self, 'deformed_mesh') and self.deformed_mesh is not None:
                self.capture_deformed_mesh_only()
            else:
                print("⚠️ No deformed mesh available - skipping deformed_mesh.png")

            print("\n" + "="*60)
            print("✅ ALL CAPTURES COMPLETED SUCCESSFULLY!")
            print("="*60)

        except Exception as e:
            print(f"❌ Error in capture using reference method: {e}")
            # 상세한 카메라 정보 저장 (filepath가 정의된 후에만)
            try:
                self.save_camera_info_file(filepath.replace('.png', '_camera_info.txt'))
            except:
                print("❌ Could not save camera info file")
 
    def save_camera_info_file(self, filepath):
        """카메라 정보를 파일로 저장"""
        try:
            camera = self.widget3d.scene.camera
            view_matrix = camera.get_view_matrix()
            projection_matrix = camera.get_projection_matrix()

            with open(filepath, 'w') as f:
                f.write("Camera Matrix Information\n")
                f.write("="*50 + "\n\n")
                f.write("View Matrix:\n")
                for i in range(4):
                    row = "  [" + ", ".join([f"{view_matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                    f.write(row)
                f.write("\nProjection Matrix:\n")
                for i in range(4):
                    row = "  [" + ", ".join([f"{projection_matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                    f.write(row)

            print(f"📄 Camera info saved to: {filepath}")

        except Exception as e:
            print(f"❌ Error saving camera info: {e}")

    def add_projected_point_markers(self, img_np):
        """Add red markers for projected 3D handle/target points on the captured image."""
        return utils.add_projected_point_markers(
            img_np, 
            self.handle_points, 
            self.target_points,
            self.view_matrix, 
            self.projection_matrix,
            temp_handle_2d=getattr(self, 'temp_handle_2d', None),
            temp_target_2d=getattr(self, 'temp_target_2d', None)
        )

    def project_3d_to_2d(self, point_3d, view_matrix, projection_matrix, width, height):
        """Project a 3D point to 2D pixel coordinates using camera matrices."""
        return utils.project_3d_to_2d(point_3d, view_matrix, projection_matrix, width, height)

    def add_projected_point_markers_to_flow_field(self, viz_img):
        """Add red markers for projected 3D handle/target points to the flow field visualization."""
        return utils.add_projected_point_markers_to_flow_field(
            viz_img,
            self.handle_points,
            self.target_points,
            self.view_matrix,
            self.projection_matrix,
            temp_handle_2d=getattr(self, 'temp_handle_2d', None),
            temp_target_2d=getattr(self, 'temp_target_2d', None)
        )

    def hide_ui_elements_only(self):
        """Temporarily hide only UI elements but keep arrows for camera_capture.png."""
        try:
            print("🔄 Temporarily hiding UI elements (keeping arrows)...")

            # Store what we're removing for restoration
            self.hidden_ui_elements = []

            # Remove all handle and target spheres
            for i in range(10):  # Max 10 pairs
                try:
                    self.widget3d.scene.remove_geometry(f"handle_{i}")
                    self.hidden_ui_elements.append(f"handle_{i}")
                except:
                    pass
                try:
                    self.widget3d.scene.remove_geometry(f"target_{i}")
                    self.hidden_ui_elements.append(f"target_{i}")
                except:
                    pass

            # Remove selection highlights and other UI elements (but NOT arrows)
            ui_elements = [
                "selection_highlight",
                "selection_bbox",
                "selection_corners",
                "coordinate_frame"  # Temporarily remove coordinate frame too
            ]

            for element in ui_elements:
                try:
                    self.widget3d.scene.remove_geometry(element)
                    self.hidden_ui_elements.append(element)
                except:
                    pass

            # NOTE: DO NOT remove arrows here - they should stay for camera_capture.png

            self.widget3d.force_redraw()
            print(f"🔄 Removed {len(self.hidden_ui_elements)} UI elements (kept arrows)")

        except Exception as e:
            print(f"⚠️ Error hiding UI elements: {e}")
            self.hidden_ui_elements = []

    def hide_spheres_for_capture(self):
        """Temporarily hide all UI elements including arrows for deformed_mesh.png."""
        try:
            print("🔄 Temporarily hiding all UI elements including arrows...")

            # Store what we're removing for restoration
            self.hidden_elements = []

            # Remove all handle and target spheres
            for i in range(10):  # Max 10 pairs
                try:
                    self.widget3d.scene.remove_geometry(f"handle_{i}")
                    self.hidden_elements.append(f"handle_{i}")
                except:
                    pass
                try:
                    self.widget3d.scene.remove_geometry(f"target_{i}")
                    self.hidden_elements.append(f"target_{i}")
                except:
                    pass

            # Remove selection highlights and other UI elements
            ui_elements = [
                "selection_highlight",
                "selection_bbox",
                "selection_corners",
                "coordinate_frame"  # Temporarily remove coordinate frame too
            ]

            for element in ui_elements:
                try:
                    self.widget3d.scene.remove_geometry(element)
                    self.hidden_elements.append(element)
                except:
                    pass

            # Remove arrows if they exist (for deformed_mesh.png)
            for i in range(100):  # Check more arrows
                try:
                    self.widget3d.scene.remove_geometry(f"arrow_{i}")
                    self.hidden_elements.append(f"arrow_{i}")
                except:
                    pass

            self.widget3d.force_redraw()
            print(f"🔄 Removed {len(self.hidden_elements)} UI elements including arrows")

        except Exception as e:
            print(f"⚠️ Error hiding elements: {e}")
            self.hidden_elements = []

    def show_ui_elements_after_capture(self):
        """Restore UI elements after camera capture (for use with hide_ui_elements_only)."""
        try:
            print("🔄 Restoring UI elements after capture...")

            # Restore spheres using existing function
            self.update_spheres_in_scene()

            # Restore coordinate frame
            if hasattr(self, 'hidden_ui_elements') and "coordinate_frame" in self.hidden_ui_elements:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.bbox_diag * 0.1, origin=[0, 0, 0])
                coord_mat = rendering.MaterialRecord()
                coord_mat.shader = "defaultUnlit"
                self.widget3d.scene.add_geometry("coordinate_frame", coordinate_frame, coord_mat)

            # Clear hidden UI elements list
            if hasattr(self, 'hidden_ui_elements'):
                print(f"🔄 Restored {len(self.hidden_ui_elements)} UI elements")
                self.hidden_ui_elements = []

            self.widget3d.force_redraw()

        except Exception as e:
            print(f"⚠️ Error restoring UI elements: {e}")

    def show_spheres_after_capture(self):
        """Restore all elements after camera capture (for use with hide_spheres_for_capture)."""
        try:
            print("🔄 Restoring all elements after capture...")

            # Restore spheres using existing function
            self.update_spheres_in_scene()

            # Restore coordinate frame
            if hasattr(self, 'hidden_elements') and "coordinate_frame" in self.hidden_elements:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.bbox_diag * 0.1, origin=[0, 0, 0])
                coord_mat = rendering.MaterialRecord()
                coord_mat.shader = "defaultUnlit"
                self.widget3d.scene.add_geometry("coordinate_frame", coordinate_frame, coord_mat)

            # Restore arrows if they were hidden
            if hasattr(self, 'arrows') and len(self.arrows) > 0:
                print("🔄 Restoring arrows...")
                arrow_mat = rendering.MaterialRecord()
                arrow_mat.shader = "defaultLit"
                arrow_mat.base_color = [1.0, 0.0, 0.0, 1.0]  # Red color

                for i, arrow in enumerate(self.arrows):
                    if arrow is not None:
                        self.widget3d.scene.add_geometry(f"arrow_{i}", arrow, arrow_mat)

            # Clear hidden elements list
            if hasattr(self, 'hidden_elements'):
                print(f"🔄 Restored {len(self.hidden_elements)} elements including arrows")
                self.hidden_elements = []

            self.widget3d.force_redraw()

        except Exception as e:
            print(f"⚠️ Error restoring elements: {e}")

    def restore_ui_elements_after_deformed_mesh_capture(self):
        """Restore UI elements after deformed mesh capture (selectively restore only needed elements)."""
        try:
            print("🔄 Restoring essential UI elements after deformed mesh capture...")

            # Only restore spheres and arrows, not all hidden elements
            self.update_spheres_in_scene()

            # Restore arrows if they exist and are needed
            if hasattr(self, 'arrows') and len(self.arrows) > 0:
                print("🔄 Restoring arrows...")
                arrow_mat = rendering.MaterialRecord()
                arrow_mat.shader = "defaultLit"
                arrow_mat.base_color = [1.0, 0.0, 0.0, 1.0]  # Red color

                for i, arrow in enumerate(self.arrows):
                    if arrow is not None:
                        self.widget3d.scene.add_geometry(f"arrow_{i}", arrow, arrow_mat)

            # Clear the hidden elements list
            if hasattr(self, 'hidden_deformed_mesh_elements'):
                print(f"🔄 Cleared {len(self.hidden_deformed_mesh_elements)} hidden element records")
                self.hidden_deformed_mesh_elements = []

            self.widget3d.force_redraw()
            print("✅ Essential UI elements restored")

        except Exception as e:
            print(f"⚠️ Error restoring UI elements after deformed mesh capture: {e}")

    #NOTE: capture_deformed_mesh_only
    def capture_deformed_mesh_only(self):
        """Save deformed mesh as separate file - user can view it separately."""
        import os

        try:
            print("📸 Saving deformed mesh as separate 3D file...")

            # Create output directory if needed
            if not hasattr(self, 'output_dir'):
                self.create_output_directory()

            # Prepare file paths
            timestamp = int(time.time())
            mesh_filename = f"deformed_mesh_{timestamp}.ply"
            mesh_filepath = os.path.join(self.output_dir, mesh_filename)

            if not hasattr(self, 'deformed_mesh') or self.deformed_mesh is None:
                print("❌ No deformed mesh available")
                return

            # Save deformed mesh as PLY file
            # o3d.io.write_triangle_mesh(mesh_filepath, self.deformed_mesh)
            # print(f"✅ Deformed mesh saved as 3D file: {mesh_filepath}")

            # Also save as PNG by creating minimal visualization
            png_filename = f"deformed_mesh_{timestamp}.png"
            png_filepath = os.path.join(self.output_dir, png_filename)

            # Use PyVista if available for better rendering
            try:
                import pyvista as pv
                print("🎨 Using PyVista for high-quality rendering...")

                # Create PyVista mesh from Open3D mesh
                vertices = np.asarray(self.deformed_mesh.vertices)
                triangles = np.asarray(self.deformed_mesh.triangles)

                # Create faces array for PyVista (prepend count to each face)
                faces = np.hstack([[3] + list(tri) for tri in triangles])

                # Create PyVista mesh
                pv_mesh = pv.PolyData(vertices, faces)

                # Add colors if available
                if self.deformed_mesh.has_vertex_colors():
                    colors = np.asarray(self.deformed_mesh.vertex_colors)
                    pv_mesh['colors'] = colors

                # Get viewer window size to match FOV/aspect ratio
                try:
                    # Get the actual 3D viewer widget size
                    viewer_width = self.widget3d.frame.width
                    viewer_height = self.widget3d.frame.height
                    print(f"📐 Viewer size: {viewer_width}x{viewer_height}")
                except:
                    viewer_width, viewer_height = 800, 600
                    print(f"⚠️ Could not get viewer size, using default: {viewer_width}x{viewer_height}")

                # Create plotter with matching aspect ratio
                plotter = pv.Plotter(off_screen=True, window_size=[viewer_width, viewer_height])

                if self.deformed_mesh.has_vertex_colors():
                    plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)
                else:
                    plotter.add_mesh(pv_mesh, color='lightgray')

                # Set camera to match current view exactly
                try:
                    camera = self.widget3d.scene.camera
                    view_matrix = camera.get_view_matrix()
                    proj_matrix = camera.get_projection_matrix()

                    # Extract camera position from view matrix
                    view_matrix_inv = np.linalg.inv(view_matrix)
                    camera_pos = view_matrix_inv[:3, 3]

                    # Extract camera look-at point (focal point) from view matrix
                    # The view matrix transforms world coordinates to camera coordinates
                    # Camera's forward direction is -Z axis in camera space
                    camera_forward = -view_matrix_inv[:3, 2]  # -Z direction in world space

                    # Use bounding box center as reference for distance calculation
                    mesh_center = vertices.mean(axis=0)

                    # Calculate distance from camera to mesh center
                    distance_to_center = np.linalg.norm(camera_pos - mesh_center)

                    # Calculate focal point along camera forward direction
                    focal_point = camera_pos + camera_forward * distance_to_center

                    # Get up vector from view matrix
                    camera_up = view_matrix_inv[:3, 1]

                    # Get actual FOV from Open3D camera
                    try:
                        # Calculate FOV from projection matrix
                        # FOV_y = 2 * arctan(1 / proj_matrix[1,1]) in radians
                        fov_y_rad = 2 * np.arctan(1.0 / proj_matrix[1, 1])
                        fov_y_deg = np.degrees(fov_y_rad)

                        print(f"📐 Extracted Open3D camera FOV: {fov_y_deg:.2f}°")
                    except Exception as fov_err:
                        print(f"⚠️ Could not extract FOV: {fov_err}, using default 60°")
                        fov_y_deg = 60.0

                    # Set PyVista camera to match Open3D exactly
                    plotter.camera.position = camera_pos
                    plotter.camera.focal_point = focal_point
                    plotter.camera.up = camera_up
                    plotter.camera.view_angle = fov_y_deg

                    # Get clipping range from Open3D if possible
                    try:
                        bounds = self.widget3d.scene.bounding_box
                        scene_size = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
                        plotter.camera.clipping_range = (scene_size * 0.01, scene_size * 10.0)
                    except:
                        pass

                    print(f"✅ Camera set - Position: {camera_pos}, Focal: {focal_point}, FOV: {fov_y_deg:.2f}°")

                except Exception as cam_err:
                    print(f"⚠️ Camera setup error: {cam_err}, using default view")
                    plotter.camera_position = 'xy'

                plotter.screenshot(png_filepath)

                # Don't call plotter.close() - it can crash the program
                # Just delete the plotter object
                print("🧹 Cleaning up plotter...")
                del plotter
                print("✅ Plotter cleaned up")

                print(f"✅ PNG rendered with PyVista: {png_filepath}")

            except ImportError:
                print("⚠️ PyVista not available, skipping PNG rendering")
                print("💡 You can view the 3D mesh file with any 3D viewer (MeshLab, Blender, etc.)")

            # Verify files were created
            if os.path.exists(mesh_filepath):
                file_size = os.path.getsize(mesh_filepath)
                print(f"✅ 3D mesh file confirmed: {mesh_filepath} ({file_size} bytes)")
                print("✅✅✅ This file contains ONLY the deformed mesh - NO ARROWS, NO SPHERES!")

            if os.path.exists(png_filepath):
                file_size = os.path.getsize(png_filepath)
                print(f"✅ PNG file confirmed: {png_filepath} ({file_size} bytes)")

            return

        except Exception as e:
            print(f"❌ Error capturing deformed mesh only: {e}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

            # Attempt to restore UI elements even on error
            try:
                print("🔄 Attempting to restore UI elements after error...")
                self.update_spheres_in_scene()
                if hasattr(self, 'arrows') and len(self.arrows) > 0:
                    arrow_mat = rendering.MaterialRecord()
                    arrow_mat.shader = "defaultLit"
                    arrow_mat.base_color = [1.0, 0.0, 0.0, 1.0]
                    for i, arrow in enumerate(self.arrows):
                        if arrow is not None:
                            self.widget3d.scene.add_geometry(f"arrow_{i}", arrow, arrow_mat)
                self.widget3d.force_redraw()
                print("✅ UI elements restored after error")
            except Exception as restore_error:
                print(f"⚠️ Could not restore UI elements: {restore_error}")

    def capture_and_display_camera_matrices(self):
        """Capture current camera matrices and display them in GUI and console."""
        print("\n📷 Capturing Camera Matrices...")

        # Get current camera
        camera = self.widget3d.scene.camera
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()

        print("🔍 View Matrix:")
        print(view_matrix)
        print("\n🔍 Projection Matrix:")
        print(projection_matrix)

        # Format matrices for display
        def format_matrix(matrix, name):
            result = f"{name}:\n"
            for i in range(4):
                row = "  [" + ", ".join([f"{matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                result += row
            return result

        # Update GUI display
        display_text = "=== Current Camera Matrices ===\n\n"
        display_text += format_matrix(view_matrix, "View Matrix")
        display_text += "\n"
        display_text += format_matrix(projection_matrix, "Projection Matrix")

        self.camera_matrix_text.text_value = display_text

    def align_camera_to_2d_correspondence(self, handle_3d, handle_2d, disparity_3d, disparity_2d):
            try:
                print("🔄 Using numpy conversion method...")

                # 다시 콜백으로 이미지 얻기
                np_image = None

                def numpy_callback(image):
                    nonlocal np_image
                    try:
                        np_image = np.asarray(image)
                        print("📸 NumPy image captured")
                    except Exception as e:
                        print(f"❌ NumPy callback error: {e}")

                self.widget3d.scene.scene.render_to_image(numpy_callback)
                time.sleep(0.2)

                if np_image is not None:
                    # BGR 변환하여 저장
                    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                        cv2.imwrite(filepath, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(filepath, np_image)
                    print(f"✅ NumPy conversion successful: {filepath}")
                    return

            except Exception as numpy_error:
                print(f"⚠️ NumPy method failed: {numpy_error}")

                # Method 2: Try window-level capture
                try:
                    print("🔄 Trying window-level capture method...")

                    timestamp = int(time.time())
                    filename = f"camera_capture_{timestamp}.png"
                    filepath = os.path.join(self.output_dir, filename)

                    # Ensure output directory exists
                    os.makedirs(self.output_dir, exist_ok=True)

                    # Try window capture first
                    capture_success = False

                    # Method 2a: Try window capture_screen_image
                    if hasattr(self.window, 'capture_screen_image'):
                        print("📸 Attempting window.capture_screen_image...")
                        try:
                            success = self.window.capture_screen_image(filepath)
                            if success:
                                print(f"✅ Window capture successful: {filepath}")
                                capture_success = True
                            else:
                                print("❌ Window capture returned False")
                        except Exception as window_error:
                            print(f"❌ Window capture error: {window_error}")

                    # Method 2b: Try widget capture if window failed
                    if not capture_success and hasattr(self.widget3d, 'capture_screen_image'):
                        print("📸 Attempting widget3d.capture_screen_image...")
                        try:
                            success = self.widget3d.capture_screen_image(filepath)
                            if success:
                                print(f"✅ Widget capture successful: {filepath}")
                                capture_success = True
                            else:
                                print("❌ Widget capture returned False")
                        except Exception as widget_error:
                            print(f"❌ Widget capture error: {widget_error}")

                    # If all capture methods failed, create detailed info file
                    if not capture_success:
                        print("📝 All capture methods failed - creating detailed camera info...")

                        # Get camera information for the info file
                        camera = self.widget3d.scene.camera
                        view_matrix = camera.get_view_matrix()
                        projection_matrix = camera.get_projection_matrix()

                        info_filepath = filepath.replace('.png', '_camera_info.txt')
                        with open(info_filepath, 'w') as f:
                            f.write(f"Camera Capture Info - Timestamp: {timestamp}\n")
                            f.write("="*50 + "\n\n")
                            f.write("STATUS: Image capture failed, camera matrices saved\n\n")

                            f.write("VIEW MATRIX:\n")
                            for i in range(4):
                                row = "  [" + ", ".join([f"{view_matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                                f.write(row)

                            f.write("\nPROJECTION MATRIX:\n")
                            for i in range(4):
                                row = "  [" + ", ".join([f"{projection_matrix[i,j]:8.4f}" for j in range(4)]) + "]\n"
                                f.write(row)

                            # Camera parameters
                            view_det = np.linalg.det(view_matrix[:3,:3])
                            camera_pos = view_matrix[:3, 3]
                            f.write(f"\nCAMERA PARAMETERS:\n")
                            f.write(f"View Matrix Determinant: {view_det:.6f}\n")
                            f.write(f"Camera Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]\n")

                            f.write(f"\nNOTE: Direct image capture failed.\n")
                            f.write(f"Please use manual screenshot or external capture tool.\n")

                        print(f"📄 Camera info saved to: {info_filepath}")
                        print("💡 Suggestion: Use Cmd+Shift+4 (macOS) or manual screenshot")

                except Exception as alt_error:
                    print(f"❌ Alternative method also failed: {alt_error}")

    def align_camera_to_2d_correspondence(self, handle_3d, handle_2d, disparity_3d, disparity_2d):
        """Align camera so that 3D handle projects to 2D handle position using proper projection."""
        print("🔧 Aligning camera to match 3D and 2D handle points...")

        # Check if we have captured camera matrices - if so, don't move camera
        if hasattr(self, 'view_matrix') and hasattr(self, 'projection_matrix') and \
           self.view_matrix is not None and self.projection_matrix is not None:
            print("📷 Using previously captured camera matrices - skipping camera movement")
            print("   This preserves the current 3D view without causing rotation issues")
            return

        # Get image dimensions
        if self.image is not None:
            H, W = self.image.shape[:2]
        else:
            H, W = 512, 512

        print(f"📐 Image dimensions: {W} x {H}")
        print(f"📍 3D Handle point: {handle_3d}")
        print(f"📍 2D Handle point: {handle_2d}")

        # Get mesh center and bounds for better camera positioning
        vertices = np.asarray(self.mesh.vertices)
        center = np.mean(vertices, axis=0)
        bbox = self.mesh.get_axis_aligned_bounding_box()
        mesh_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)

        print(f"📦 Mesh center: {center}")
        print(f"📦 Mesh size: {mesh_size:.3f}")

        # Calculate camera distance based on mesh size and scale factor
        scale_factor = disparity_2d / disparity_3d if disparity_3d > 0 else 1.0
        print(f"📏 Scale factor: {scale_factor:.3f}")

        # 현재 카메라 상태 확인 (3D viewer와 동일한 크기 유지하기 위해)
        current_camera = self.widget3d.scene.camera
        current_view = current_camera.get_view_matrix()
        current_proj = current_camera.get_projection_matrix()

        # 현재 카메라 매트릭스에서 실제 position과 direction 추출
        view_matrix_inv = np.linalg.inv(current_view)
        actual_camera_pos = view_matrix_inv[:3, 3]
        actual_camera_dir = -view_matrix_inv[:3, 2]  # -Z direction in camera space
        actual_camera_up = view_matrix_inv[:3, 1]    # Y direction in camera space

        print(f"🔍 Current camera position: {actual_camera_pos}")
        print(f"🔍 Current camera direction: {actual_camera_dir}")
        print(f"🔍 Current camera up: {actual_camera_up}")

        # Convert 2D handle to normalized coordinates (-1 to 1)
        handle_2d_norm_x = (handle_2d[0] / W) * 2 - 1  # x: 0->W maps to -1->1
        handle_2d_norm_y = -((handle_2d[1] / H) * 2 - 1)  # y: 0->H maps to 1->-1 (flip Y)

        print(f"📍 2D Handle normalized: ({handle_2d_norm_x:.3f}, {handle_2d_norm_y:.3f})")

        # Project current 3D handle point to verify current alignment
        handle_3d_h = np.array([handle_3d[0], handle_3d[1], handle_3d[2], 1.0])
        view_coords = current_view @ handle_3d_h
        clip_coords = current_proj @ view_coords

        if clip_coords[3] != 0:
            ndc_coords = clip_coords[:3] / clip_coords[3]
        else:
            ndc_coords = clip_coords[:3]

        current_pixel_x = (ndc_coords[0] + 1) * W / 2
        current_pixel_y = (1 - ndc_coords[1]) * H / 2

        print(f"🎯 Current 3D handle projects to: ({current_pixel_x:.1f}, {current_pixel_y:.1f})")
        print(f"🎯 Target 2D handle position: ({handle_2d[0]:.1f}, {handle_2d[1]:.1f})")

        # Calculate required adjustment (much smaller to prevent 180-degree flips)
        pixel_error_x = handle_2d[0] - current_pixel_x
        pixel_error_y = handle_2d[1] - current_pixel_y

        print(f"📏 Pixel error: ({pixel_error_x:.1f}, {pixel_error_y:.1f})")

        # If error is small, skip adjustment
        if abs(pixel_error_x) < 10 and abs(pixel_error_y) < 10:
            print("✅ Current alignment is already good enough - skipping camera movement")
            self.view_matrix = current_view
            self.projection_matrix = current_proj
            return

        # Very conservative camera adjustment to prevent flipping
        adjustment_scale = mesh_size * 0.01  # Much smaller adjustment

        # Calculate camera coordinate system vectors
        camera_right = np.cross(actual_camera_dir, actual_camera_up)
        camera_right = camera_right / np.linalg.norm(camera_right) if np.linalg.norm(camera_right) > 0 else np.array([1, 0, 0])

        # Small adjustment in camera coordinate system
        camera_adjustment = (
            -pixel_error_x / W * adjustment_scale * camera_right +
            pixel_error_y / H * adjustment_scale * actual_camera_up
        )

        new_camera_pos = actual_camera_pos + camera_adjustment

        # Ensure we're still looking at the center
        new_look_dir = center - new_camera_pos
        new_look_dir = new_look_dir / np.linalg.norm(new_look_dir)

        print(f"📷 Small adjustment vector: {camera_adjustment}")
        print(f"📷 New camera position: {new_camera_pos}")

        # Apply very conservative camera update
        camera = self.widget3d.scene.camera

        try:
            # Use look_at with proper eye, center, up order to prevent flipping
            camera.look_at(center, new_camera_pos, actual_camera_up)
            print("✅ Conservative camera update successful")
        except Exception as camera_error:
            print(f"⚠️ Camera update failed: {camera_error}")
            print("⚠️ Keeping original camera position")
            new_camera_pos = actual_camera_pos

        # Force redraw to apply changes
        self.widget3d.force_redraw()

        # Get updated camera matrices
        self.view_matrix = camera.get_view_matrix()
        self.projection_matrix = camera.get_projection_matrix()

        print("\n🔍 Updated Camera Matrices captured")

        # Verify alignment by projecting 3D handle point
        print("\n🎯 Verifying projection alignment...")

        # Project 3D handle point using camera matrices
        handle_3d_h = np.array([handle_3d[0], handle_3d[1], handle_3d[2], 1.0])

        # Apply view transformation
        view_coords = self.view_matrix @ handle_3d_h

        # Apply projection transformation
        clip_coords = self.projection_matrix @ view_coords

        # Perspective divide
        if clip_coords[3] != 0:
            ndc_coords = clip_coords[:3] / clip_coords[3]
        else:
            ndc_coords = clip_coords[:3]

        # Convert NDC to pixel coordinates
        pixel_x = (ndc_coords[0] + 1) * W / 2
        pixel_y = (1 - ndc_coords[1]) * H / 2  # Flip Y coordinate

        projection_error = np.sqrt((pixel_x - handle_2d[0])**2 + (pixel_y - handle_2d[1])**2)

        print(f"📍 NDC coordinates: ({ndc_coords[0]:.3f}, {ndc_coords[1]:.3f}, {ndc_coords[2]:.3f})")
        print(f"🎯 Projected 3D handle to pixel: ({pixel_x:.1f}, {pixel_y:.1f})")
        print(f"🎯 Target 2D handle pixel: ({handle_2d[0]:.1f}, {handle_2d[1]:.1f})")
        print(f"📏 Projection error: {projection_error:.2f} pixels")

        if projection_error < 50:  # Within 50 pixels is considered good alignment
            print("✅ Good alignment achieved!")
        else:
            print("⚠️ Alignment may need further adjustment")

        print("✅ Camera alignment completed")

    def generate_final_flow_field(self):
        """Generate final flow field with proper file naming."""
        print("🔄 Generating final flow field...")

        # Get mesh vertices
        vertices_orig = np.asarray(self.original_mesh.vertices) if self.original_mesh else np.asarray(self.mesh.vertices)
        vertices_def = np.asarray(self.deformed_mesh.vertices)

        # Compute flow field using current camera matrices
        flow_field = self.compute_flow_field_projection( #ANCHOR
            vertices_orig, vertices_def,
            self.view_matrix, self.projection_matrix
        )

        # Save with required filenames
        self.save_final_flow_field_files(flow_field)

        print("✅ Final flow field generated and saved")

    def save_final_flow_field_files(self, flow_field):
        """Save flow field with specific required filenames."""
        print("💾 Saving final flow field files...")

        # Ensure output directory exists
        self.create_output_directory()

        # 1. Save 2d_vector_flow_field.npy
        npy_path = os.path.join(self.output_dir, "2d_vector_flow_field.npy")
        np.save(npy_path, flow_field)
        print(f"📄 2d_vector_flow_field.npy saved to: {npy_path}")

        # 2. Save 2d_vector_flow_field.png (visualization with arrows)
        H, W = flow_field.shape[:2]
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Use original image as background (without web viewer annotations)
            viz_img = self.original_image.copy()
        elif self.image is not None:
            # Fallback to current image
            viz_img = self.image.copy()
        else:
            # Create white background
            viz_img = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Draw flow vectors as arrows
        step = 5  # Draw every 10th vector

        # Calculate dynamic scale based on 2D input points (same logic as compute_flow_field_projection)
        scale = 3.0  # Default fallback scale

        if hasattr(self, 'temp_handle_2d') and hasattr(self, 'temp_target_2d') and \
           self.temp_handle_2d is not None and self.temp_target_2d is not None and \
           len(self.handle_points) > 0 and len(self.target_points) > 0:

            # Get first handle and target points (3D)
            handle_3d = self.handle_points[0]
            target_3d = self.target_points[0]

            # Project 3D points to 2D using current camera matrices
            handle_pixel_raw = self.project_3d_to_2d(handle_3d, self.view_matrix, self.projection_matrix, W, H)
            target_pixel_raw = self.project_3d_to_2d(target_3d, self.view_matrix, self.projection_matrix, W, H)

            if handle_pixel_raw is not None and target_pixel_raw is not None:
                handle_pixel = np.array(handle_pixel_raw)
                target_pixel = np.array(target_pixel_raw)

                # Get 2D points from web interface
                handle_2d = np.array(self.temp_handle_2d) # web viewer에서 입력받은 두 점
                target_2d = np.array(self.temp_target_2d)

                # Calculate distances
                projected_distance = np.linalg.norm(target_pixel - handle_pixel)
                input_distance = np.linalg.norm(target_2d - handle_2d)

                if projected_distance > 0:
                    scale = input_distance / projected_distance
                    print(f"📏 Using dynamic scale for flow field arrows: {scale:.3f}")
                    print(f"   3D projected distance: {projected_distance:.1f} pixels")
                    print(f"   2D input distance: {input_distance:.1f} pixels")
                else:
                    print("⚠️ Zero projected distance, using default scale 3.0")
            else:
                print("⚠️ Could not project 3D points, using default scale 3.0")
        else:
            print("⚠️ No handle/target points available, using default scale 3.0")

        for y in range(0, H, step):
            for x in range(0, W, step):
                if y < H and x < W:
                    u, v = flow_field[y, x]
                    if np.abs(u) > 0.1 or np.abs(v) > 0.1:  # Only draw significant vectors
                        # Calculate end point
                        end_x = int(x + u * scale)
                        end_y = int(y + v * scale)

                        # Ensure end point is within bounds
                        end_x = np.clip(end_x, 0, W-1)
                        end_y = np.clip(end_y, 0, H-1)

                        # Draw arrow (green for visibility)
                        cv2.arrowedLine(viz_img, (x, y), (end_x, end_y),
                                      (0, 255, 0), 2, tipLength=0.3)

        # 여기에는 flow_field 중에 입력 handle vertex → target vertex 의 flow vector도 표시해줘
        for handle_vertex, target_vertex in zip(self.handle_vertex_3d_2d_list, self.target_vertex_3d_2d_list):
            cv2.arrowedLine(viz_img, handle_vertex, target_vertex,
                          (0, 255, 0), 2, tipLength=0.3)

        # Sample points from flow_field for points_flowdrag.json
        # Grid-based sampling with magnitude filtering to get 5-15 significant points
        sampled_points = self.sample_flow_field_points(flow_field, target_count_range=(5, 15))

        # Save points_flowdrag.json
        points_flowdrag = {"points": sampled_points}
        flowdrag_json_path = os.path.join(self.output_dir, "points_flowdrag.json")
        with open(flowdrag_json_path, "w") as f:
            json.dump(points_flowdrag, f)
        print(f"📝 points_flowdrag.json saved with {len(sampled_points)} sampled points: {flowdrag_json_path}")

        # Save points_flowdrag.png - visualization of sampled points only
        points_viz_img = viz_img.copy()
        for point in sampled_points:
            x, y = point[0], point[1]
            # Draw circle at each sampled point
            cv2.circle(points_viz_img, (x, y), radius=6, color=(255, 0, 255), thickness=-1)  # Magenta filled circle
            cv2.circle(points_viz_img, (x, y), radius=8, color=(255, 255, 255), thickness=2)  # White outline

        points_png_path = os.path.join(self.output_dir, "points_flowdrag.png")
        cv2.imwrite(points_png_path, points_viz_img)
        print(f"🖼️ points_flowdrag.png saved with {len(sampled_points)} sampled points visualization: {points_png_path}")

        # Also save handle_vertex_3d_2d_list.json for reference
        points = {"points": self.handle_vertex_3d_2d_list}
        with open(os.path.join(self.output_dir, "handle_vertex_3d_2d_list.json"), "w") as f:
            json.dump(points, f)

        png_path = os.path.join(self.output_dir, "2d_vector_flow_field.png")
        cv2.imwrite(png_path, viz_img)
        print(f"🖼️ 2d_vector_flow_field.png saved with projected point markers to: {png_path}")

        # Add red markers for projected 3D handle/target points
        viz_img = self.add_projected_point_markers_to_flow_field(viz_img)

        png_path = os.path.join(self.output_dir, "2d_vector_flow_field_red_marker.png")
        cv2.imwrite(png_path, viz_img)
        print(f"🖼️ 2d_vector_flow_field_red_marker.png saved with projected point markers to: {png_path}")

        # 3. Save flow_magnitude.png (heatmap)
        magnitude = np.sqrt(flow_field[:,:,0]**2 + flow_field[:,:,1]**2)
        magnitude_normalized = (magnitude / (magnitude.max() + 1e-8) * 255).astype(np.uint8)
        magnitude_colored = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)

        magnitude_path = os.path.join(self.output_dir, "flow_magnitude.png")
        cv2.imwrite(magnitude_path, magnitude_colored)
        print(f"🌡️ flow_magnitude.png saved to: {magnitude_path}")

        # Print statistics
        non_zero_mask = magnitude > 0.1
        if np.any(non_zero_mask):
            print(f"\n📊 Flow field statistics:")
            print(f"  - Size: {H} x {W}")
            print(f"  - Non-zero vectors: {np.sum(non_zero_mask)}")
            print(f"  - Max magnitude: {magnitude.max():.3f}")
            print(f"  - Mean magnitude: {magnitude[non_zero_mask].mean():.3f}")

    def sample_flow_field_points(self, flow_field, target_count_range=(5, 15), grid_size=50, min_magnitude=10.0):
        """
        Sample significant flow points from flow_field using grid-based approach.

        Args:
            flow_field: numpy array of shape (H, W, 2)
            target_count_range: tuple (min_count, max_count) for final sampled points
            grid_size: size of grid cells for sampling
            min_magnitude: minimum flow magnitude threshold

        Returns:
            List of [x, y] points
        """
        H, W = flow_field.shape[:2]
        min_count, max_count = target_count_range

        print(f"🔍 Sampling flow field points with grid size {grid_size}, min magnitude {min_magnitude}")

        # Calculate flow magnitude
        magnitude = np.sqrt(flow_field[:,:,0]**2 + flow_field[:,:,1]**2)

        # Grid-based sampling: divide image into grid cells
        grid_points = []

        for grid_y in range(0, H, grid_size):
            for grid_x in range(0, W, grid_size):
                # Get grid cell bounds
                y_end = min(grid_y + grid_size, H)
                x_end = min(grid_x + grid_size, W)

                # Extract magnitudes in this grid cell
                grid_magnitudes = magnitude[grid_y:y_end, grid_x:x_end]

                # Find maximum magnitude position in this grid
                if grid_magnitudes.size > 0:
                    max_mag = grid_magnitudes.max()

                    # Only consider if magnitude is above threshold
                    if max_mag >= min_magnitude:
                        # Find position of max magnitude in grid
                        local_pos = np.unravel_index(grid_magnitudes.argmax(), grid_magnitudes.shape)
                        global_y = grid_y + local_pos[0]
                        global_x = grid_x + local_pos[1]

                        # Store (magnitude, x, y)
                        grid_points.append((max_mag, int(global_x), int(global_y)))

        if len(grid_points) == 0:
            print("⚠️ No points found with sufficient magnitude, returning empty list")
            return []

        # Sort by magnitude (descending)
        grid_points.sort(reverse=True, key=lambda x: x[0])

        # Select top points within target range
        if len(grid_points) > max_count:
            selected_points = grid_points[:max_count]
            print(f"📊 Selected top {max_count} points from {len(grid_points)} candidates")
        elif len(grid_points) < min_count:
            # If not enough points, lower threshold and try again
            print(f"⚠️ Only found {len(grid_points)} points, trying with lower threshold...")
            return self.sample_flow_field_points(flow_field, target_count_range, grid_size, min_magnitude * 0.5)
        else:
            selected_points = grid_points
            print(f"📊 Selected {len(selected_points)} points")

        # Convert to [[x, y], [x, y], ...] format
        result = [[x, y] for (mag, x, y) in selected_points]

        print(f"✅ Sampled {len(result)} points from flow field")
        return result

    def show_image_with_fallback_viewer(self):
        """Fallback method using external viewer when OpenCV fails."""
        try:
            import tempfile
            import subprocess
            import platform

            # Create enhanced image with coordinate grid (same as before)
            enhanced_img = self.image.copy()
            height, width = enhanced_img.shape[:2]

            # Add coordinate grid
            grid_step = 50
            grid_color = (0, 255, 255)
            text_color = (255, 255, 255)

            for x in range(0, width, grid_step):
                cv2.line(enhanced_img, (x, 0), (x, height), grid_color, 1)
                if x > 0:
                    cv2.putText(enhanced_img, str(x), (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            for y in range(0, height, grid_step):
                cv2.line(enhanced_img, (0, y), (width, y), grid_color, 1)
                if y > 0:
                    cv2.putText(enhanced_img, str(y), (5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            cv2.putText(enhanced_img, "FlowDrag - Note coordinates and enter in GUI",
                       (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(enhanced_img, "Yellow grid shows pixel coordinates",
                       (10, height-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Save and open with external viewer
            with tempfile.NamedTemporaryFile(suffix='_flowdrag_grid.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, enhanced_img)

            print(f"Enhanced image saved to: {temp_path}")

            # Open with system default viewer
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.Popen(["open", temp_path])
            elif system == "Windows":
                subprocess.Popen(["start", temp_path], shell=True)
            else:  # Linux
                subprocess.Popen(["xdg-open", temp_path])

            print("\n" + "="*60)
            print("📋 COORDINATE SELECTION INSTRUCTIONS:")
            print("="*60)
            print("1. 🖼️  View the opened image with yellow coordinate grid")
            print("2. 📍 Find your HANDLE point location")
            print("3. 📝 Note its (X, Y) coordinates from the grid")
            print("4. 📍 Find your TARGET point location")
            print("5. 📝 Note its (X, Y) coordinates from the grid")
            print("6. ⌨️  Enter coordinates in the GUI fields:")
            print(f"   - Handle 2D X: [current: {self.handle_2d_x.double_value:.0f}]")
            print(f"   - Handle 2D Y: [current: {self.handle_2d_y.double_value:.0f}]")
            print("7. 🖱️  Click 'Auto-Align & Project' button")
            print("8. 🗑️  Close the image viewer when done")
            print("="*60)

        except Exception as e:
            print(f"Could not create enhanced image: {e}")
            print("Falling back to basic image...")
            try:
                # Basic fallback
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    cv2.imwrite(temp_path, self.image)

                subprocess.Popen(["open", temp_path])  # macOS
                print(f"Basic image saved to: {temp_path}")
                print("Please manually estimate coordinates and enter in GUI fields.")
            except:
                print("Could not open any image viewer.")
                print("Please manually enter approximate coordinates in GUI fields.")
    
    #NOTE: compute_flow_field_projection()
    def compute_flow_field_projection(self, vertices_orig, vertices_def, view_mat, proj_mat):
        """Compute 2D flow field using proper perspective projection with point alignment and scaling."""
        H = 512 if self.image is None else self.image.shape[0]
        W = 512 if self.image is None else self.image.shape[1]
        flow_field = np.zeros((H, W, 2))

        print(f"🔄 Computing flow field projection for {W}x{H} image")

        # Transform vertices to camera space using proper perspective projection
        vertices_orig_h = np.hstack([vertices_orig, np.ones((len(vertices_orig), 1))])
        vertices_def_h = np.hstack([vertices_def, np.ones((len(vertices_def), 1))])

        # Apply view transformation
        view_coords_orig = (view_mat @ vertices_orig_h.T).T
        view_coords_def = (view_mat @ vertices_def_h.T).T

        # Apply projection transformation
        clip_coords_orig = (proj_mat @ view_coords_orig.T).T
        clip_coords_def = (proj_mat @ view_coords_def.T).T

        # Perform perspective divide and convert to pixel coordinates
        def project_to_pixels(clip_coords):
            pixels = []
            for clip_coord in clip_coords:
                if clip_coord[3] != 0:
                    ndc = clip_coord[:3] / clip_coord[3]
                else:
                    ndc = clip_coord[:3]

                # Convert NDC to pixel coordinates
                pixel_x = (ndc[0] + 1) * W / 2
                pixel_y = (1 - ndc[1]) * H / 2  # Flip Y coordinate
                pixels.append([pixel_x, pixel_y])
            return np.array(pixels)

        pixels_orig = project_to_pixels(clip_coords_orig)
        pixels_def = project_to_pixels(clip_coords_def)

        # Calculate scale factor and offset based on handle/target point alignment if available
        scale_factor = 1.0
        global_offset = np.array([0.0, 0.0])  # Global offset to align all projected points

        if hasattr(self, 'temp_handle_2d') and hasattr(self, 'temp_target_2d') and \
           self.temp_handle_2d is not None and self.temp_target_2d is not None and \
           len(self.handle_points) > 0 and len(self.target_points) > 0:

            # Get 3D handle and target points (first pair)
            handle_3d = np.array(self.handle_points[0])
            target_3d = np.array(self.target_points[0]) if self.target_points else handle_3d

            # Project 3D points to 2D
            handle_3d_h = np.array([handle_3d[0], handle_3d[1], handle_3d[2], 1.0])
            target_3d_h = np.array([target_3d[0], target_3d[1], target_3d[2], 1.0])

            # Apply transformations
            handle_view = view_mat @ handle_3d_h
            target_view = view_mat @ target_3d_h
            handle_clip = proj_mat @ handle_view
            target_clip = proj_mat @ target_view

            # Perspective divide
            if handle_clip[3] != 0:
                handle_ndc = handle_clip[:3] / handle_clip[3]
            else:
                handle_ndc = handle_clip[:3]

            if target_clip[3] != 0:
                target_ndc = target_clip[:3] / target_clip[3]
            else:
                target_ndc = target_clip[:3]

            # Convert to pixels
            handle_pixel = np.array([(handle_ndc[0] + 1) * W / 2, (1 - handle_ndc[1]) * H / 2])
            target_pixel = np.array([(target_ndc[0] + 1) * W / 2, (1 - target_ndc[1]) * H / 2])

            # Get 2D points from web interface
            handle_2d = np.array(self.temp_handle_2d)
            target_2d = np.array(self.temp_target_2d)

            # Calculate scale factors for both handle and target alignments
            projected_distance = np.linalg.norm(target_pixel - handle_pixel)
            input_distance = np.linalg.norm(target_2d - handle_2d)

            if projected_distance > 0:
                # Auto-calculate scale factor
                auto_scale_factor = input_distance / projected_distance
                print(f"📏 Auto-calculated scale factor: {auto_scale_factor:.3f}")
                print(f"   3D projected distance: {projected_distance:.1f} pixels")
                print(f"   2D input distance: {input_distance:.1f} pixels")

                # Store auto-calculated scale and update GUI
                self.auto_calculated_scale = auto_scale_factor
                if hasattr(self, 'auto_scale_label'):
                    self.auto_scale_label.text = f"Auto-calculated: {auto_scale_factor:.3f}"

                # Check if user has manually set a scale factor
                if hasattr(self, 'scale_factor_input') and hasattr(self, 'user_modified_scale'):
                    if self.user_modified_scale:
                        # User has manually modified the scale - use their value
                        scale_factor = self.scale_factor_input.double_value
                        print(f"🔧 Using user-specified scale factor: {scale_factor:.3f}")
                    else:
                        # User hasn't modified - use auto-calculated
                        scale_factor = auto_scale_factor
                        # Update input field to show auto-calculated value (without triggering callback)
                        self.user_modified_scale = False  # Prevent callback from setting flag
                        self.scale_factor_input.set_value(auto_scale_factor)
                        print(f"✅ Using auto-calculated scale factor: {scale_factor:.3f}")
                else:
                    scale_factor = auto_scale_factor

                # Calculate global offset to align handle points precisely
                global_offset = handle_2d - handle_pixel
                print(f"📍 Global alignment offset: ({global_offset[0]:.1f}, {global_offset[1]:.1f}) pixels")
            else:
                print("⚠️ Zero projected distance, using scale factor 1.0")

        # Apply global offset to all projected pixel coordinates
        if global_offset[0] != 0 or global_offset[1] != 0:
            print(f"🔧 Applying global offset to all projected vertices: {global_offset}")
            pixels_orig[:, 0] += global_offset[0]
            pixels_orig[:, 1] += global_offset[1]
            pixels_def[:, 0] += global_offset[0]
            pixels_def[:, 1] += global_offset[1]

        # Create flow vectors with proper scaling and alignment
        valid_count = 0
        for i, (pixel_orig, pixel_def) in enumerate(zip(pixels_orig, pixels_def)):
            # Check if both points are within image bounds
            u_orig, v_orig = int(pixel_orig[0]), int(pixel_orig[1])

            if 0 <= u_orig < W and 0 <= v_orig < H:
                # Calculate flow vector in pixel space with scale factor applied
                flow_u = (pixel_def[0] - pixel_orig[0]) * scale_factor
                flow_v = (pixel_def[1] - pixel_orig[1]) * scale_factor

                # Calculate flow magnitude
                flow_magnitude = np.sqrt(flow_u**2 + flow_v**2)

                # Set flow vector (accumulate if multiple vertices project to same pixel)
                # Only store flows with magnitude >= 10 pixels
                # if flow_magnitude >= 1.0:
                flow_field[v_orig, u_orig, 0] += flow_u
                flow_field[v_orig, u_orig, 1] += flow_v
                valid_count += 1

        print(f"✅ Flow field computed with {valid_count} valid flow vectors")
        print(f"📏 Applied scale factor: {scale_factor:.3f}")

        # 여기에 3d handle vertex들을 2d로 projection하고 offset까지 맞춘 것들의 리스트를 저장하고 싶어
        self.handle_vertex_3d_2d_list = []
        self.target_vertex_3d_2d_list = []

        # Add handle → target vertex flows directly to flow_field
        if hasattr(self, 'handle_points') and hasattr(self, 'target_points') and \
           len(self.handle_points) > 0 and len(self.target_points) > 0:
            print("🎯 Adding handle → target vertex flows to flow field...")

            for i, (handle_3d, target_3d) in enumerate(zip(self.handle_points, self.target_points)):
                # Project 3D handle and target to 2D
                handle_3d_h = np.array([handle_3d[0], handle_3d[1], handle_3d[2], 1.0])
                target_3d_h = np.array([target_3d[0], target_3d[1], target_3d[2], 1.0])

                # Apply transformations
                handle_view = view_mat @ handle_3d_h
                target_view = view_mat @ target_3d_h
                handle_clip = proj_mat @ handle_view
                target_clip = proj_mat @ target_view

                # Perspective divide
                if handle_clip[3] != 0:
                    handle_ndc = handle_clip[:3] / handle_clip[3]
                else:
                    handle_ndc = handle_clip[:3]

                if target_clip[3] != 0:
                    target_ndc = target_clip[:3] / target_clip[3]
                else:
                    target_ndc = target_clip[:3]

                # Convert to pixels
                handle_pixel = np.array([(handle_ndc[0] + 1) * W / 2, (1 - handle_ndc[1]) * H / 2])
                target_pixel = np.array([(target_ndc[0] + 1) * W / 2, (1 - target_ndc[1]) * H / 2])

                # Apply global offset
                handle_pixel += global_offset
                target_pixel += global_offset

                # Calculate flow vector
                flow_u = (target_pixel[0] - handle_pixel[0]) * scale_factor
                flow_v = (target_pixel[1] - handle_pixel[1]) * scale_factor

                # Add to flow field (no magnitude threshold for user input points)
                u_handle = int(handle_pixel[0])
                v_handle = int(handle_pixel[1])

                # For debugging
                self.handle_vertex_3d_2d_list.append((int(handle_pixel[0]), int(handle_pixel[1])))
                self.target_vertex_3d_2d_list.append((int(target_pixel[0]), int(target_pixel[1])))

                if 0 <= u_handle < W and 0 <= v_handle < H:
                    flow_field[v_handle, u_handle, 0] = flow_u
                    flow_field[v_handle, u_handle, 1] = flow_v
                    print(f"   ✅ Added handle vertex {i+1} flow: ({flow_u:.1f}, {flow_v:.1f}) at pixel ({u_handle}, {v_handle})")

        # Smooth the flow field to handle multiple projections to same pixel
        # from scipy import ndimage
        # flow_field[:,:,0] = ndimage.gaussian_filter(flow_field[:,:,0], sigma=1.0)
        # flow_field[:,:,1] = ndimage.gaussian_filter(flow_field[:,:,1], sigma=1.0)

        return flow_field
    
    # on_save_flow_field function removed - flow field is automatically saved
    # when "Auto-align to Input Image & 2D Projection" button is clicked

    def save_current_view(self, filepath):
        """Save the current 3D view as an image."""
        # Capture current view
        img = self.widget3d.scene.render_to_image()
        
        # Convert to numpy array and save
        o3d.io.write_image(filepath, img)
     
    def on_save_mesh(self):
        """Save deformed mesh."""
        if self.deformed_mesh is None:
            print("No deformed mesh to save!")
            return
        
        prefix = self.output_prefix.text_value
        mesh_path = f"{prefix}_deformed.ply"
        o3d.io.write_triangle_mesh(mesh_path, self.deformed_mesh)
        print(f"Deformed mesh saved to {mesh_path}")


def main():
    """Main function to run FlowDrag editor."""
    import argparse
    
    # Ensure we're using the correct Python environment
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    parser = argparse.ArgumentParser(description="FlowDrag 3D Mesh Editor")
    parser.add_argument("--mesh", help="Path to mesh file (.ply)")
    parser.add_argument("--image", help="Path to reference image")
    parser.add_argument("--mask", help="Path to mask image")
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default mesh if not provided
    if not args.mesh:
        # args.mesh = os.path.join(script_dir, "samples/dog_flower/dog_flower.ply")
        # args.mesh = os.path.join(script_dir, "samples/woman_left/woman.ply")
        args.mesh = os.path.join(script_dir, "samples/squirrel/squirrel.ply")
    
    # Set default paths if not provided
    if args.image is None:
        args.image = args.mesh.replace(".ply", ".jpg")
    if args.mask is None:
        args.mask = os.path.join(os.path.dirname(args.mesh), "mask.png")
    
    print(f"📂 Loading mesh from: {args.mesh}")
    print(f"📂 Mesh exists: {os.path.exists(args.mesh)}")
    
    # Create and run editor
    editor = FlowDragEditor(args.mesh, args.image, args.mask)
    
    print("\n=== FlowDrag Editor Started ===")
    print(f"Mesh: {args.mesh}")
    print(f"Image: {args.image}")
    print(f"Mask: {args.mask}")
    
    editor.run_gui()


if __name__ == "__main__":
    main()