from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
import trimesh.creation
import trimesh.ray
import trimesh.transformations
import viser
import viser.transforms as tf
from viser.theme import TitlebarConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH_PATH = Path(__file__).resolve().parent / "examples" / "bowl.ply"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visions" / "ram" / "template" / "annotated"


class ViserApp:
    def __init__(
        self,
        ply_file: str | Path,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        scale_value: float = 0.002,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.mesh_path = Path(ply_file).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        try:
            self.server = viser.ViserServer(host=host, port=port)
        except TypeError:
            self.server = viser.ViserServer(ip=host, port=port)
        self.server.gui.configure_theme(
            brand_color=(130, 0, 150),
            titlebar_content=TitlebarConfig(buttons=(), image=None),
        )
        self.server.scene.set_up_direction("+z")

        self.mesh: trimesh.Trimesh = cast(trimesh.Trimesh, trimesh.load_mesh(self.mesh_path))
        self.scale_mesh = scale_value
        self.mesh.vertices *= self.scale_mesh
        if hasattr(self.mesh.visual, "vertex_colors"):
            colors = np.array(self.mesh.visual.vertex_colors, dtype=np.float32)
            self.mesh.visual.vertex_colors = colors / 255.0
        else:
            white = np.ones((len(self.mesh.vertices), 4), dtype=np.float32)
            self.mesh.visual.vertex_colors = white

        self.mesh_handle = self.server.scene.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
        )

        self.hit_pos_handles: list[viser.SceneNodeHandle] = []
        self.gripper_dict: dict[str, viser.SceneNodeHandle] = {}
        self.extra_gui_handles: list[viser.GuiHandle] = []
        self.place_plane_objects: list[viser.SceneNodeHandle] = []
        self.support_plane_objects: list[viser.SceneNodeHandle] = []

        self.export_gripper_data: list[dict] = []
        self.export_place_plane_data: list[dict] = []
        self.export_support_plane_data: list[dict] = []
        self.export_point_data: list[dict] = []
        self.export_sliding_direction_data: list[dict] = []
        self.export_rotation_axis_data: list[dict] = []

        self.add_sliding_direction_button_handle: viser.GuiButtonHandle | None = None
        self.add_rotation_axis_button_handle: viser.GuiButtonHandle | None = None
        self.ply_file: str = str(self.mesh_path)

        self._bind_client_events()

    def _bind_client_events(self) -> None:
        """Register all GUI events when a client connects."""

        @self.server.on_client_connect
        def on_client_connect(client: viser.ClientHandle) -> None:
            self._register_add_point_button(client)
            self._register_add_sphere_button(client)
            self._register_paint_mesh_button(client)
            self._register_define_place_plane_button(client)
            self._register_define_support_plane_button(client)
            self._register_add_sliding_direction_button(client)
            self._register_add_rotation_axis_button(client)
            self._register_export_button(client)
            self._register_clear_scene_button(client)

    # --- GUI Registration Methods ---

    def _register_add_point_button(self, client: viser.ClientHandle) -> None:
        """Register the Add Point button to capture point coordinates from scene clicks."""
        add_point_button_handle = client.gui.add_button(
            "Add Point", icon=viser.Icon.PLUS
        )

        @add_point_button_handle.on_click
        def on_click(_):
            add_point_button_handle.disabled = True

            @client.scene.on_pointer_event(event_type="click")
            def on_pointer_click(event: viser.ScenePointerEvent) -> None:
                hit = self._get_hit_point(event)
                if hit is None:
                    client.scene.remove_pointer_callback()
                    return

                sphere = trimesh.creation.icosphere(radius=0.05)
                sphere.vertices += hit
                sphere.visual.vertex_colors = (0.0, 0.0, 1.0, 1.0)
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=f"/point_{len(self.export_point_data)}", mesh=sphere
                )
                self.hit_pos_handles.append(sphere_handle)

                point_info = {"position": tuple(hit / self.scale_mesh)}
                self.export_point_data.append(point_info)
                print(f"Added point at: {point_info['position']}")
                client.scene.remove_pointer_callback()

            @client.scene.on_pointer_callback_removed
            def on_pointer_removed():
                add_point_button_handle.disabled = False

    def _register_add_sphere_button(self, client: viser.ClientHandle) -> None:
        """Register the Add sphere button to create a grasp point and gripper from scene clicks."""
        click_button_handle = client.gui.add_button(
            "Add sphere", icon=viser.Icon.MOUSE
        )

        @click_button_handle.on_click
        def on_click(_):
            click_button_handle.disabled = True

            @client.scene.on_pointer_event(event_type="click")
            def on_pointer_click(event: viser.ScenePointerEvent) -> None:
                self._handle_click_event(client, event)

            @client.scene.on_pointer_callback_removed
            def on_pointer_removed():
                click_button_handle.disabled = False

    def _register_paint_mesh_button(self, client: viser.ClientHandle) -> None:
        """Register the Paint mesh button to edit mesh colors with rectangle selection."""
        paint_button_handle = client.gui.add_button(
            "Paint mesh", icon=viser.Icon.PAINT
        )

        @paint_button_handle.on_click
        def on_click(_):
            paint_button_handle.disabled = True

            @client.scene.on_pointer_event(event_type="rect-select")
            def on_rect_select(message: viser.ScenePointerEvent) -> None:
                client.scene.remove_pointer_callback()
                camera = message.client.camera
                R_world_mesh = tf.SO3(self.mesh_handle.wxyz)
                R_mesh_world = R_world_mesh.inverse()
                R_camera_world = tf.SE3.from_rotation_and_translation(
                    tf.SO3(camera.wxyz), camera.position
                ).inverse()
                vertices = cast(np.ndarray, self.mesh.vertices)
                vertices = (R_mesh_world.as_matrix() @ vertices.T).T
                vertices = (
                    R_camera_world.as_matrix()
                    @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T
                ).T[:, :3]

                fov, aspect = camera.fov, camera.aspect
                vertices_proj = vertices[:, :2] / vertices[:, 2].reshape(-1, 1)
                vertices_proj /= np.tan(fov / 2)
                vertices_proj[:, 0] /= aspect
                vertices_proj = (1 + vertices_proj) / 2

                mask = (
                    (vertices_proj > np.array(message.screen_pos[0]))
                    & (vertices_proj < np.array(message.screen_pos[1]))
                ).all(axis=1)[..., None]

                self.mesh.visual.vertex_colors = np.where(
                    mask, (0.5, 0.0, 0.7, 1.0), (0.9, 0.9, 0.9, 1.0)
                )

                self.mesh_handle = self.server.scene.add_mesh_trimesh(
                    name="/mesh",
                    mesh=self.mesh,
                )

            @client.scene.on_pointer_callback_removed
            def on_rect_callback_removed():
                paint_button_handle.disabled = False
    
    def _register_define_place_plane_button(self, client: viser.ClientHandle) -> None:
        """Register the Define Place Plane button to define a green plane from 5 clicked points."""
        place_plane_button = client.gui.add_button(
            "Define Place Plane", icon=viser.Icon.LAYOUT_GRID
        )

        @place_plane_button.on_click
        def on_click(_):
            place_plane_button.disabled = True
            plane_points: list[np.ndarray] = []

            @client.scene.on_pointer_event(event_type="click")
            def on_plane_click(event: viser.ScenePointerEvent) -> None:
                hit = self._get_hit_point(event)
                if hit is None:
                    return

                sphere = trimesh.creation.icosphere(radius=0.05)
                sphere.vertices += hit
                sphere.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=f"/place_plane_point_{len(plane_points)}",
                    mesh=sphere,
                )
                self.hit_pos_handles.append(sphere_handle)
                plane_points.append(hit)

                if len(plane_points) >= 5:
                    client.scene.remove_pointer_callback()
                    centroid, normal = self._fit_plane(plane_points)
                    q = self._quaternion_from_vectors(np.array([0, 0, 1]), normal)
                    grid_handle = self.server.scene.add_grid(
                        name=f"/place_plane_{int(time.time())}",
                        width=5.0,
                        height=5.0,
                        cell_color=(0.0, 1.0, 0.0),
                        wxyz=tuple(q.tolist()),
                        position=tuple(centroid),
                    )
                    frame_handle = self.server.scene.add_frame(
                        name=f"/place_plane_frame_{int(time.time())}",
                        axes_length=0.5,
                        axes_radius=0.025,
                        wxyz=tuple(q.tolist()),
                        position=tuple(centroid),
                    )
                    self.place_plane_objects.extend([grid_handle, frame_handle])
                    export_info = {
                        "centroid": tuple(centroid / self.scale_mesh),
                        "normal": tuple(normal),
                    }
                    self.export_place_plane_data.append(export_info)
                    place_plane_button.disabled = False

            @client.scene.on_pointer_callback_removed
            def on_plane_callback_removed():
                place_plane_button.disabled = False

    def _register_define_support_plane_button(
        self, client: viser.ClientHandle
    ) -> None:
        """Register the Define Support Plane button to define a red plane from 5 clicked points."""
        support_plane_button = client.gui.add_button(
            "Define Support Plane", icon=viser.Icon.LAYOUT_GRID
        )

        @support_plane_button.on_click
        def on_click(_):
            support_plane_button.disabled = True
            plane_points: list[np.ndarray] = []

            @client.scene.on_pointer_event(event_type="click")
            def on_plane_click(event: viser.ScenePointerEvent) -> None:
                hit = self._get_hit_point(event)
                if hit is None:
                    return

                sphere = trimesh.creation.icosphere(radius=0.05)
                sphere.vertices += hit
                sphere.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=f"/support_plane_point_{len(plane_points)}",
                    mesh=sphere,
                )
                self.hit_pos_handles.append(sphere_handle)
                plane_points.append(hit)
                if len(plane_points) >= 5:
                    client.scene.remove_pointer_callback()
                    centroid, normal = self._fit_plane(plane_points)
                    q = self._quaternion_from_vectors(np.array([0, 0, 1]), normal)
                    grid_handle = self.server.scene.add_grid(
                        name=f"/support_plane_{int(time.time())}",
                        width=2.0,
                        height=2.0,
                        cell_color=(1.0, 0.0, 0.0),
                        wxyz=tuple(q.tolist()),
                        position=tuple(centroid),
                    )
                    frame_handle = self.server.scene.add_frame(
                        name=f"/support_plane_frame_{int(time.time())}",
                        axes_length=0.5,
                        axes_radius=0.025,
                        wxyz=tuple(q.tolist()),
                        position=tuple(centroid),
                    )
                    self.support_plane_objects.extend([grid_handle, frame_handle])
                    export_info = {
                        "centroid": tuple(centroid / self.scale_mesh),
                        "normal": tuple(normal),
                    }
                    self.export_support_plane_data.append(export_info)
                    support_plane_button.disabled = False

            @client.scene.on_pointer_callback_removed
            def on_plane_callback_removed():
                support_plane_button.disabled = False

    def _register_add_sliding_direction_button(
        self, client: viser.ClientHandle
    ) -> None:
        """Register the Add Sliding Direction button for the latest grasp point."""
        self.add_sliding_direction_button_handle = client.gui.add_button(
            "Add Sliding Direction",
            icon=viser.Icon.ARROW_RIGHT,
            disabled=True,
        )

        @self.add_sliding_direction_button_handle.on_click
        def on_click(_):
            if not self.export_gripper_data:
                return

            self.add_sliding_direction_button_handle.disabled = True
            plane_points: list[np.ndarray] = []

            last_gripper_info = self.export_gripper_data[-1]
            grasp_index = len(self.export_gripper_data) - 1
            arrow_origin = np.array(last_gripper_info["world_hit_pos"])

            @client.scene.on_pointer_event(event_type="click")
            def on_plane_click(event: viser.ScenePointerEvent) -> None:
                hit = self._get_hit_point(event)
                if hit is None:
                    return

                sphere = trimesh.creation.icosphere(radius=0.02)
                sphere.vertices += hit
                sphere.visual.vertex_colors = (1.0, 0.5, 0.0, 1.0)
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=f"/sliding_plane_point_{grasp_index}_{len(plane_points)}",
                    mesh=sphere,
                )
                self.hit_pos_handles.append(sphere_handle)
                plane_points.append(hit)

                if len(plane_points) >= 4:
                    client.scene.remove_pointer_callback()
                    _, normal = self._fit_plane(plane_points)

                    for i, data in enumerate(self.export_sliding_direction_data):
                        if data["grasp_index"] == grasp_index:
                            try:
                                data["handle"].remove()
                            except KeyError:
                                pass
                            self.export_sliding_direction_data.pop(i)
                            break

                    def flip_arrow(handle):
                        clicked_data = next(
                            (
                                d
                                for d in self.export_sliding_direction_data
                                if d["handle"] == handle
                            ),
                            None,
                        )
                        if clicked_data is None:
                            return

                        new_normal = -clicked_data["normal"]
                        clicked_data["normal"] = new_normal
                        new_arrow_mesh = self._create_arrow_mesh(
                            arrow_origin, new_normal
                        )

                        try:
                            clicked_data["handle"].remove()
                        except KeyError:
                            pass

                        new_handle = self.server.scene.add_mesh_trimesh(
                            name=f"/sliding_direction_{clicked_data['grasp_index']}",
                            mesh=new_arrow_mesh,
                        )
                        clicked_data["handle"] = new_handle
                        new_handle.on_click(lambda _: flip_arrow(new_handle))

                    arrow_mesh = self._create_arrow_mesh(arrow_origin, normal)
                    arrow_handle = self.server.scene.add_mesh_trimesh(
                        name=f"/sliding_direction_{grasp_index}", mesh=arrow_mesh
                    )
                    arrow_handle.on_click(lambda _: flip_arrow(arrow_handle))

                    self.export_sliding_direction_data.append(
                        {
                            "grasp_index": grasp_index,
                            "normal": normal,
                            "handle": arrow_handle,
                        }
                    )

            @client.scene.on_pointer_callback_removed
            def on_pointer_removed():
                if self.add_sliding_direction_button_handle:
                    self.add_sliding_direction_button_handle.disabled = False

    def _register_add_rotation_axis_button(self, client: viser.ClientHandle) -> None:
        """Register the Add Rotation Axis button for the latest grasp point."""
        self.add_rotation_axis_button_handle = client.gui.add_button(
            "Add Rotation Axis", icon=viser.Icon.REPEAT, disabled=True
        )

        @self.add_rotation_axis_button_handle.on_click
        def on_click(_):
            if not self.export_gripper_data:
                return

            self.add_rotation_axis_button_handle.disabled = True
            line_points: list[np.ndarray] = []

            last_gripper_info = self.export_gripper_data[-1]
            grasp_index = len(self.export_gripper_data) - 1
            rotation_origin = np.array(last_gripper_info["world_hit_pos"])

            @client.scene.on_pointer_event(event_type="click")
            def on_line_click(event: viser.ScenePointerEvent) -> None:
                hit = self._get_hit_point(event)
                if hit is None:
                    return

                sphere = trimesh.creation.icosphere(radius=0.02)
                sphere.vertices += hit
                sphere.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=f"/rot_axis_point_{grasp_index}_{len(line_points)}",
                    mesh=sphere,
                )
                self.hit_pos_handles.append(sphere_handle)
                line_points.append(hit)

                if len(line_points) >= 4:
                    client.scene.remove_pointer_callback()
                    centroid, direction = self._fit_line(line_points)

                    for i, data in enumerate(self.export_rotation_axis_data):
                        if data["grasp_index"] == grasp_index:
                            try:
                                data["handle"].remove()
                            except KeyError:
                                pass
                            self.export_rotation_axis_data.pop(i)
                            break

                    def flip_axis(handle):
                        clicked_data = next(
                            (
                                d
                                for d in self.export_rotation_axis_data
                                if d["handle"] == handle
                            ),
                            None,
                        )
                        if clicked_data is None:
                            return

                        new_dir = -clicked_data["direction"]
                        clicked_data["direction"] = new_dir
                        new_mesh = self._create_rotation_visualization_mesh(
                            rotation_origin, clicked_data["axis_centroid"], new_dir
                        )
                        try:
                            clicked_data["handle"].remove()
                        except KeyError:
                            pass

                        new_handle = self.server.scene.add_mesh_trimesh(
                            name=f"/rotation_axis_{clicked_data['grasp_index']}",
                            mesh=new_mesh,
                        )
                        clicked_data["handle"] = new_handle
                        new_handle.on_click(lambda _: flip_axis(new_handle))

                    rot_mesh = self._create_rotation_visualization_mesh(
                        rotation_origin, centroid, direction
                    )
                    axis_handle = self.server.scene.add_mesh_trimesh(
                        name=f"/rotation_axis_{grasp_index}", mesh=rot_mesh
                    )
                    axis_handle.on_click(lambda _: flip_axis(axis_handle))

                    self.export_rotation_axis_data.append(
                        {
                            "grasp_index": grasp_index,
                            "direction": direction,
                            "axis_centroid": centroid,
                            "handle": axis_handle,
                        }
                    )

            @client.scene.on_pointer_callback_removed
            def on_pointer_removed():
                if self.add_rotation_axis_button_handle:
                    self.add_rotation_axis_button_handle.disabled = False

    def _register_export_button(self, client: viser.ClientHandle) -> None:
        """Register the Export Data button to export all defined data to pkl and json files."""
        export_button = client.gui.add_button(
            "Export Data", icon=viser.Icon.DATABASE
        )

        @export_button.on_click
        def on_click(_):
            gripper_export = {}
            for i, gripper_info in enumerate(self.export_gripper_data):
                if gripper_info.get("pose") is not None:
                    gripper_export[f"gripper_{i}"] = {
                        "position": np.array(
                            gripper_info["hit_pos"], dtype=np.float32
                        ),
                        "orientation": np.array(
                            gripper_info["pose"]["orientation"], dtype=np.float32
                        ),
                    }

            plane_export = {}
            for i, plane in enumerate(self.export_place_plane_data):
                plane_export[f"place_plane_{i}"] = {
                    "centroid": np.array(plane["centroid"], dtype=np.float32),
                    "normal": np.array(plane["normal"], dtype=np.float32),
                }
            for i, plane in enumerate(self.export_support_plane_data):
                plane_export[f"support_plane_{i}"] = {
                    "centroid": np.array(plane["centroid"], dtype=np.float32),
                    "normal": np.array(plane["normal"], dtype=np.float32),
                }

            base_dir = self.output_dir / Path(self.ply_file).stem
            plane_dir = base_dir / "plane"
            grasp_dir = base_dir / "grasp"
            plane_dir.mkdir(parents=True, exist_ok=True)
            grasp_dir.mkdir(parents=True, exist_ok=True)

            with open(grasp_dir / "gripper.pkl", "wb") as f:
                pickle.dump(gripper_export, f)
            with open(plane_dir / "plane.pkl", "wb") as f:
                pickle.dump(plane_export, f)

            print("Exported gripper data to", grasp_dir / "gripper.pkl")
            print("Exported plane data to", plane_dir / "plane.pkl")

            json_export_data = {
                "name": Path(self.ply_file).stem,
                "primitives": {},
            }
            primitives = json_export_data["primitives"]

            global_rot_info = {}
            if self.export_rotation_axis_data:
                rot_data = self.export_rotation_axis_data[0]
                global_rot_info = {
                    "rotation_axis_direction": list(rot_data["direction"]),
                    "rotation_axis_point": (
                        np.array(rot_data["axis_centroid"]) / self.scale_mesh
                    ).tolist(),
                }

            for i, gripper_info in enumerate(self.export_gripper_data):
                if gripper_info.get("pose") is not None:
                    contact_key = f"contact_point_{i}"
                    primitives[contact_key] = {
                        "type": "contact_point",
                        "description": f"Contact point {i}",
                        "position": list(gripper_info["hit_pos"]),
                        "orientation": list(gripper_info["pose"]["orientation"]),
                    }
                    for slide_data in self.export_sliding_direction_data:
                        if slide_data["grasp_index"] == i:
                            primitives[contact_key]["sliding_direction"] = list(
                                slide_data["normal"]
                            )
                            break
                    
                    primitives[contact_key].update(global_rot_info)

            plane_counter = 0
            all_planes_data = (
                self.export_place_plane_data + self.export_support_plane_data
            )
            for plane in all_planes_data:
                normal = np.array(plane["normal"])
                axes = ["x", "y", "z"]
                axis_vectors = [
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                ]
                dots = [np.abs(np.dot(normal, v)) for v in axis_vectors]
                align_axis = axes[np.argmax(dots)]
                primitives[f"plane_{plane_counter}"] = {
                    "type": "plane",
                    "description": f"Plane {plane_counter}",
                    "position": list(plane["centroid"]),
                    "orientation": list(plane["normal"]),
                    "align_with_axis": align_axis,
                }
                plane_counter += 1

            for i, point in enumerate(self.export_point_data):
                primitives[f"point_{i}"] = {
                    "type": "point",
                    "description": f"Point {i}",
                    "position": list(point["position"]),
                }

            json_file = base_dir / f"{base_dir.name}.json"
            with open(json_file, "w") as f:
                json.dump(json_export_data, f, indent=4)

            print("Exported JSON data to", json_file)

    def _register_clear_scene_button(self, client: viser.ClientHandle) -> None:
        """Register the Clear scene button to remove all added geometry and GUI elements."""
        clear_button_handle = client.gui.add_button("Clear scene", icon=viser.Icon.X)

        @clear_button_handle.on_click
        def on_click(_):
            for handle in self.hit_pos_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            self.hit_pos_handles.clear()

            all_objects = (
                list(self.gripper_dict.values())
                + self.place_plane_objects
                + self.support_plane_objects
            )
            for obj in all_objects:
                try:
                    obj.remove()
                except KeyError:
                    pass
            self.gripper_dict.clear()
            self.place_plane_objects.clear()
            self.support_plane_objects.clear()

            for data in self.export_sliding_direction_data:
                try:
                    data["handle"].remove()
                except KeyError:
                    pass
            self.export_sliding_direction_data.clear()

            for data in self.export_rotation_axis_data:
                try:
                    data["handle"].remove()
                except KeyError:
                    pass
            self.export_rotation_axis_data.clear()

            for ctrl in self.extra_gui_handles:
                ctrl.remove()
            self.extra_gui_handles.clear()

            self.export_gripper_data.clear()
            self.export_place_plane_data.clear()
            self.export_support_plane_data.clear()
            self.export_point_data.clear()

            self.mesh.visual.vertex_colors = (0.9, 0.9, 0.9, 1.0)
            self.mesh_handle = self.server.scene.add_mesh_trimesh(
                name="/mesh",
                mesh=self.mesh,
            )

            if self.add_sliding_direction_button_handle is not None:
                self.add_sliding_direction_button_handle.disabled = True
            if self.add_rotation_axis_button_handle is not None:
                self.add_rotation_axis_button_handle.disabled = True

    # --- Event Handlers & Core Logic ---

    def _handle_click_event(
        self,
        client: viser.ClientHandle,
        event: viser.ScenePointerEvent,
    ) -> None:
        """Handle Add sphere clicks and create a grasp point and gripper."""
        R_world_mesh = tf.SO3(self.mesh_handle.wxyz)
        R_mesh_world = R_world_mesh.inverse()
        origin = (R_mesh_world @ np.array(event.ray_origin)).reshape(1, 3)
        direction = (R_mesh_world @ np.array(event.ray_direction)).reshape(1, 3)

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        hit_pos, _, hit_face_ids = intersector.intersects_location(origin, direction)

        if len(hit_pos) == 0:
            return

        client.scene.remove_pointer_callback()

        closest_index = np.argmin(np.sum((hit_pos - origin) ** 2, axis=-1))
        hit_pos_val = hit_pos[closest_index]
        hit_face_index = hit_face_ids[closest_index]
        face_normal = self.mesh.face_normals[hit_face_index]
        world_hit_pos = R_world_mesh @ hit_pos_val

        hit_pos_mesh = trimesh.creation.icosphere(radius=0.05)
        hit_pos_mesh.position = hit_pos_val
        hit_pos_mesh.normal = face_normal
        hit_pos_mesh.vertices += world_hit_pos
        hit_pos_mesh.visual.vertex_colors = (0.5, 0.0, 0.7, 1.0)

        hit_pos_handle = self.server.scene.add_mesh_trimesh(
            name=f"/hit_pos_{len(self.hit_pos_handles)}", mesh=hit_pos_mesh
        )
        self.hit_pos_handles.append(hit_pos_handle)

        @hit_pos_handle.on_click
        def on_hit_pos_click(_):
            self._create_gripper(client, hit_pos_handle, hit_pos_mesh, world_hit_pos)

    def _create_gripper(
        self,
        client: viser.ClientHandle,
        hit_pos_handle: viser.GlbHandle,
        hit_pos_mesh: trimesh.Trimesh,
        world_hit_pos: np.ndarray,
    ) -> None:
        """Create a gripper from a grasp point and add GUI sliders for pose adjustment."""
        d = 0.7
        L = 1.0
        H = 1.0
        center = np.array(hit_pos_mesh.position)
        gripper_info = {
            "hit_pos": tuple(center / self.scale_mesh),
            "world_hit_pos": tuple(world_hit_pos),
            "pose": None,
        }
        self.export_gripper_data.append(gripper_info)

        n = np.array(hit_pos_mesh.normal, dtype=float)
        n /= np.linalg.norm(n)
        up = np.array([0, 0, 1], dtype=float)
        if np.abs(np.dot(up, n)) > 0.99:
            up = np.array([0, 1, 0], dtype=float)
        local_y = up - np.dot(up, n) * n
        local_y /= np.linalg.norm(local_y)
        base_pos = center - 0.2 * local_y

        def to_global_from_local(local_point: tuple[float, float]) -> np.ndarray:
            return base_pos + local_point[0] * n + local_point[1] * local_y

        orig_points = {
            "f1_start": np.array([-d / 2, 0]),
            "f1_end": np.array([-d / 2, L]),
            "f2_start": np.array([d / 2, 0]),
            "f2_end": np.array([d / 2, L]),
            "extra_start": np.array([0, L]),
            "extra_end": np.array([0, L + H]),
        }
        current_points = {
            k: to_global_from_local(tuple(v)) for k, v in orig_points.items()
        }
        last_center = 0.0
        last_axis = 0.0
        last_extra = 0.0

        gripper_mesh = trimesh.util.concatenate(
            [
                self._cylinder_between(
                    current_points["f1_start"], current_points["f1_end"], radius=0.01
                ),
                self._cylinder_between(
                    current_points["f2_start"], current_points["f2_end"], radius=0.01
                ),
                self._cylinder_between(
                    current_points["f1_end"], current_points["f2_end"], radius=0.01
                ),
                self._cylinder_between(
                    current_points["extra_start"],
                    current_points["extra_end"],
                    radius=0.01,
                ),
            ]
        )
        gripper_mesh.visual.vertex_colors = (0.5, 0.0, 0.7, 1.0)

        gripper_handle = self.server.scene.add_mesh_trimesh(
            name=f"/gripper_{hit_pos_handle.name}", mesh=gripper_mesh
        )
        self.gripper_dict[hit_pos_handle.name] = gripper_handle

        def rotate_points_around_axis(
            points: dict,
            axis_point: np.ndarray,
            axis_dir: np.ndarray,
            angle_deg: float,
        ) -> dict:
            theta = -np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            new_points = {}
            for key, p in points.items():
                v = p - axis_point
                rotated = (
                    axis_point
                    + cos_t * v
                    + sin_t * np.cross(axis_dir, v)
                    + (1 - cos_t) * np.dot(axis_dir, v) * axis_dir
                )
                new_points[key] = rotated
            return new_points

        def update_pose_display() -> None:
            pos = current_points["extra_end"]
            axle_vec = current_points["f2_end"] - current_points["f1_end"]
            x_axis = (
                axle_vec / np.linalg.norm(axle_vec)
                if np.linalg.norm(axle_vec) > 1e-6
                else np.array([0, 1, 0])
            )
            extra_vec = current_points["extra_start"] - current_points["extra_end"]
            z_axis = (
                extra_vec / np.linalg.norm(extra_vec)
                if np.linalg.norm(extra_vec) > 1e-6
                else np.array([0, 0, -1])
            )
            y_axis = np.cross(z_axis, x_axis)
            y_axis = (
                y_axis / np.linalg.norm(y_axis)
                if np.linalg.norm(y_axis) > 1e-6
                else np.array([-1, 0, 0])
            )
            R = np.column_stack((x_axis, y_axis, z_axis))
            angles_rad = trimesh.transformations.euler_from_matrix(R, axes="sxyz")
            angles_deg = np.degrees(angles_rad)
            pose_position_display.value = tuple(pos)
            pose_orientation_display.value = tuple(angles_deg)
            gripper_info["pose"] = {
                "position": tuple(pos / self.scale_mesh),
                "orientation": tuple(angles_deg),
            }

        def update_gripper_combined() -> None:
            nonlocal gripper_handle, current_points, last_center, last_axis
            delta_center = slider_center.value - last_center
            if abs(delta_center) > 1e-6:
                p1, p2, p3 = (
                    current_points["f1_start"],
                    current_points["f1_end"],
                    current_points["f2_end"],
                )
                v1 = p2 - p1
                x_axis_local = (
                    v1 / np.linalg.norm(v1)
                    if np.linalg.norm(v1) > 1e-6
                    else np.array([1, 0, 0])
                )
                v2 = p3 - p2
                normal = np.cross(
                    v1, v2 if np.linalg.norm(v2) > 1e-6 else np.array([0, 1, 0])
                )
                normal = (
                    normal / np.linalg.norm(normal)
                    if np.linalg.norm(normal) > 1e-6
                    else np.array([0, 0, 1])
                )
                y_axis_local = np.cross(normal, x_axis_local)
                y_axis_local /= np.linalg.norm(y_axis_local)
                theta = np.deg2rad(delta_center)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                for key, p in current_points.items():
                    vec = p - center
                    i, j = np.dot(vec, x_axis_local), np.dot(vec, y_axis_local)
                    new_vec = (
                        (i * cos_t + j * sin_t) * x_axis_local
                        + (-i * sin_t + j * cos_t) * y_axis_local
                    )
                    current_points[key] = center + new_vec
                last_center = slider_center.value

            delta_axis = slider_axis.value - last_axis
            if abs(delta_axis) > 1e-6:
                axis_vec = current_points["f2_start"] - current_points["f1_start"]
                axis_dir = (
                    axis_vec / np.linalg.norm(axis_vec)
                    if np.linalg.norm(axis_vec) > 1e-6
                    else np.array([1, 0, 0])
                )
                current_points = rotate_points_around_axis(
                    current_points, center, axis_dir, delta_axis
                )
                last_axis = slider_axis.value

            new_gripper_mesh = trimesh.util.concatenate(
                [
                    self._cylinder_between(
                        current_points["f1_start"],
                        current_points["f1_end"],
                        radius=0.01,
                    ),
                    self._cylinder_between(
                        current_points["f2_start"],
                        current_points["f2_end"],
                        radius=0.01,
                    ),
                    self._cylinder_between(
                        current_points["f1_end"], current_points["f2_end"], radius=0.01
                    ),
                    self._cylinder_between(
                        current_points["extra_start"],
                        current_points["extra_end"],
                        radius=0.01,
                    ),
                ]
            )
            new_gripper_mesh.visual.vertex_colors = (0.5, 0.0, 0.7, 1.0)
            gripper_handle.remove()
            gripper_handle = self.server.scene.add_mesh_trimesh(
                name=f"/gripper_{hit_pos_handle.name}", mesh=new_gripper_mesh
            )
            self.gripper_dict[hit_pos_handle.name] = gripper_handle
            update_pose_display()

        slider_center = client.gui.add_slider(
            "Rotate Gripper (°)", min=-180, max=180, step=0.1, initial_value=0
        )
        slider_axis = client.gui.add_slider(
            "Rotate Gripper Around Axis (°)",
            min=-180,
            max=180,
            step=0.1,
            initial_value=0,
        )
        slider_extra_rotate = client.gui.add_slider(
            "Rotate Gripper Around Extra Axis (°)",
            min=-180,
            max=180,
            step=0.1,
            initial_value=0,
        )
        self.extra_gui_handles.extend(
            [slider_center, slider_axis, slider_extra_rotate]
        )

        slider_center.on_update(lambda _: update_gripper_combined())
        slider_axis.on_update(lambda _: update_gripper_combined())

        @slider_extra_rotate.on_update
        def _(_):
            nonlocal last_extra, current_points
            delta_extra = slider_extra_rotate.value - last_extra
            if abs(delta_extra) > 1e-6:
                axis_point = current_points["extra_start"]
                axis_vec = current_points["extra_end"] - current_points["extra_start"]
                axis_dir = (
                    axis_vec / np.linalg.norm(axis_vec)
                    if np.linalg.norm(axis_vec) > 1e-6
                    else np.array([1, 0, 0])
                )
                current_points = rotate_points_around_axis(
                    current_points, axis_point, axis_dir, delta_extra
                )
                last_extra = slider_extra_rotate.value
                update_gripper_combined()

        pose_position_display = client.gui.add_vector3(
            "Gripper Position",
            initial_value=(0, 0, 0),
            disabled=True,
            hint="World coordinate of extra_end",
        )
        pose_orientation_display = client.gui.add_vector3(
            "Gripper Orientation (deg)",
            initial_value=(0, 0, 0),
            disabled=True,
            hint="Euler angles (rx, ry, rz)",
        )
        self.extra_gui_handles.extend(
            [pose_position_display, pose_orientation_display]
        )
        update_pose_display()

        if self.add_sliding_direction_button_handle is not None:
            self.add_sliding_direction_button_handle.disabled = False
        if self.add_rotation_axis_button_handle is not None:
            self.add_rotation_axis_button_handle.disabled = False

    # --- Geometric & Visualization Helpers ---

    def _get_hit_point(self, event: viser.ScenePointerEvent) -> np.ndarray | None:
        """Compute the nearest intersection between a mouse ray and the scene mesh."""
        R_world_mesh = tf.SO3(self.mesh_handle.wxyz)
        R_mesh_world = R_world_mesh.inverse()
        origin = (R_mesh_world @ np.array(event.ray_origin)).reshape(1, 3)
        direction = (R_mesh_world @ np.array(event.ray_direction)).reshape(1, 3)

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        hit_pos, _, _ = intersector.intersects_location(origin, direction)

        if len(hit_pos) == 0:
            return None

        closest_index = np.argmin(np.sum((hit_pos - origin) ** 2, axis=-1))
        hit_pos_val = hit_pos[closest_index]
        return R_world_mesh @ hit_pos_val

    def _fit_plane(self, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Fit a plane to 3D points with SVD and return its centroid and normal."""
        pts = np.array(points)
        centroid = pts.mean(axis=0)
        _, _, Vh = np.linalg.svd(pts - centroid)
        normal = Vh[-1]
        return centroid, normal

    def _fit_line(self, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Fit a line to 3D points with PCA and return its centroid and direction."""
        pts = np.array(points)
        centroid = pts.mean(axis=0)
        _, _, Vh = np.linalg.svd(pts - centroid)
        direction = Vh[0]
        return centroid, direction

    def _cylinder_between(
        self, p1: np.ndarray, p2: np.ndarray, radius: float, sections: int = 16
    ) -> trimesh.Trimesh:
        """Create a cylinder mesh with endpoints p1 and p2."""
        vec = p2 - p1
        height = np.linalg.norm(vec)
        if np.isclose(height, 0):
            return trimesh.Trimesh()
        cyl = trimesh.creation.cylinder(
            radius=radius, height=height, sections=sections
        )
        cyl.apply_translation([0, 0, height / 2])
        T = trimesh.geometry.align_vectors(np.array([0, 0, 1]), vec / height)
        cyl.apply_transform(T)
        cyl.apply_translation(p1)
        return cyl

    def _quaternion_from_vectors(
        self, v_from: np.ndarray, v_to: np.ndarray
    ) -> np.ndarray:
        """Compute the quaternion (wxyz) that rotates vector v_from to vector v_to."""
        v_from_norm = v_from / np.linalg.norm(v_from)
        v_to_norm = v_to / np.linalg.norm(v_to)
        dot = np.dot(v_from_norm, v_to_norm)
        if dot < -0.999999:
            axis = np.cross(v_from_norm, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v_from_norm, np.array([0, 1, 0]))
            axis /= np.linalg.norm(axis)
            return np.array([0.0, axis[0], axis[1], axis[2]])
        axis = np.cross(v_from_norm, v_to_norm)
        s = np.sqrt((1.0 + dot) * 2.0)
        invs = 1.0 / s
        return np.array([s * 0.5, axis[0] * invs, axis[1] * invs, axis[2] * invs])

    def _create_arrow_mesh(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> trimesh.Trimesh:
        """Create an arrow mesh for direction visualization."""
        length = 0.5
        radius = 0.01
        direction_norm = direction / np.linalg.norm(direction)
        shaft_height = length * 0.8
        shaft = trimesh.creation.cylinder(radius=radius, height=shaft_height)
        shaft.apply_translation([0, 0, shaft_height / 2])
        head_height = length * 0.2
        head_radius = radius * 2.5
        head = trimesh.creation.cone(radius=head_radius, height=head_height)
        head.apply_translation([0, 0, shaft_height])
        arrow = trimesh.util.concatenate([shaft, head])
        T = trimesh.geometry.align_vectors(np.array([0, 0, 1]), direction_norm)
        arrow.apply_transform(T)
        arrow.apply_translation(origin)
        arrow.visual.vertex_colors = (1.0, 0.5, 0.0, 1.0)
        return arrow

    def _create_rotation_visualization_mesh(
        self,
        trajectory_origin: np.ndarray,
        axis_origin: np.ndarray,
        axis_dir: np.ndarray,
        arc_angle_deg: float = 270.0,
    ) -> trimesh.Trimesh:
        """Create visualization meshes for the rotation axis and rotation trajectory arc."""
        axis_dir_norm = axis_dir / np.linalg.norm(axis_dir)
        v = trajectory_origin - axis_origin
        t = np.dot(v, axis_dir_norm)
        circle_center = axis_origin + t * axis_dir_norm
        radius_vec = trajectory_origin - circle_center
        radius = np.linalg.norm(radius_vec)
        axis_length = max(0.5, radius * 2.5)
        axis_line = self._cylinder_between(
            axis_origin - axis_dir_norm * axis_length / 2.0,
            axis_origin + axis_dir_norm * axis_length / 2.0,
            radius=0.005,
        )
        axis_line.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)

        if np.isclose(radius, 0):
            return axis_line

        v_perp1 = radius_vec / radius
        v_perp2 = np.cross(axis_dir_norm, v_perp1)
        num_segments = 64
        angles = np.linspace(0, np.deg2rad(arc_angle_deg), num_segments)
        path_points = [
            circle_center + radius * (np.cos(a) * v_perp1 + np.sin(a) * v_perp2)
            for a in angles
        ]
        arc_tube = trimesh.util.concatenate(
            [
                self._cylinder_between(p1, p2, radius=0.01)
                for p1, p2 in zip(path_points[:-1], path_points[1:])
            ]
        )
        arc_tube.visual.vertex_colors = (0.2, 0.8, 0.8, 1.0)

        end_point = path_points[-1]
        tangent = end_point - path_points[-2]
        head_height = 0.05
        head_radius = 0.025
        arrow_head = trimesh.creation.cone(radius=head_radius, height=head_height)
        T = trimesh.geometry.align_vectors(np.array([0, 0, 1]), tangent)
        arrow_head.apply_transform(T)
        arrow_head.apply_translation(end_point)
        arrow_head.visual.vertex_colors = (0.2, 0.8, 0.8, 1.0)

        return trimesh.util.concatenate([axis_line, arc_tube, arrow_head])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive RAM template primitive annotator."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH_PATH,
        help="Path to the .ply mesh to annotate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where exported primitive JSON/PKL files will be written.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.002,
        help="Scale applied to the mesh for visualization; exported positions are unscaled.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Viser server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = ViserApp(
        ply_file=args.mesh,
        output_dir=args.output_dir,
        scale_value=args.scale,
        host=args.host,
        port=args.port,
    )
    print(f"Template annotator is running for: {app.mesh_path}")
    print(f"Exports will be written under: {app.output_dir}")
    while True:
        time.sleep(10.0)
