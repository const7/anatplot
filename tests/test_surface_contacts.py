"""Contact projection and routing on cortical panels."""

import numpy as np
import pytest
import pyvista as pv

from anatplot import SurfacePlot

pytestmark = pytest.mark.render


@pytest.mark.parametrize("layout", ["grid", "row"])
def test_contacts_follow_hemisphere_views_without_moving_cameras(layout):
    mesh = pv.Sphere(radius=5)
    with SurfacePlot(
        left=mesh,
        right=mesh,
        views=["lateral", "medial"],
        layout=layout,
    ) as plot:
        cameras = [
            (tuple(r.camera_position), r.camera.parallel_scale)
            for r in plot.plotter.renderers
        ]
        plot.plotter.subplot(*divmod(3, plot.plotter.shape[1]))
        # Positive X deliberately: hemisphere routing must not guess from XYZ.
        xyz = np.array([[5.0, -1, 0], [5.0, 1, 0]])
        actors = plot.add_contacts(
            xyz, hemisphere="left", project=False, radius=0.6, color=["red", "blue"]
        )
        assert len(actors) == 2
        assert plot.plotter.renderers.active_index == 3
        selected = [0, 2] if layout == "grid" else [0, 1]
        for index, (renderer, camera) in enumerate(
            zip(plot.plotter.renderers, cameras)
        ):
            assert len(renderer.actors) == (2 if index in selected else 1)
            assert tuple(renderer.camera_position) == camera[0]
            assert renderer.camera.parallel_scale == camera[1]
        plot.add_parcellation({"left": np.zeros(mesh.n_points)}, cmap={0: "lightgray"})
        assert actors[0] in plot.plotter.renderers[selected[0]].actors.values()


def test_default_projection_maps_to_display_and_offsets_spheres():
    source = pv.PolyData.from_regular_faces(
        np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]]), [[0, 1, 2]]
    )
    display = source.copy()
    display.points = np.array([[10.0, 0, 2], [14, 0, 2], [10, 6, 2]])
    xyz = np.array([[0.5, 0.5, -3.0]])
    with SurfacePlot(left=display, projection_surface=(source, None)) as plot:
        projected = plot.add_contacts(xyz, radius=0.2)[0]
        np.testing.assert_allclose(
            projected.mapper.dataset.center, [11, 1.5, 2.25], atol=1e-5
        )
        source.points[:] += [100, 0, 0]
        repeated = plot.add_contacts(xyz, radius=0.2)[0]
        np.testing.assert_allclose(
            repeated.mapper.dataset.center, [11, 1.5, 2.25], atol=1e-5
        )
        original = plot.add_contacts(xyz, project=False, radius=0.2)[0]
        np.testing.assert_allclose(original.mapper.dataset.center, xyz[0], atol=1e-5)
    np.testing.assert_array_equal(xyz, [[0.5, 0.5, -3.0]])


def test_default_projection_uses_display_mesh():
    display = pv.PolyData.from_regular_faces(
        np.array([[10.0, 0, 2], [14, 0, 2], [10, 6, 2]]), [[0, 1, 2]]
    )
    with SurfacePlot(left=display) as plot:
        actor = plot.add_contacts([[0.5, 0.5, -3.0]], max_distance=None, radius=0.2)[0]
        np.testing.assert_allclose(
            actor.mapper.dataset.center, [10, 0.5, 2.25], atol=1e-5
        )


def test_projection_rejects_unmatched_topology():
    source = pv.Sphere()
    display = source.copy()
    display.faces = display.faces.reshape(-1, 4)[::-1].ravel()
    with pytest.raises(ValueError, match="topology"):
        SurfacePlot(left=display, projection_surface=(source, None))


def test_polygon_surface_requires_explicit_original_coordinate_contacts():
    mesh = pv.Plane()
    with SurfacePlot(left=mesh) as plot:
        with pytest.raises(ValueError, match="triangle"):
            plot.add_contacts([[0, 0, 0]])
        assert len(plot.plotter.renderer.actors) == 1
        actor = plot.add_contacts([[0, 0, 0]], project=False)[0]
        np.testing.assert_allclose(actor.mapper.dataset.center, [0, 0, 0], atol=1e-5)


def test_projection_distance_limit_uses_original_surface_and_preserves_colors():
    source = pv.PolyData.from_regular_faces(
        np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]]), [[0, 1, 2]]
    )
    display = source.copy().translate((30, 0, 20))
    xyz = np.array([[0.5, 0.5, -6], [0.5, 0.5, -6.01], [0.7, 0.5, 0]])
    original = xyz.copy()
    with SurfacePlot(left=display, projection_surface=(source, None)) as plot:
        with pytest.warns(UserWarning) as recorded:
            actors = plot.add_contacts(
                xyz,
                radius=0.2,
                color=["red", "gold", "blue"],
            )
        assert len(recorded) == 1
        assert len(actors) == 1
        geometry = actors[0].mapper.dataset
        for rgb, center in (
            ([255, 0, 0], [30.5, 0.5, 20.25]),
            ([0, 0, 255], [30.7, 0.5, 20.25]),
        ):
            points = geometry.points[np.all(geometry["contact_rgb"] == rgb, axis=1)]
            np.testing.assert_allclose(
                (points.min(axis=0) + points.max(axis=0)) / 2, center, atol=1e-5
            )
        np.testing.assert_array_equal(
            np.unique(geometry["contact_rgb"], axis=0), [[0, 0, 255], [255, 0, 0]]
        )
    np.testing.assert_array_equal(xyz, original)


def test_all_contacts_beyond_limit_leave_existing_scene_unchanged():
    with SurfacePlot(left=pv.Sphere(radius=1)) as plot:
        previous = plot.add_contacts([[1, 0, 0]])
        camera = tuple(plot.plotter.camera_position)
        with pytest.warns(UserWarning) as recorded:
            actors = plot.add_contacts([[20, 0, 0], [30, 0, 0]])
        assert len(recorded) == 1
        assert actors == []
        assert len(plot.plotter.renderer.actors) == 2
        assert previous[0] in plot.plotter.renderer.actors.values()
        assert tuple(plot.plotter.camera_position) == camera
        assert len(plot.add_contacts([[20, 0, 0]], max_distance=None)) == 1


def test_distance_limit_checks_explicitly_selected_hemisphere():
    with SurfacePlot(
        left=pv.Sphere(radius=1, center=(-10, 0, 0)),
        right=pv.Sphere(radius=1, center=(10, 0, 0)),
    ) as plot:
        assert len(plot.add_contacts([[10, 0, 0]])) == 1
        with pytest.warns(UserWarning):
            assert plot.add_contacts([[10, 0, 0]], hemisphere="left") == []


def test_automatic_assignment_uses_anatomical_surfaces_and_preserves_colors():
    left = pv.Sphere(radius=1, center=(5, 0, 0))
    right = pv.Sphere(radius=1, center=(10, 0, 0))
    # Display positions deliberately swap proximity; localization must use originals.
    display_left = left.copy().translate((20, 0, 0))
    display_right = right.copy().translate((-20, 0, 0))
    xyz = np.array([[10, 0, 0], [5, 0, 0], [10, 0.2, 0]])
    with SurfacePlot(
        left=display_left,
        right=display_right,
        projection_surface=(left, right),
        views=["lateral", "medial"],
    ) as plot:
        cameras = [tuple(r.camera_position) for r in plot.plotter.renderers]
        plot.plotter.subplot(*divmod(3, plot.plotter.shape[1]))
        actors = plot.add_contacts(
            xyz,
            color=["red", "blue", "gold"],
            radius=0.2,
        )
        assert len(actors) == 4
        assert plot.plotter.renderers.active_index == 3
        for actor in actors[:2]:
            assert set(map(tuple, actor.mapper.dataset["contact_rgb"])) == {(0, 0, 255)}
        for actor in actors[2:]:
            assert set(map(tuple, actor.mapper.dataset["contact_rgb"])) == {
                (255, 0, 0),
                (255, 215, 0),
            }
        assert [tuple(r.camera_position) for r in plot.plotter.renderers] == cameras
        indices = [[0, 2], [1, 3]]
        for group, panels in zip([actors[:2], actors[2:]], indices):
            for actor, index in zip(group, panels):
                assert actor in plot.plotter.renderers[index].actors.values()
    np.testing.assert_array_equal(xyz, [[10, 0, 0], [5, 0, 0], [10, 0.2, 0]])


def test_original_positions_route_by_display_surface_without_offset():
    source_left = pv.Sphere(center=(10, 0, 0))
    source_right = pv.Sphere(center=(5, 0, 0))
    with SurfacePlot(
        left=pv.Sphere(center=(5, 0, 0)),
        right=pv.Sphere(center=(10, 0, 0)),
        projection_surface=(source_left, source_right),
    ) as plot:
        xyz = [[10, 0, 0], [5, 0, 0]]
        actors = plot.add_contacts(
            xyz, project=False, max_distance=0.1, color=["red", "blue"]
        )
        np.testing.assert_allclose(actors[0].mapper.dataset.center, xyz[1], atol=1e-5)
        np.testing.assert_allclose(actors[1].mapper.dataset.center, xyz[0], atol=1e-5)
