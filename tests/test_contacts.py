import numpy as np
import pytest
import pyvista as pv

from anatplot import VolumePlot

pytestmark = pytest.mark.render


def test_contact_positions_radii_colors_and_input_ownership():
    # Include a contact outside the background and a coincident pair.
    xyz = np.array([[-3.0, 0.0, 0.0], [30.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    original = xyz.copy()
    with VolumePlot(left=pv.Sphere(radius=8)) as scene:
        direction = scene.plotter.camera.direction
        actor = scene.add_contacts(xyz, radius=1.25, color=["red", "#0000ff", "lime"])
        mesh = actor.mapper.dataset
        np.testing.assert_allclose(scene.plotter.camera.direction, direction)
        for center, color in zip(original, ([255, 0, 0], [0, 0, 255], [0, 255, 0])):
            points = mesh.points[np.all(mesh["contact_rgb"] == color, axis=1)]
            np.testing.assert_allclose(points.mean(axis=0), center, atol=1e-5)
            np.testing.assert_allclose(
                np.linalg.norm(points - center, axis=1), 1.25, atol=1e-5
            )
        np.testing.assert_array_equal(xyz, original)
        copied = mesh.points.copy()
        xyz[:] = 100
        np.testing.assert_array_equal(mesh.points, copied)
        other = scene.add_contacts([[0, 0, 0]], color="gold")
        assert other is not actor
        assert actor in scene.plotter.renderer.actors.values()
        np.testing.assert_array_equal(
            np.unique(other.mapper.dataset["contact_rgb"], axis=0), [[255, 215, 0]]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xyz": []},
        {"xyz": [[0, 0, float("nan")]]},
        {"color": ["red"]},
    ],
)
def test_invalid_contact_inputs_do_not_change_scene(kwargs):
    with VolumePlot(left=pv.Sphere()) as scene:
        options = {"xyz": [[0, 0, 0], [1, 1, 1]], **kwargs}
        before = len(scene.plotter.renderer.actors)
        with pytest.raises(ValueError):
            scene.add_contacts(**options)
        assert len(scene.plotter.renderer.actors) == before
