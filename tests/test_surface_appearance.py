"""Surface appearance remains independent of data, geometry, and cameras."""

import numpy as np
import pytest
import pyvista as pv

from anatplot import SurfacePlot

pytestmark = pytest.mark.render


@pytest.mark.parametrize("color", ["#979797", "#f4f4f2"])
def test_surface_defaults_do_not_depend_on_base_color(color):
    with SurfacePlot(
        left=pv.Sphere(), views=["lateral", "medial"], color=color
    ) as plot:
        for renderer in plot.plotter.renderers:
            actor = next(iter(renderer.actors.values()))
            assert actor.prop.lighting
            assert actor.prop.ambient == 0.05
            assert actor.prop.diffuse == 1.0
            assert actor.prop.specular == 0.1
            assert actor.prop.specular_power == 1.0
            np.testing.assert_array_equal(
                actor.mapper.dataset["layer_rgb"][0], pv.Color(color).int_rgb
            )
            assert len(renderer.lights) == 1
            light = renderer.lights[0]
            assert light.is_headlight
            assert light.intensity == 1.0
            assert light.diffuse_color == pv.Color("white")


def test_native_appearance_survives_layers_contacts_and_snapshots():
    mesh = pv.Sphere()
    with SurfacePlot(left=mesh, views=["lateral", "medial"]) as plot:
        surfaces = []
        cameras = []
        for renderer in plot.plotter.renderers:
            actor = next(iter(renderer.actors.values()))
            actor.prop.ambient = 0.6
            actor.prop.diffuse = 0.45
            actor.prop.specular = 0.02
            actor.prop.specular_power = 20
            surfaces.append(actor)
            renderer.remove_all_lights()
            renderer.add_light(pv.Light(light_type="cameralight", intensity=0.7))
            cameras.append(
                (tuple(renderer.camera_position), renderer.camera.parallel_scale)
            )
        plot.add_layer(mesh.points[:, 2], cbar=True)
        contacts = plot.add_contacts([[-1, 0, 0]], radius=0.1)
        plot.add_parcellation(np.ones(mesh.n_points), cmap={1: "red"})
        figure = plot.to_figure(render_dpi=60)
        assert len(figure.axes) == 2
        for renderer, surface, camera in zip(plot.plotter.renderers, surfaces, cameras):
            assert (
                tuple(renderer.camera_position),
                renderer.camera.parallel_scale,
            ) == camera
            assert surface.prop.ambient == 0.6
            assert surface.prop.diffuse == 0.45
            assert surface.prop.specular == 0.02
            assert surface.prop.specular_power == 20
            assert len(renderer.lights) == 1
            assert renderer.lights[0].is_camera_light
            assert renderer.lights[0].intensity == 0.7
        assert all(actor.prop.specular_power == 25 for actor in contacts)
