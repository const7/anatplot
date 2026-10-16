import base64
import io
import json
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest
import pyvista as pv
from matplotlib.colors import ListedColormap

from anatplot.plotting import ROIPlot, SurfacePlot, VolumePlot

pytestmark = pytest.mark.render


def hemispheres():
    return pv.Sphere(center=(-1.1, 0, 0)), pv.Sphere(center=(1.1, 0, 0))


@pytest.mark.parametrize(
    "layout,shape", [("grid", (2, 2)), ("row", (1, 4)), ("column", (4, 1))]
)
def test_named_views_layout_and_anatomical_directions(layout, shape):
    left, right = hemispheres()
    with SurfacePlot(
        left=left,
        right=right,
        views=["lateral", "medial"],
        layout=layout,
    ) as plot:
        assert plot.plotter.shape == shape
        for h, side in enumerate(("left", "right")):
            for v, view in enumerate(("lateral", "medial")):
                row, col = (v, h) if layout == "grid" else divmod(h * 2 + v, shape[1])
                plot.plotter.subplot(row, col)
                camera = plot.plotter.camera
                direction = np.asarray(camera.position) - camera.focal_point
                expected = (
                    -1
                    if (side, view) in [("left", "lateral"), ("right", "medial")]
                    else 1
                )
                assert np.sign(direction[0]) == expected
                np.testing.assert_allclose(direction[1:], 0, atol=1e-8)
                np.testing.assert_allclose(camera.up, (0, 0, 1))
        scales = [r.camera.parallel_scale for r in plot.plotter.renderers]
        np.testing.assert_allclose(scales, scales[0])


@pytest.mark.parametrize(
    "view,direction,up",
    [
        ("dorsal", (0, 0, 1), (0, 1, 0)),
        ("ventral", (0, 0, -1), (0, 1, 0)),
        ("anterior", (0, 1, 0), (0, 0, 1)),
        ("posterior", (0, -1, 0), (0, 0, 1)),
    ],
)
def test_global_anatomical_views_and_zoom(view, direction, up):
    with SurfacePlot(left=pv.Sphere(), views=view) as fitted:
        scale = fitted.plotter.camera.parallel_scale
    with SurfacePlot(left=pv.Sphere(), views=view, zoom=1.5) as plot:
        camera = plot.plotter.camera
        actual = np.asarray(camera.position) - camera.focal_point
        np.testing.assert_allclose(actual / np.linalg.norm(actual), direction)
        np.testing.assert_allclose(camera.up, up)
        assert camera.parallel_scale == pytest.approx(scale / 1.5)


def test_html_export_preserves_cameras_layers_and_contacts(tmp_path):
    left, right = hemispheres()
    with SurfacePlot(left=left, right=right, views=["lateral", "medial"]) as plot:
        plot.add_parcellation(
            {"left": np.zeros(left.n_points), "right": np.ones(right.n_points)},
            cmap={0: "red", 1: "blue"},
        )
        plot.add_contacts([[-1.1, 0, 0]], project=False, color="gold")
        expected = []
        for i, renderer in enumerate(plot.plotter.renderers):
            camera = renderer.camera
            camera.parallel_scale = 2.0 + i
            camera.parallel_projection = i % 2 == 0
            camera.view_angle = 25.0 + i
            expected.append(
                (camera.parallel_projection, camera.parallel_scale, camera.view_angle)
            )
        cameras = [tuple(r.camera_position) for r in plot.plotter.renderers]
        assert "<html" in plot.export_html().getvalue().lower()
        plot.export_html(tmp_path / "scene.html")
        html = (tmp_path / "scene.html").read_text()
        assert [tuple(r.camera_position) for r in plot.plotter.renderers] == cameras
        payload = re.search(r'const base64Str = "([^"]+)"', html).group(1)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
            data = json.loads(archive.read("index.json"))
        actual = []
        for renderer in data["scene"]["dependencies"]:
            for node in renderer["dependencies"]:
                if node["type"] == "vtkOpenGLCamera":
                    props = node["properties"]
                    actual.append(
                        (
                            props["parallelProjection"],
                            props["parallelScale"],
                            props["viewAngle"],
                        )
                    )
        assert actual == expected
        nodes = [data["scene"]]
        exported_colors = set()
        while nodes:
            node = nodes.pop()
            nodes.extend(node.get("dependencies", []))
            if node["type"] == "vtkPolyData":
                for field in node["properties"]["fields"]:
                    if field["name"] in ("layer_rgb", "contact_rgb"):
                        encoded = data["hashes"][field["hash"]]["content"]
                        values = np.frombuffer(
                            base64.b64decode(encoded), dtype=np.uint8
                        )
                        rgb = tuple(np.unique(values.reshape(-1, 3), axis=0).ravel())
                        exported_colors.add((field["name"], rgb))
        assert exported_colors == {
            ("layer_rgb", (255, 0, 0)),
            ("layer_rgb", (0, 0, 255)),
            ("contact_rgb", (255, 215, 0)),
        }
        plot.to_figure().savefig(tmp_path / "after-html.png")


@pytest.mark.parametrize("parallel_projection", [True, False])
def test_volume_defaults_to_top_view_and_contacts_preserve_framing(parallel_projection):
    left, right = hemispheres()
    with VolumePlot(left=left, right=right) as plot:
        camera = plot.plotter.camera
        direction = np.asarray(camera.position) - camera.focal_point
        np.testing.assert_allclose(direction / np.linalg.norm(direction), (0, 0, 1))
        np.testing.assert_allclose(camera.up, (0, 1, 0))
        assert camera.parallel_projection

        plot.plotter.camera_position = [(8, -10, 12), (0, 0, 0), (0, 0, 1)]
        camera.zoom(2)
        camera.parallel_projection = parallel_projection
        camera.view_angle = 31
        framing = (tuple(plot.plotter.camera_position), camera.parallel_scale)
        clipping = camera.clipping_range
        plot.add_contacts([[-20, 0, 0], [0, 0, -80]], radius=0.1)
        assert (tuple(plot.plotter.camera_position), camera.parallel_scale) == framing
        assert camera.parallel_projection == parallel_projection
        assert camera.view_angle == 31
        assert camera.clipping_range != clipping


@pytest.mark.parametrize("first_content", ["roi", "contacts"])
def test_roi_camera_fits_first_content_and_preserves_later_framing(first_content):
    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[2:7, 4:12, 4:12] = 1
    data[9:14, 4:12, 4:12] = 2
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_xyzt_units("mm")
    with ROIPlot(image, window_size=(240, 180)) as plot:
        first = (
            plot.add_roi([1])
            if first_content == "roi"
            else plot.add_contacts([[3, 7, 7]], radius=2)
        )
        camera = plot.plotter.camera
        np.testing.assert_allclose(camera.focal_point, first.mapper.dataset.center)
        direction = np.asarray(camera.position) - camera.focal_point
        np.testing.assert_allclose(
            direction / np.linalg.norm(direction), np.array([0, 1, 1]) / np.sqrt(2)
        )
        np.testing.assert_allclose(camera.up, (0, 0, 1))
        assert camera.parallel_projection
        plot.plotter.camera_position = [(40, -40, 40), (8, 8, 8), (0, 0, 1)]
        camera.parallel_scale = 18
        camera.zoom(2)
        camera.view_angle = 37
        framing = (tuple(plot.plotter.camera_position), camera.parallel_scale)
        clipping = camera.clipping_range
        plot.add_roi([2])
        assert (tuple(plot.plotter.camera_position), camera.parallel_scale) == framing
        plot.add_contacts([[30, 0, -80]], radius=0.5)
        assert (tuple(plot.plotter.camera_position), camera.parallel_scale) == framing
        assert camera.parallel_projection
        assert camera.view_angle == 37
        assert camera.clipping_range != clipping
        figure = plot.to_figure(render_dpi=100)
        assert np.any(figure.axes[0].images[0].get_array()[:, :, :3] < 245)
        assert (tuple(plot.plotter.camera_position), camera.parallel_scale) == framing
        plot.plotter.reset_camera()
        np.testing.assert_allclose(camera.focal_point, plot.plotter.renderer.center)
        assert camera.parallel_scale > framing[1]


def test_interactive_display_retains_scene_for_export(tmp_path, monkeypatch):
    monkeypatch.setattr(pv.global_theme, "notebook", False)
    mesh = pv.Sphere()
    with SurfacePlot(left=mesh) as plot:
        camera = tuple(plot.plotter.camera_position)
        assert not plot.plotter.iren.interactor.IsA("vtkGenericRenderWindowInteractor")
        assert plot.plotter.off_screen
        assert plot.plotter.render_window.GetOffScreenRendering()
        assert not plot.plotter.render_window.GetShowWindow()
        plot.to_figure(render_dpi=100)
        assert plot.plotter.off_screen
        assert plot.plotter.render_window.GetOffScreenRendering()
        assert not plot.plotter.render_window.GetShowWindow()
        plot.add_layer(mesh.points[:, 2])
        assert plot.plotter.off_screen
        assert plot.plotter.render_window.GetOffScreenRendering()
        assert not plot.plotter.render_window.GetShowWindow()
        starts = []
        monkeypatch.setattr(plot.plotter.iren, "start", lambda: starts.append(True))
        plot.view()
        assert starts == [True]
        assert not plot.plotter.off_screen
        assert not plot.plotter.render_window.GetOffScreenRendering()
        assert plot.plotter.render_window.GetShowWindow()
        assert plot.plotter.render_window is not None
        plot.to_figure(render_dpi=100).savefig(tmp_path / "retained.png")
        assert tuple(plot.plotter.camera_position) == camera
        plot.close()
        assert plot.plotter.render_window is None


def test_figure_is_independent_and_preserves_scene(tmp_path):
    with SurfacePlot(left=pv.Sphere(), window_size=(200, 150)) as plot:
        camera = plot.plotter.camera.copy()
        figure = plot.to_figure(figsize=(4, 3), dpi=100)
        image = np.asarray(figure.axes[0].images[0].get_array()).copy()
        assert np.any(image[:, :, :3] < 245)
        assert plot.plotter.window_size == [200, 150]
        assert plot.plotter.camera == camera
        plot.add_contacts([[0, 0, 0]], radius=0.2)
        np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), image)
    figure.axes[0].set_title("Editable after scene closes")
    figure.savefig(tmp_path / "snapshot.png")
    figure.savefig(tmp_path / "snapshot.svg")
    assert "Editable after scene closes" in (tmp_path / "snapshot.svg").read_text()
    from PIL import Image

    assert Image.open(tmp_path / "snapshot.png").size == (400, 300)


def test_render_density_controls_native_pixels_independently_of_canvas():
    with SurfacePlot(left=pv.Sphere(), window_size=(200, 150)) as plot:
        for density in [None, 150]:
            figure = plt.figure(figsize=(2, 1.5), dpi=250)
            ax = figure.add_axes((0, 0, 1, 1))
            options = {} if density is None else {"render_dpi": density}
            plot.to_figure(ax=ax, **options)
            pixels = ax.images[0].get_array()
            effective_density = 600 if density is None else density
            np.testing.assert_allclose(
                pixels.shape[:2],
                [1.5 * effective_density, 2 * effective_density],
                atol=1,
                rtol=0,
            )
            assert figure.dpi == 250
            assert plot.plotter.window_size == [200, 150]
            plt.close(figure)


def test_existing_axes_preserve_siblings_and_figure_settings():
    figure, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=120)
    (line,) = axes[1].plot([0, 1], [2, 3])
    axes[0].set_title("Existing title")
    with SurfacePlot(left=pv.Sphere()) as plot:
        assert plot.to_figure(ax=axes[0]) is figure
        assert axes[0].get_title() == "Existing title"
        assert list(axes[1].lines) == [line]
        assert not axes[1].images
        np.testing.assert_allclose(figure.get_size_inches(), [7, 3])
        assert figure.dpi == 120
        with pytest.raises(ValueError):
            plot.to_figure(ax=axes[0], figsize=(2, 2))
        with pytest.raises(ValueError):
            plot.to_figure(ax=axes[0], dpi=300)


def test_global_image_defaults_cannot_flip_or_stretch_anatomy():
    mesh = pv.Sphere()
    with plt.rc_context({"image.origin": "lower", "image.aspect": "auto"}):
        figure = plt.figure(figsize=(4, 2), dpi=100)
        ax = figure.add_axes((0, 0, 1, 1))
        with SurfacePlot(left=mesh, window_size=(400, 300)) as plot:
            plot.add_layer(mesh.points[:, 2], cmap=ListedColormap(["blue", "red"]))
            plot.to_figure(ax=ax, render_dpi=100)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        upper, lower = np.split(pixels, 2)
        assert np.count_nonzero(upper[:, :, 0] > 2 * upper[:, :, 2].astype(float)) > 0
        assert not np.any(upper[:, :, 2] > 2 * upper[:, :, 0].astype(float))
        assert np.count_nonzero(lower[:, :, 2] > 2 * lower[:, :, 0].astype(float)) > 0
        assert not np.any(lower[:, :, 0] > 2 * lower[:, :, 2].astype(float))
        positions = ax.transData.transform([[0, 0], [100, 0], [0, 100]])
        np.testing.assert_allclose(
            np.linalg.norm(positions[1] - positions[0]),
            np.linalg.norm(positions[2] - positions[0]),
        )


@pytest.mark.parametrize("off_screen", [False, True])
def test_screenshot_failure_restores_scene(monkeypatch, off_screen):
    with SurfacePlot(left=pv.Sphere(), window_size=(200, 150)) as plot:
        plot.plotter.off_screen = off_screen
        plot.plotter.render_window.SetOffScreenRendering(off_screen)
        camera = plot.plotter.camera.copy()

        def fail(**kwargs):
            raise RuntimeError("render failed")

        monkeypatch.setattr(plot.plotter, "screenshot", fail)
        with pytest.raises(RuntimeError, match="render failed"):
            plot.to_figure(figsize=(4, 3), dpi=100)
        assert plot.plotter.window_size == [200, 150]
        assert plot.plotter.off_screen == off_screen
        assert bool(plot.plotter.render_window.GetOffScreenRendering()) == off_screen
        assert plot.plotter.camera == camera


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Cocoa requires main-thread windows"
)
def test_macos_background_snapshot_and_desktop_view_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(pv.global_theme, "notebook", False)

    def snapshot():
        with SurfacePlot(left=pv.Sphere(), window_size=(200, 150)) as plot:
            figure = plot.to_figure(figsize=(2, 1.5), render_dpi=150)
            assert np.any(figure.axes[0].images[0].get_array()[:, :, :3] < 245)
            with pytest.raises(RuntimeError):
                plot.view()
            return figure

    with ThreadPoolExecutor(max_workers=1) as executor:
        figure = executor.submit(snapshot).result()
    figure.savefig(tmp_path / "background.png")
