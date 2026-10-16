"""Observable vertex coloring, composition, and export contracts."""

import numpy as np
import pytest
import pyvista as pv

from anatplot import SurfacePlot

pytestmark = pytest.mark.render


def colors(plot, panel=0):
    actor = next(iter(plot.plotter.renderers[panel].actors.values()))
    actor.mapper.update()
    return np.asarray(pv.wrap(actor.mapper.GetInput()).point_data["layer_rgb"])


def test_layers_preserve_geometry_views_inputs_and_actor_count():
    mesh = pv.Sphere()
    original = mesh.points.copy()
    values = np.linspace(0, 1, mesh.n_points)
    with SurfacePlot(left=mesh, views=["lateral", "medial"]) as plot:
        cameras = [tuple(r.camera_position) for r in plot.plotter.renderers]
        plot.add_layer(values, cmap="viridis")
        np.testing.assert_array_equal(colors(plot), colors(plot, 1))
        assert not np.array_equal(colors(plot)[0], colors(plot)[-1])
        for r, camera in zip(plot.plotter.renderers, cameras):
            assert tuple(r.camera_position) == camera
            assert len(r.actors) == 1
    np.testing.assert_array_equal(mesh.points, original)
    np.testing.assert_array_equal(values, np.linspace(0, 1, mesh.n_points))
    assert "layer_rgb" not in mesh.point_data


def test_shared_range_zero_mask_and_color_alpha():
    mesh = pv.Sphere()
    n = mesh.n_points
    with SurfacePlot(left=mesh, right=mesh) as plot:
        plot.add_layer({"left": np.zeros(n), "right": np.ones(n)}, cmap="viridis")
        lut = pv.LookupTable(cmap="viridis").values
        np.testing.assert_array_equal(colors(plot)[0], lut[0, :3])
        np.testing.assert_array_equal(colors(plot, 1)[0], lut[-1, :3])
        packed = np.ma.array(np.full(2 * n, 7), mask=True)
        packed.mask[0] = False
        plot.add_parcellation(packed, cmap={7: "red"})
        np.testing.assert_array_equal(colors(plot)[0], [255, 0, 0])
        np.testing.assert_array_equal(colors(plot, 1)[0], lut[-1, :3])
        previous = colors(plot).copy()
        right = colors(plot, 1).copy()
        data = np.ma.array(np.full(n, 7.0), mask=np.arange(n) != 0)
        plot.add_parcellation({"left": data}, cmap={7: "#FF000080"})
        expected = np.rint(
            np.array([255, 0, 0]) * (128 / 255) + previous[0] * (127 / 255)
        )
        np.testing.assert_array_equal(colors(plot)[0], expected)
        np.testing.assert_array_equal(colors(plot)[1:], previous[1:])
        np.testing.assert_array_equal(colors(plot, 1), right)
        assert np.ma.getmaskarray(data).sum() == n - 1


def test_invalid_continuous_layer_preserves_colors_and_colorbars():
    mesh = pv.Sphere()
    with SurfacePlot(left=mesh, right=mesh) as plot:
        plot.add_layer({"left": mesh.points[:, 2]}, cbar=True)
        left = colors(plot).copy()
        right = colors(plot, 1).copy()
        with pytest.raises(ValueError):
            plot.add_layer(
                {"left": mesh.points[:, 1], "right": np.ones(mesh.n_points - 1)},
                cbar=True,
            )
        np.testing.assert_array_equal(colors(plot), left)
        np.testing.assert_array_equal(colors(plot, 1), right)
        assert len(plot.to_figure(render_dpi=100).axes) == 2


def test_per_layer_colorbars_match_ranges_and_copy_options():
    mesh = pv.Sphere()
    n = mesh.n_points
    with SurfacePlot(left=mesh, right=mesh) as plot:
        cameras = [r.camera.copy() for r in plot.plotter.renderers]
        actor_counts = [len(r.actors) for r in plot.plotter.renderers]
        left = np.ma.array(np.full(n, -2.0), mask=False)
        left[0] = -100
        left.mask[0] = True
        ticks = [-2, 0, 4]
        options = {"ticks": ticks}
        plot.add_layer(
            {"left": left, "right": np.full(n, 4.0)},
            cbar=True,
            cbar_label="First",
            cbar_kws=options,
        )
        ticks[:] = [0]
        options["ticks"] = [0]
        first_colors = colors(plot).copy()
        figure = plot.to_figure()
        first_bar = figure.axes[1]._colorbar
        assert (first_bar.norm.vmin, first_bar.norm.vmax) == (-2, 4)
        np.testing.assert_array_equal(first_bar.get_ticks(), [-2, 0, 4])
        np.testing.assert_array_equal(
            np.rint(first_bar.cmap(first_bar.norm(-2))[:3] * np.array([255] * 3)),
            first_colors[1],
        )
        plot.add_layer(
            np.full(2 * n, 7.0), cmap="magma", cbar=True, cbar_label="Constant"
        )
        constant_rgb = colors(plot)[0].copy()
        plot.add_parcellation(np.full(2 * n, 1), cmap={1: "red"})
        combined = plot.to_figure()
        bars = {axis.get_xlabel(): axis._colorbar for axis in combined.axes[1:]}
        assert set(bars) == {"First", "Constant"}
        assert (bars["First"].norm.vmin, bars["First"].norm.vmax) == (-2, 4)
        np.testing.assert_array_equal(
            np.rint(np.array(bars["Constant"].cmap(0.5)[:3]) * 255), constant_rgb
        )
        np.testing.assert_array_equal(bars["Constant"].get_ticks(), [7])
        assert len(figure.axes) == 2
        for r, camera, count in zip(plot.plotter.renderers, cameras, actor_counts):
            assert r.camera == camera
            assert len(r.actors) == count
        assert not plot.plotter.renderers.shadow_renderer.actors


def test_explicit_color_range_clips_and_matches_colorbar():
    mesh = pv.Sphere()
    data = np.resize([-4.0, 0.0, 2.0, 4.0, 8.0], mesh.n_points)
    with SurfacePlot(left=mesh) as plot:
        plot.add_layer(data, color_range=(0, 4), cbar=True)
        figure = plot.to_figure()
        bar = figure.axes[1]._colorbar
        expected = np.rint(bar.cmap(bar.norm(data))[:, :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(colors(plot), expected)
        before = colors(plot).copy()
        for limits in [(4, 0), (0, 0), (0, float("inf"))]:
            with pytest.raises(ValueError):
                plot.add_layer(data, color_range=limits, cbar=True)
        np.testing.assert_array_equal(colors(plot), before)
        assert len(plot.to_figure().axes) == 2


def test_parcellation_sparse_ids_shared_colors_masks_and_override():
    from matplotlib import colormaps

    mesh = pv.Sphere()
    n = mesh.n_points
    left = np.resize([1, 2], n)
    right = np.resize([2, 100], n)
    with SurfacePlot(left=mesh, right=mesh) as plot:
        plot.add_parcellation({"left": left, "right": right})
        expected = np.rint(np.asarray(colormaps["tab20"]([0, 1, 2]))[:, :3] * 255)
        np.testing.assert_array_equal(colors(plot)[:2], expected[:2])
        np.testing.assert_array_equal(colors(plot, 1)[:2], expected[1:])
        previous = colors(plot, 1).copy()
        masked = np.ma.array(right, mask=right == 100)
        plot.add_parcellation({"right": masked}, cmap={2: "red"})
        np.testing.assert_array_equal(colors(plot, 1)[0], [255, 0, 0])
        np.testing.assert_array_equal(colors(plot, 1)[1], previous[1])
        with pytest.raises(ValueError, match="integer label"):
            plot.add_parcellation({"left": np.full(n, 1.5)})
        with pytest.raises(ValueError, match="every unmasked label"):
            plot.add_parcellation({"left": left}, cmap={1: "red"})
        np.testing.assert_array_equal(left, np.resize([1, 2], n))


def test_continuous_alpha_and_missing_values_preserve_underlying_colors():
    from matplotlib.colors import ListedColormap

    mesh = pv.Sphere()
    values = np.ma.array(np.resize([0.0, 1.0, np.nan, 0.0], mesh.n_points))
    values.mask = np.arange(mesh.n_points) % 4 == 3
    original = values.copy()
    cmap = ListedColormap([(1, 0, 0, 0.5), (0, 0, 1, 1)])
    with SurfacePlot(left=mesh, color="lime") as plot:
        before = colors(plot).copy()
        plot.add_layer(values, cmap=cmap, color_range=(0, 1))
        actual = colors(plot)
        # PyVista's palette quantizes alpha to uint8 before compositing.
        alpha = pv.LookupTable(cmap=cmap).values[0, 3] / 255
        np.testing.assert_array_equal(
            actual[0], np.rint([255 * alpha, 255 * (1 - alpha), 0])
        )
        np.testing.assert_array_equal(actual[1], [0, 0, 255])
        np.testing.assert_array_equal(actual[2:4], before[2:4])
        plot.add_parcellation(np.ma.masked_all(mesh.n_points), cmap={})
        np.testing.assert_array_equal(colors(plot), actual)
    np.testing.assert_array_equal(values.data, original.data)
    np.testing.assert_array_equal(values.mask, original.mask)
