"""Small anatomical plotting entry points backed by persistent PyVista scenes."""

import io
import json
import sys
import warnings
import zipfile
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from threading import current_thread, main_thread
from types import TracebackType
from typing import Any, Literal, TypeVar

import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .anatomy import _project_contacts, load_surface, roi_surface

_PlotType = TypeVar("_PlotType", bound="_BasePlot")

_SURFACE_VIEWS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "lateral": ((-1, 0, 0), (0, 0, 1)),
    "medial": ((1, 0, 0), (0, 0, 1)),
    "dorsal": ((0, 0, 1), (0, 1, 0)),
    "ventral": ((0, 0, -1), (0, 1, 0)),
    "anterior": ((0, 1, 0), (0, 0, 1)),
    "posterior": ((0, -1, 0), (0, 0, 1)),
}


def _hemispheres(
    left: pv.PolyData | str | Path | None,
    right: pv.PolyData | str | Path | None,
) -> dict[str, pv.PolyData]:
    """Load independent hemisphere meshes in left-then-right order.

    Parameters
    ----------
    left, right : pyvista.PolyData | str | pathlib.Path | None
        RAS millimetre surfaces accepted by load_surface. None omits a side;
        at least one is required.

    Returns
    -------
    meshes : dict[str, pyvista.PolyData]
        Copied meshes keyed by left/right, retaining coordinates and topology.
    """
    meshes = {
        side: load_surface(source)
        for side, source in (("left", left), ("right", right))
        if source is not None
    }
    if not meshes:
        raise ValueError("provide at least one hemisphere surface")
    return meshes


def _contact_inputs(
    xyz: ArrayLike,
    radius: float,
    color: str | Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    """Validate contact coordinates, radius, and colors before rendering.

    Parameters
    ----------
    xyz : numpy.typing.ArrayLike
        Finite RAS millimetre coordinates, shape (n_contacts, 3), with n_contacts > 0.
    radius : float
        Finite positive sphere radius in millimetres.
    color : str | collections.abc.Sequence[str]
        One PyVista color string or one per contact.

    Returns
    -------
    points : numpy.ndarray
        Copied float64 coordinates, shape (n_contacts, 3), in input order.
    rgb : numpy.ndarray
        Copied uint8 colors, shape (n_contacts, 3), in the same order. Alpha is ignored.
    """
    points = np.array(xyz, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("xyz must have shape (n_contacts, 3), with n_contacts > 0")
    if not np.isfinite(points).all():
        raise ValueError("xyz must contain finite RAS coordinates in millimetres")
    if np.ndim(radius) != 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a finite positive scalar in millimetres")
    colors = [color] * len(points) if isinstance(color, str) else list(color)
    if len(colors) != len(points) or any(not isinstance(c, str) for c in colors):
        raise ValueError("color must be a string or one color string per contact")
    rgb = np.array([pv.Color(c).int_rgb for c in colors], dtype=np.uint8)
    return points, rgb


def _contact_mesh(
    points: NDArray[np.floating],
    radius: float,
    rgb: NDArray[np.uint8],
) -> pv.PolyData:
    """Build sphere glyphs for validated contacts.

    Parameters
    ----------
    points : numpy.ndarray
        RAS millimetre centers, shape (n_contacts, 3), in input order.
    radius : float
        Shared positive sphere radius in millimetres.
    rgb : numpy.ndarray
        Uint8 RGB colors, shape (n_contacts, 3), aligned with points.

    Returns
    -------
    spheres : pyvista.PolyData
        Combined sphere geometry carrying contact_rgb colors; inputs are unchanged.
    """
    centers = pv.PolyData(points)
    centers.point_data["contact_rgb"] = rgb
    return centers.glyph(geom=pv.Sphere(radius=radius), orient=False, scale=False)


class _BasePlot:
    """Own the renderer; anatomy preparation stays in the public entry points."""

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        window_size: tuple[int, int],
        background: str,
    ) -> None:
        """Initialize the persistent renderer and its display settings.

        Parameters
        ----------
        shape : tuple[int, int]
            Renderer grid as (rows, columns), with positive dimensions.
        window_size : tuple[int, int]
            Initial total window dimensions as (width, height), in pixels.
        background : str
            PyVista background color name or hex string.

        Returns
        -------
        None
            Initializes plotter and colorbar metadata on this instance. Creates VTK
            graphics resources in offscreen mode without opening a viewer or Figure.
        """
        self._colorbars: list[tuple[ScalarMappable, dict[str, Any]]] = []
        # Cocoa native interactors require the main thread; background snapshots
        # need PyVista's offscreen interactor and Cocoa context setup.
        self.plotter: pv.Plotter = pv.Plotter(
            shape=shape,
            window_size=window_size,
            off_screen=(
                sys.platform == "darwin" and current_thread() is not main_thread()
            ),
            border=False,
        )
        # Keep static updates offscreen while retaining the native interactor
        # created above for a later explicit view().
        self.plotter.off_screen = True
        self.plotter.render_window.SetOffScreenRendering(True)
        self.plotter.set_background(background)
        for renderer in self.plotter.renderers:
            renderer.enable_parallel_projection()
        self.plotter.enable_depth_peeling()

    def _add_surface(
        self,
        mesh: pv.PolyData,
        *,
        color: str,
        opacity: float,
        scalars: str | None = None,
    ) -> pv.Actor:
        """Add a shaded surface to the active renderer without fitting or rendering.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Prepared RAS millimetre mesh retained by the actor.
        color : str
            PyVista surface color, used when scalars is None.
        opacity : float
            Surface opacity in [0, 1].
        scalars : str | None
            RGB point-data field, or None for uniform color.

        Returns
        -------
        actor : pyvista.Actor
            Added actor; geometry and camera framing are unchanged.
        """
        return self.plotter.add_mesh(
            mesh,
            color=color,
            scalars=scalars,
            rgb=scalars is not None,
            opacity=opacity,
            smooth_shading=True,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )

    def _add_contact_mesh(
        self,
        spheres: pv.PolyData,
        lighting: bool,
    ) -> pv.Actor:
        """Add contact glyphs to the active renderer without fitting or rendering.

        Parameters
        ----------
        spheres : pyvista.PolyData
            RAS millimetre spheres carrying uint8 contact_rgb point data.
        lighting : bool
            Whether scene lights shade the colors.

        Returns
        -------
        actor : pyvista.Actor
            Added opaque actor, retaining sphere geometry, radius, and colors.
        """
        return self.plotter.add_mesh(
            spheres,
            scalars="contact_rgb",
            rgb=True,
            opacity=1.0,
            smooth_shading=True,
            lighting=lighting,
            specular=0.1,
            specular_power=25.0,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )

    def to_figure(
        self,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: float | None = None,
        render_dpi: float = 600,
    ) -> Figure:
        """Create an independent Matplotlib snapshot of the current scene.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, default=None
            Target 2D Axes. None creates a Figure; supplied axes retain existing artists
            and siblings. Each call adds a snapshot and the requested colorbars.
        figsize : tuple[float, float] | None, default=None
            New Figure width and height in inches, default window_size / 100.
            Only valid when ax is None.
        dpi : float | None, default=None
            New Figure canvas DPI, default 150. Only valid when ax is None;
            supplied figures retain their size and DPI.
        render_dpi : float, default=600
            Finite positive brain-image pixels per inch, measured after layout and
            rounded up to whole pixels. Lower values provide lighter previews.

        Returns
        -------
        figure : matplotlib.figure.Figure
            New or supplied Figure, without displaying or saving. Use pyplot.show()
            or figure.savefig(). The brain occupies one Axes with its aspect preserved.

        Notes
        -----
        The snapshot keeps scene cameras and window state unchanged and survives close().
        Later scene edits, Figure resizing, or export at higher DPI do not rerender it;
        create another snapshot at the intended size and render_dpi. PNG dimensions
        follow Figure size and export DPI. PDF/SVG embed the original brain pixels with
        vector titles and colorbars. Rendering requires a working VTK graphics context.
        """
        import matplotlib.pyplot as plt

        if np.ndim(render_dpi) != 0 or not np.isfinite(render_dpi) or render_dpi <= 0:
            raise ValueError("render_dpi must be finite and positive")
        if ax is not None and (figsize is not None or dpi is not None):
            raise ValueError("figsize and dpi apply only when creating a new Figure")
        if ax is not None and ax.name != "rectilinear":
            raise ValueError("ax must be a two-dimensional rectilinear Axes")
        window = np.asarray(self.plotter.window_size)
        if ax is None:
            figure, ax = plt.subplots(
                figsize=window / 100 if figsize is None else figsize,
                dpi=150 if dpi is None else dpi,
                layout="constrained",
            )
        else:
            figure = ax.figure
        ax.set_axis_off()
        # Establish image aspect before layout determines the required pixels.
        artist = ax.imshow(
            np.zeros((1, 1, 3), dtype=np.uint8),
            extent=(0, window[0], window[1], 0),
            interpolation="none",
            origin="upper",
            aspect="equal",
        )
        for mapping, options in reversed(self._colorbars):
            # Colorbar may expand a constant norm; each Figure owns its mapping.
            bar = figure.colorbar(deepcopy(mapping), ax=ax, **deepcopy(options))
            # Matplotlib rasterizes long palettes by default; retain vector bars.
            bar.solids.set_rasterized(False)
        # Keep titles above the layout cell even with a fixed-aspect image.
        ax.set_anchor("N")
        figure.canvas.draw()
        bounds = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        ratio = min(bounds.width / window[0], bounds.height / window[1]) * render_dpi
        size = np.maximum(1, np.ceil(window * ratio)).astype(int)
        image = self._screenshot(tuple(size))
        artist.set_data(image)
        return figure

    def view(
        self,
        *,
        jupyter_backend: Literal["trame", "server", "client"] | None = None,
    ) -> None:
        """Open the current scene for three-dimensional exploration.

        Parameters
        ----------
        jupyter_backend : {"trame", "server", "client"} | None, default=None
            Notebook viewer backend. None selects Trame in Notebooks, allowing
            server/client switching. Server streams VTK images; client renders
            geometry in the browser. Explicit backends require Notebook mode.

        Returns
        -------
        None
            Displays the viewer while retaining this scene for later additions or export.
            The viewer return value is not exposed.

        Notes
        -----
        Displays geometry without Matplotlib titles or colorbars. Desktop use
        opens a native window; on macOS it requires the main thread. Press Q
        to return to Python while retaining the scene. Closing via OS controls may destroy
        its graphics context. Notebook viewers require an open scene. Browser
        rendering may differ from VTK. Use to_figure() for a publication snapshot
        and export_html() for an independent browser scene.
        """
        if jupyter_backend is not None:
            if jupyter_backend not in ("trame", "server", "client"):
                raise ValueError(
                    'jupyter_backend must be "trame", "server", or "client"'
                )
            if not self.plotter.notebook:
                raise ValueError("jupyter_backend requires Notebook mode")
        if self.plotter.notebook and jupyter_backend is None:
            jupyter_backend = "trame"
        if (
            not self.plotter.notebook
            and sys.platform == "darwin"
            and current_thread() is not main_thread()
        ):
            raise RuntimeError("Desktop view() on macOS requires the main thread")
        self.plotter.off_screen = bool(self.plotter.notebook)
        self.plotter.render_window.SetOffScreenRendering(self.plotter.off_screen)
        self.plotter.show(auto_close=False, jupyter_backend=jupyter_backend)

    def _screenshot(self, size: tuple[int, int]) -> NDArray[np.uint8]:
        """Render pixels and restore the scene's window state, including on failure.

        Parameters
        ----------
        size : tuple[int, int]
            Required image (width, height) in pixels.

        Returns
        -------
        pixels : numpy.ndarray
            Uint8 RGB or RGBA image, shape (height, width, 3 or 4), according to PyVista's
            transparency setting. Uses offscreen rendering without refitting cameras.
        """
        off_screen = self.plotter.off_screen
        window = self.plotter.window_size
        self.plotter.off_screen = True
        self.plotter.render_window.SetOffScreenRendering(True)
        try:
            self.plotter.window_size = size
            self.plotter.render()
            return self.plotter.screenshot(scale=1, return_img=True)
        finally:
            self.plotter.off_screen = off_screen
            self.plotter.render_window.SetOffScreenRendering(off_screen)
            # Restoring an onscreen window can itself restore its native size.
            self.plotter.window_size = window

    def export_html(self, path: str | Path | None = None) -> io.StringIO | None:
        """Export a self-contained interactive browser scene.

        Parameters
        ----------
        path : str | pathlib.Path | None, default=None
            Destination HTML file; its parent directory must exist. PyVista
            changes the suffix to .html. If omitted, return an in-memory buffer.

        Returns
        -------
        io.StringIO | None
            HTML buffer when path is None, otherwise None after writing the file.

        Notes
        -----
        Uses Trame scene export and its offline viewer, preserving camera
        projection and zoom. The exported file embeds geometry and a VTK.js
        viewer: readers need WebGL, but no Python server or installation.
        Browser lighting and transparency can differ from native VTK screenshots.
        Browser camera changes do not update the original Python scene.
        """
        from trame_vtk.modules.vtk.serializers.utils import reference_id
        from trame_vtk.tools.vtksz2html import write_html

        data = json.loads(self.plotter.trame.export_vtksz(None, format="json"))
        cameras = {
            reference_id(renderer.GetActiveCamera()): renderer.GetActiveCamera()
            for renderer in self.plotter.render_window.GetRenderers()
        }
        # Trame's scene serializer omits projection and zoom. Preserve them for
        # the offline viewer without changing cameras or global serializers.
        for renderer in data["scene"]["dependencies"]:
            for node in renderer["dependencies"]:
                if node["id"] in cameras:
                    camera = cameras[node["id"]]
                    node["properties"].update(
                        parallelProjection=bool(camera.GetParallelProjection()),
                        parallelScale=camera.GetParallelScale(),
                        viewAngle=camera.GetViewAngle(),
                    )
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as scene:
            scene.writestr("index.json", json.dumps(data))
        html = io.StringIO()
        write_html(archive.getvalue(), html)
        html.seek(0)
        if path is None:
            return html
        Path(path).with_suffix(".html").write_text(html.getvalue(), encoding="utf-8")
        return None

    def close(self) -> None:
        """Release the scene's graphics resources.

        Returns
        -------
        None
            Closes the underlying PyVista plotter. Existing Matplotlib snapshots and
            independent HTML exports remain usable; this scene cannot be rendered again.
        """
        self.plotter.close()

    def __enter__(self: _PlotType) -> _PlotType:
        """Enter a context that owns this plot's graphics resources.

        Returns
        -------
        plot : _BasePlot
            This same instance, retaining its concrete plot subclass. Entering does
            not create another scene or change its camera.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the plot when leaving its context, including after an error.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception class from the context body, or None on a normal exit.
        exc_value : BaseException | None
            Exception instance from the context body, or None on a normal exit.
        traceback : types.TracebackType | None
            Context body's exception traceback, or None on a normal exit.

        Returns
        -------
        None
            Releases graphics resources without suppressing an exception from the
            context body. Existing Matplotlib snapshots remain independent.
        """
        self.close()


class SurfacePlot(_BasePlot):
    """Display cortical hemispheres in named anatomical views."""

    def __init__(
        self,
        *,
        left: pv.PolyData | str | Path | None = None,
        right: pv.PolyData | str | Path | None = None,
        projection_surface: tuple[
            pv.PolyData | str | Path | None,
            pv.PolyData | str | Path | None,
        ]
        | None = None,
        views: str | Sequence[str] = "lateral",
        layout: Literal["grid", "row", "column"] = "grid",
        zoom: float = 1.0,
        color: str = "#979797",
        opacity: float = 1.0,
        window_size: tuple[int, int] = (1000, 600),
        background: str = "white",
    ) -> None:
        """Initialize hemisphere panels and their shared camera scale.

        Parameters
        ----------
        left, right : pyvista.PolyData | str | pathlib.Path | None, default=None
            Hemisphere meshes or surface files accepted by ``load_surface``. Supply
            at least one. Coordinates must be RAS millimetres; geometry is copied
            without translation, registration, or inflation. Panels are ordered
            left then right; absent hemispheres have no panel.
        projection_surface : tuple | None, default=None
            Anatomical localization surfaces in (left, right) order, each a
            PolyData or local surface path; use None for an absent hemisphere.
            Supplied and displayed hemispheres must match and have identical
            ordered triangles and corresponding vertices. Meshes are copied.
            Use original pial or midthickness geometry when displaying inflated
            surfaces. None localizes on the display meshes themselves; it cannot
            recover anatomical distances from a deformed mesh. This pair is
            shared by every projected add_contacts call. It is unused when
            project=False. Supplied meshes are still loaded and validated at
            construction. Face winding must point outward; names and vertex
            counts alone cannot verify alignment.
        views : str | collections.abc.Sequence[str], default="lateral"
            Anatomical views in display order: lateral, medial, dorsal, ventral,
            anterior, posterior. Lateral/medial directions depend on hemisphere.
            All views use RAS axes. Dorsal/ventral views put anterior at the top;
            other views put superior at the top.
        layout : str, default="grid"
            "grid": rows are views and columns are hemispheres (left then right).
            "row" or "column": flatten panels in hemisphere-first order, retaining
            the requested view order within each hemisphere. One hemisphere is
            supported with the same rules.
        zoom : float, default=1.0
            Positive magnification relative to the common fitted parallel scale.
            Values above one enlarge the surfaces and may crop them.
        color : str, default="#979797"
            Uniform surface color understood by PyVista. The neutral gray matches
            the midpoint of surfplot's default Greys_r backdrop.
        opacity : float, default=1.0
            Surface opacity from 0 (transparent) to 1 (opaque).
        window_size : tuple[int, int], default=(1000, 600)
            Total window width and height in pixels, including all panels.
        background : str, default="white"
            PyVista background color.

        Returns
        -------
        None
            Initializes this plot instance and its persistent PyVista scene.

        Attributes
        ----------
        plotter : pyvista.Plotter
            Persistent renderer. Use ``subplot(row, column)`` to select a panel
            before changing its camera. Advanced controls remain native PyVista APIs.

        Notes
        -----
        Starts with the requested cameras and a shared parallel scale. Camera changes
        affect display only. Use ``add_layer`` for vertex data. Contacts default to
        nearest-surface projection followed by display mapping. Localization uses
        ``projection_surface`` when supplied, otherwise the display meshes themselves.

        Surface materials follow surfplot/BrainSpace: ambient 0.05, diffuse 1.0,
        specular 0.1, and specular power 1.0, with smooth shading and one white
        camera headlight per panel. These defaults do not depend on PyVista's
        theme. Materials and lights remain adjustable through ``plotter``.
        """
        views = [views] if isinstance(views, str) else list(views)
        if not views or any(view not in _SURFACE_VIEWS for view in views):
            raise ValueError(f"views must contain names from {tuple(_SURFACE_VIEWS)}")
        if layout not in ("grid", "row", "column"):
            raise ValueError("layout must be 'grid', 'row', or 'column'")
        if not np.isfinite(zoom) or zoom <= 0:
            raise ValueError("zoom must be finite and positive")
        meshes = _hemispheres(left, right)
        sources = meshes
        if projection_surface is not None:
            sources = _hemispheres(*projection_surface)
            if sources.keys() != meshes.keys():
                raise ValueError(
                    "projection_surface must match the displayed hemispheres"
                )
            for side, source in sources.items():
                target = meshes[side]
                if (
                    not source.is_all_triangles
                    or source.n_points != target.n_points
                    or not np.array_equal(source.faces, target.faces)
                ):
                    raise ValueError(
                        "projection and display surfaces must share ordered triangle topology"
                    )
        count = len(meshes) * len(views)
        shape = {
            "grid": (len(views), len(meshes)),
            "row": (1, count),
            "column": (count, 1),
        }[layout]
        super().__init__(
            shape=shape,
            window_size=window_size,
            background=background,
        )
        self._meshes: dict[str, pv.PolyData] = meshes
        self._projection_meshes: dict[str, pv.PolyData] = sources
        self._panels: dict[str, list[tuple[int, int]]] = {side: [] for side in meshes}
        for mesh in meshes.values():
            mesh.point_data["layer_rgb"] = np.tile(
                pv.Color(color).int_rgb, (mesh.n_points, 1)
            ).astype(np.uint8)
        for hemisphere_index, (side, mesh) in enumerate(meshes.items()):
            for view_index, view in enumerate(views):
                panel_index = hemisphere_index * len(views) + view_index
                row, column = (
                    (view_index, hemisphere_index)
                    if layout == "grid"
                    else divmod(panel_index, shape[1])
                )
                self._panels[side].append((row, column))
                self.plotter.subplot(row, column)
                # Match surfplot/BrainSpace's soft surface material, without
                # PyVista's multi-light kit washing out vertex colors.
                renderer = self.plotter.renderer
                renderer.remove_all_lights()
                renderer.add_light(pv.Light(light_type="headlight", intensity=1.0))
                actor = self._add_surface(
                    mesh, color=color, opacity=opacity, scalars="layer_rgb"
                )
                actor.prop.lighting = True
                actor.prop.ambient = 0.05
                actor.prop.diffuse = 1.0
                actor.prop.specular = 0.1
                actor.prop.specular_power = 1.0
                direction, up = _SURFACE_VIEWS[view]
                if side == "right" and view in ("lateral", "medial"):
                    direction = tuple(-value for value in direction)
                self.plotter.view_vector(direction, viewup=up)
        scale = max(r.camera.parallel_scale for r in self.plotter.renderers) / zoom
        for renderer in self.plotter.renderers:
            renderer.camera.parallel_scale = scale
        self.plotter.subplot(0, 0)

    def add_contacts(
        self,
        xyz: ArrayLike,
        *,
        hemisphere: Literal["left", "right"] | None = None,
        project: bool = True,
        max_distance: float | None = 6.0,
        radius: float = 1.0,
        color: str | Sequence[str] = "crimson",
        lighting: bool = True,
    ) -> list[pv.Actor]:
        """Add projected or original-coordinate contacts to hemisphere views.

        Parameters
        ----------
        xyz : numpy.typing.ArrayLike
            Finite coordinates of shape (n_contacts, 3), with at least one row.
            Columns are R, A, S in millimetres, aligned with the localization
            mesh (or display mesh when project=False). Input
            coordinates are copied; row order is preserved.
        hemisphere : str | None, default=None
            None assigns each contact to the closest available localization
            surface, or display surface when project=False. Exact
            distance ties select left. A single displayed hemisphere receives
            all contacts. "left" or "right" overrides assignment for every row.
            No assignment is inferred from the sign of X.
        project : bool, default=True
            Project to the nearest point on the selected localization mesh,
            using the constructor's projection_surface when supplied, otherwise
            the display mesh. Map its triangle and barycentric position to the
            display mesh.
            Move the sphere center outward along the display face normal by
            radius + 0.05 mm. False retains supplied positions without offset.
        max_distance : float | None, default=6.0
            Maximum distance in millimetres from an input contact to its selected
            localization surface. Contacts farther away are excluded
            with one summary warning; equality is included. None disables the
            limit. Ignored when project=False. Distance is measured before
            display mapping and sphere offset.
        radius : float, default=1.0
            Positive sphere radius in millimetres, shared by this call's contacts.
        color : str | collections.abc.Sequence[str], default="crimson"
            One named or hex color, or one color string per contact in row order.
            Spheres are opaque and participate in normal depth ordering.
        lighting : bool, default=True
            Shade sphere colors using scene lights and highlights. False uses
            unlit colors. Geometry, radius, projection, and depth ordering
            are identical for both settings.

        Returns
        -------
        actors : list[pyvista.Actor]
            Actors ordered by hemisphere (left then right), then requested view.
            Only hemispheres receiving contacts create actors. Row and color
            order are preserved within each group. Repeated calls retain actors.
            An empty list is returned when every contact exceeds the distance limit.

        Notes
        -----
        Cameras and zoom are preserved. Automatic assignment chooses a display
        hemisphere, not verified anatomical membership. Projection loses depth and
        does not establish cortical membership, including for deep contacts.
        Contacts within the projection distance limit are added, but overlap,
        far-side occlusion and view clipping can still hide them. The limit is a
        display selection, not a measure of localization accuracy. Changing
        radius also changes the outward display offset. No registration or ROI
        filtering is performed. Use
        project=False for original or externally prepared display coordinates.
        """
        if hemisphere is not None and hemisphere not in self._panels:
            raise ValueError(
                "hemisphere must name a displayed hemisphere: left or right"
            )
        points, rgb = _contact_inputs(xyz, radius, color)
        if project and max_distance is not None:
            if (
                np.ndim(max_distance) != 0
                or not np.isfinite(max_distance)
                or max_distance <= 0
            ):
                raise ValueError(
                    "max_distance must be finite and positive in millimetres"
                )
        sources = self._projection_meshes if project else self._meshes
        sides = list(self._panels) if hemisphere is None else [hemisphere]
        if project and any(not sources[side].is_all_triangles for side in sides):
            raise ValueError("contact projection requires triangle surfaces")
        localized = {}
        if project or len(sides) > 1:
            localized = {
                side: sources[side].find_closest_cell(points, return_closest_point=True)
                for side in sides
            }
            distances = np.column_stack(
                [
                    np.sum((points - closest) ** 2, axis=1)
                    for _, closest in localized.values()
                ]
            )
            assignments = np.argmin(distances, axis=1)
        else:
            assignments = np.zeros(len(points), dtype=int)
        included = np.ones(len(points), dtype=bool)
        if project and max_distance is not None:
            selected_distances = distances[np.arange(len(points)), assignments]
            included = selected_distances <= max_distance**2
        groups = {}
        for index, side in enumerate(sides):
            selected = (assignments == index) & included
            if not selected.any():
                continue
            centers = points[selected]
            if project:
                cells, closest = localized[side]
                centers = _project_contacts(
                    sources[side],
                    self._meshes[side],
                    cells[selected],
                    closest[selected],
                    offset=radius + 0.05,
                )
            groups[side] = _contact_mesh(centers, radius, rgb[selected])
        excluded = np.count_nonzero(~included)
        if excluded:
            warnings.warn(
                f"Excluded {excluded} of {len(points)} contacts farther than "
                f"{max_distance:g} mm from the anatomical localization surface; "
                f"{len(points) - excluded} contacts remain for display.",
                UserWarning,
                stacklevel=2,
            )
        active = self.plotter.renderers.active_index
        actors = []
        try:
            for side, spheres in groups.items():
                for row, column in self._panels[side]:
                    self.plotter.subplot(row, column)
                    actor = self._add_contact_mesh(spheres, lighting)
                    actors.append(actor)
                    self.plotter.reset_camera_clipping_range()
        finally:
            self.plotter.subplot(*divmod(active, self.plotter.shape[1]))
        self.plotter.render()
        return actors

    def add_layer(
        self,
        data: ArrayLike | dict[str, ArrayLike],
        *,
        cmap: str | Colormap = "viridis",
        color_range: tuple[float, float] | None = None,
        cbar: bool = False,
        cbar_label: str = "",
        cbar_kws: dict[str, Any] | None = None,
    ) -> None:
        """Color surface vertices, covering earlier layers where data are present.

        Parameters
        ----------
        data : numpy.typing.ArrayLike | dict[str, numpy.typing.ArrayLike]
            One-dimensional vertex values, in surface vertex order. An array
            contains all displayed hemispheres, left then right. A dictionary
            with "left" and/or "right" keys can update individual hemispheres.
            NaN and masked values retain the previous color. Zero is a normal
            value; infinity is rejected. Inputs are copied, never resampled.
        cmap : str | matplotlib.colors.Colormap, default="viridis"
            A PyVista-supported colormap name or Matplotlib colormap. Without
            color_range, finite values across both supplied hemispheres set the
            range and constant data use the middle color. Color alpha blends
            with the previous layer; overall surface opacity is unchanged.
        color_range : tuple[float, float] | None, default=None
            Finite increasing (minimum, maximum) for continuous colors. Values
            outside this range use endpoint colors. None uses finite values
            across both supplied hemispheres; constant data use the middle
            color. Reuse a range to compare separate plots.
        cbar : bool, default=False
            Include this layer's own colorbar in subsequent Figures. Each call
            adds a separate bar; False leaves earlier requests intact. Bars do
            not appear in view() or HTML.
            Masked and NaN values are excluded from automatic ranges.
        cbar_label : str, default=""
            Colorbar title, such as the statistic name or units.
        cbar_kws : dict | None, default=None
            Matplotlib Figure.colorbar options for this layer only, such as
            ticks, format, location, orientation, fraction, pad, or shrink.
            Requires cbar=True. Options are copied when the layer is added.
            Default bar length uses shrink=0.5; override it in this dictionary.
            Bars default to horizontal; location can select another side and
            its native orientation. The last layer's bar is outermost when
            bars share a side. Reuse a dictionary for a common style.
            Mapping, target axes, and label are controlled by anatplot: use
            cmap, color_range, and cbar_label rather than the colorbar keys
            mappable, cmap, norm, ax, cax, or label.

        Returns
        -------
        None
            Updates surface colors in place and records this layer's optional colorbar.
            Geometry, cameras, and existing Figures are unchanged.

        Notes
        -----
        Colors update every view without changing cameras or adding overlapping
        meshes. Repeated calls composite colors in order; layers cannot yet be
        removed or edited independently. The colorbar shows the colormap before
        alpha blending with earlier layers or shading by scene lights.
        Use add_parcellation for discrete labels and label-to-color dictionaries.
        Vertex colors can blend across triangle interiors. Vertex counts alone cannot
        establish correspondence: templates and vertex ordering must match.
        File loading and atlas lookup tables remain upstream operations.
        """
        options = {} if cbar_kws is None else deepcopy(cbar_kws)
        if cbar_kws is not None and not cbar:
            raise ValueError("cbar_kws requires cbar=True")
        reserved = options.keys() & {"mappable", "cmap", "norm", "ax", "cax", "label"}
        if reserved:
            raise ValueError(
                f"cbar_kws cannot override layer mapping, axes, or label: {sorted(reserved)}"
            )
        if isinstance(cmap, dict):
            raise ValueError(
                "add_layer requires a continuous colormap; use add_parcellation for labels"
            )
        if color_range is not None:
            limits = np.asarray(color_range, dtype=float)
            if (
                limits.shape != (2,)
                or not np.isfinite(limits).all()
                or limits[0] >= limits[1]
            ):
                raise ValueError(
                    "color_range requires finite increasing continuous limits"
                )
            color_range = tuple(limits)
        values_by_side = self._layer_values(data)
        palette = np.asarray(pv.LookupTable(cmap=cmap).values) / 255.0
        finite = np.concatenate([v[np.isfinite(v)] for v in values_by_side.values()])
        if not len(finite):
            return
        low, high = (finite.min(), finite.max()) if color_range is None else color_range
        if low == high:
            palette = np.tile(palette[len(palette) // 2], (len(palette), 1))
        mapping = ScalarMappable(
            Normalize(low, high, clip=True), ListedColormap(palette)
        )
        colors_by_side = {
            side: mapping.to_rgba(values) for side, values in values_by_side.items()
        }
        if cbar:
            options = {"fraction": 0.06, "pad": 0.08, "shrink": 0.5, **options}
            if "orientation" not in options and "location" not in options:
                options["orientation"] = "horizontal"
            options["label"] = cbar_label
            if low == high:
                options.setdefault("ticks", [low])
            self._colorbars.append((mapping, options))
        self._composite_colors(colors_by_side)

    def _layer_values(
        self,
        data: ArrayLike | dict[str, ArrayLike],
    ) -> dict[str, NDArray[np.float64]]:
        """Copy vertex values into validated per-hemisphere arrays.

        Parameters
        ----------
        data : numpy.typing.ArrayLike | dict[str, numpy.typing.ArrayLike]
            One value per displayed vertex, packed left then right or keyed by displayed
            hemisphere. Masks become NaN; infinity, wrong lengths, and unknown sides fail
            before the scene changes.

        Returns
        -------
        values_by_side : dict[str, numpy.ndarray]
            Independent float64 arrays, shape (n_vertices,) per side, in mesh vertex order.
        """
        if not isinstance(data, dict):
            values = np.ma.asarray(data)
            sizes = [mesh.n_points for mesh in self._meshes.values()]
            if values.ndim != 1 or len(values) != sum(sizes):
                raise ValueError("data must have one value per displayed vertex")
            data = dict(zip(self._meshes, np.split(values, np.cumsum(sizes)[:-1])))
        if not data or not data.keys() <= self._meshes.keys():
            raise ValueError("data keys must name displayed hemispheres: left or right")
        values_by_side = {}
        for side, values in data.items():
            values = np.ma.asarray(values, dtype=float).filled(np.nan).copy()
            if values.ndim != 1 or len(values) != self._meshes[side].n_points:
                raise ValueError(f"{side} data must have one value per surface vertex")
            if np.isinf(values).any():
                raise ValueError("data must not contain infinity")
            values_by_side[side] = values
        return values_by_side

    def _composite_colors(
        self,
        colors_by_side: dict[str, NDArray[np.float64]],
    ) -> None:
        """Blend prepared vertex colors into the surface and render the scene.

        Parameters
        ----------
        colors_by_side : dict[str, numpy.ndarray]
            RGBA arrays of shape (n_vertices, 4) in mesh vertex order, with channels
            in [0, 1]. Zero alpha retains earlier colors, including missing vertices.

        Returns
        -------
        None
            Updates layer_rgb point data without modifying inputs, geometry, cameras,
            or existing Figure snapshots.
        """
        for side, rgba in colors_by_side.items():
            colors = self._meshes[side].point_data["layer_rgb"]
            alpha = rgba[:, 3:4]
            self._meshes[side].point_data["layer_rgb"] = np.rint(
                255 * rgba[:, :3] * alpha + colors * (1 - alpha)
            ).astype(np.uint8)
        self.plotter.render()

    def add_parcellation(
        self,
        data: ArrayLike | dict[str, ArrayLike],
        *,
        cmap: str | Colormap | dict[int, str] = "tab20",
    ) -> None:
        """Assign discrete colors to atlas labels, independent of label spacing.

        Parameters
        ----------
        data : numpy.typing.ArrayLike | dict[str, numpy.typing.ArrayLike]
            One integer label per surface vertex, in the same ordering as
            ``add_layer``. Labels may be sparse. NaN and masked values retain
            previous colors; zero is a normal label and must be masked if it
            denotes background. No registration or resampling is performed.
        cmap : str | matplotlib.colors.Colormap | dict[int, str], default="tab20"
            Assign successive palette entries to sorted unique labels across
            both supplied hemispheres. Colors repeat when the palette is
            exhausted; the same ID has the same color in both hemispheres.
            A dictionary specifies exact colors for every unmasked label.

        Returns
        -------
        None
            Updates the existing vertex colors without adding meshes or a colorbar.
            Geometry, cameras, and source labels are unchanged.

        Notes
        -----
        Palette assignments depend on the labels supplied in this call. Use a
        fixed dictionary to preserve colors across different label subsets.
        Colors composite like ``add_layer`` and can blend within triangles;
        exact parcel outlines and colorbars are not provided.
        """
        from matplotlib import colormaps

        values_by_side = self._layer_values(data)
        values = np.concatenate(list(values_by_side.values()))
        labels = np.unique(values[np.isfinite(values)])
        if np.any(labels != np.floor(labels)):
            raise ValueError("atlas data must contain integer label IDs")
        if isinstance(cmap, dict):
            if not set(labels) <= cmap.keys():
                raise ValueError("cmap must provide a color for every unmasked label")
            palette = np.array([pv.Color(cmap[label]).float_rgba for label in labels])
        else:
            colormap = colormaps[cmap] if isinstance(cmap, str) else cmap
            palette = np.asarray(colormap(np.arange(len(labels)) % colormap.N))
        if not len(labels):
            return
        colors_by_side = {}
        for side, values in values_by_side.items():
            mask = np.isfinite(values)
            rgba = np.zeros((len(values), 4))
            rgba[mask] = palette[np.searchsorted(labels, values[mask])]
            colors_by_side[side] = rgba
        self._composite_colors(colors_by_side)


class _ContactPlot(_BasePlot):
    """Shared contact rendering for single-scene anatomical plots."""

    def add_contacts(
        self,
        xyz: ArrayLike,
        *,
        radius: float = 1.0,
        color: str | Sequence[str] = "crimson",
        lighting: bool = True,
    ) -> pv.Actor:
        """Add opaque contact spheres at their supplied coordinates.

        Parameters
        ----------
        xyz : numpy.typing.ArrayLike
            Finite coordinates with shape (n_contacts, 3), with at least one
            contact. Columns are R, A, S in millimetres, in the same coordinate
            frame as the displayed anatomy. Input order is preserved.
        radius : float, default=1.0
            Positive finite sphere radius in millimetres, shared by all contacts.
            This is a display size, not an estimate of electrode dimensions.
        color : str | collections.abc.Sequence[str], default="crimson"
            One named or hex color for all contacts, or exactly one color string
            per contact in xyz order. Examples: "crimson" or ["red", "#3366CC"].
            Colors are interpreted by PyVista; spheres are opaque.
        lighting : bool, default=True
            Shade sphere colors using scene lights and highlights. False uses
            unlit colors. Geometry, radius, coordinates, and depth ordering
            are identical for both settings.

        Returns
        -------
        actor : pyvista.Actor
            One actor containing all contact spheres added by this call.
            Subsequent calls add separate actors without replacing earlier ones.

        Notes
        -----
        Coordinates are copied and never projected, registered, filtered by ROI
        membership, or moved to avoid occlusion. Contacts outside an ROI remain
        visible at their supplied locations. Spheres use real mesh geometry so
        their physical size and colors are retained in PNG and HTML exports.
        The first contact batch in an empty scene fits the camera. Later additions
        preserve camera position, direction, projection, and zoom; only the depth
        clipping range is updated. Contacts outside the current frame may remain
        offscreen. Use plotter.reset_camera() to fit all actors when needed.
        Surface transparency affects visibility; contacts obey normal depth order.
        """
        points, rgb = _contact_inputs(xyz, radius, color)
        spheres = _contact_mesh(points, radius, rgb)
        first_actor = not self.plotter.renderer.actors
        actor = self._add_contact_mesh(spheres, lighting)
        if first_actor:
            self.plotter.reset_camera(render=False)
        else:
            self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        return actor


class VolumePlot(_ContactPlot):
    """Display a translucent whole-brain surface background in one scene."""

    def __init__(
        self,
        *,
        left: pv.PolyData | str | Path | None = None,
        right: pv.PolyData | str | Path | None = None,
        color: str = "#D8D8D8",
        opacity: float = 0.1,
        window_size: tuple[int, int] = (900, 650),
        background: str = "white",
    ) -> None:
        """Initialize the whole-brain background and its dorsal camera.

        Parameters
        ----------
        left, right : pyvista.PolyData | str | pathlib.Path | None, default=None
            Anatomical hemisphere surfaces accepted by ``load_surface``. Supply at
            least one, normally both pial meshes. Meshes must already share one RAS
            coordinate frame in millimetres. Positions are retained exactly.
        color : str, default="#D8D8D8"
            Uniform PyVista surface color.
        opacity : float, default=0.1
            Brain background opacity from 0 to 1.
        window_size : tuple[int, int], default=(900, 650)
            Window width and height in pixels.
        background : str, default="white"
            PyVista background color.

        Returns
        -------
        None
            Initializes this plot instance and its persistent PyVista scene.

        Attributes
        ----------
        plotter : pyvista.Plotter
            Persistent renderer, including the shared scene camera.

        Notes
        -----
        Uses surface meshes, not voxel volume rendering or slices. The initial
        top (dorsal) view puts anterior at the top and anatomical left on the left.
        The translucent background provides whole-brain context. Inflated meshes
        do not represent physical anatomy.
        Contact coordinates must already match the anatomical surfaces; no
        projection or registration is performed.
        Adding contacts preserves the initial or manually adjusted camera framing.
        """
        meshes = _hemispheres(left, right)
        super().__init__(
            shape=(1, 1),
            window_size=window_size,
            background=background,
        )
        for mesh in meshes.values():
            self._add_surface(mesh, color=color, opacity=opacity)
        self.plotter.view_xy()


class ROIPlot(_ContactPlot):
    """Display selected region boundaries from a discrete NIfTI image."""

    def __init__(
        self,
        image: nib.Nifti1Image | nib.Nifti2Image | str | Path,
        *,
        window_size: tuple[int, int] = (900, 650),
        background: str = "white",
    ) -> None:
        """Initialize an empty region scene with an anterior-superior camera.

        Parameters
        ----------
        image : nibabel.Nifti1Image | nibabel.Nifti2Image | str | pathlib.Path
            Binary mask or discrete label image accepted by ``roi_surface``. A path
            is opened once; the image is retained and must not be modified while
            adding regions. Call ``add_roi`` to select what to display.
        window_size : tuple[int, int], default=(900, 650)
            Window width and height in pixels.
        background : str, default="white"
            PyVista background color.

        Returns
        -------
        None
            Initializes this plot instance and its persistent PyVista scene.

        Attributes
        ----------
        plotter : pyvista.Plotter
            Persistent scene. Region actors and camera controls are native PyVista.

        Notes
        -----
        The initial parallel view looks from anterior and superior toward the
        regions, with superior as view-up. In RAS this uses camera offset (0, 1, 1)
        and view-up (0, 0, 1); anatomical left appears on the viewer's right.
        The first region or contact batch fits the camera. Later additions preserve
        framing; use plotter.reset_camera() to include all actors again. For
        reproducible figures, set PyVista camera parameters in Python before
        to_figure(). Use view() to explore the anatomy.
        """
        self._image = nib.load(image) if isinstance(image, (str, Path)) else image
        super().__init__(
            shape=(1, 1),
            window_size=window_size,
            background=background,
        )
        self.plotter.view_vector((0, 1, 1), viewup=(0, 0, 1))

    def add_roi(
        self,
        labels: ArrayLike,
        *,
        color: str = "#D98C70",
        opacity: float = 0.1,
        smoothing: bool = True,
    ) -> pv.Actor:
        """Add the outer boundary of selected labels, preserving existing framing.

        Parameters
        ----------
        labels : collections.abc.Sequence[int]
            Positive IDs present in the image, merged into one display region.
            For a binary mask, use [1]. Separate calls retain separate actors.
        color : str, default="#D98C70"
            Uniform PyVista surface color.
        opacity : float, default=0.1
            Surface opacity from 0 to 1. The default is a translucent
            anatomical background for displaying internal structures.
        smoothing : bool, default=True
            Enable SurfaceNets smoothing; see ``roi_surface`` for limitations.

        Returns
        -------
        actor : pyvista.Actor
            Added surface actor. Its properties can be adjusted with PyVista.

        Notes
        -----
        Original voxel labels and the image affine are never modified. The first
        addition to an empty scene fits the camera. Later additions preserve camera
        position, direction, projection, and zoom; only the depth clipping range is
        updated. New regions outside the current frame may remain offscreen. Use
        plotter.reset_camera() to fit all actors when needed.
        """
        mesh = roi_surface(self._image, labels, smoothing=smoothing)
        first_actor = not self.plotter.renderer.actors
        actor = self._add_surface(mesh, color=color, opacity=opacity)
        if first_actor:
            self.plotter.reset_camera(render=False)
        else:
            self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        return actor
