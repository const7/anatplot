# Save and display

The anatplot `plot` holds a 3D scene while you add data and choose a camera.
`plot.to_figure()` captures it in a Matplotlib `fig` that you can save or
annotate. Use `plot.view()` to explore the live scene, or `plot.export_html()`
to share an independent browser scene.

## Save a figure

Capture the current scene and save the result as a PNG:

```python
fig = plot.to_figure()
fig.savefig("brain.png", dpi=300)
```

For PDF or SVG, use `fig.savefig("brain.pdf")` or
`fig.savefig("brain.svg")`. Matplotlib titles and colorbars remain vectors;
the brain image is embedded as captured pixels. These formats do not need an
extra `savefig` DPI setting.

Call `to_figure()` after scene changes. It captures a **snapshot**: later calls
such as `plot.add_contacts(...)` do not change an existing `fig`; call
`to_figure()` again. The figure remains usable after `plot.close()` or after
leaving a `with` block.

The default `render_dpi=600` controls detail captured in the brain image,
independently of Matplotlib's figure DPI. For ordinary PNG output, keep that
default and use `fig.savefig("brain.png", dpi=300)`. This `savefig` DPI sets
the PNG pixel dimensions: a 10 × 7 inch figure at 300 DPI is
3000 × 2100 pixels.

| To adjust | Use |
| --- | --- |
| Figure size | `to_figure(figsize=(10, 7))`, or set `figsize` when creating your own Matplotlib figure |
| PNG pixel dimensions | `fig.savefig(..., dpi=300)` |
| Detail captured in the brain | `to_figure(render_dpi=600)`; raise or lower it for more detail or a lighter preview |

Raising only `savefig` DPI cannot recover missing brain detail. If you enlarge
the figure substantially, make a new snapshot at its intended size.

For titles and layouts with other charts, create a Matplotlib Axes and pass
it to `to_figure()`. The default Matplotlib DPI can still produce a sharp
brain snapshot:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("My cortical map")
plot.to_figure(ax=ax)
```

Set the title before capturing the scene so the brain is sized for the
available space.

For two separate brain plots in one figure, give each plot its own Matplotlib
Axes; the [advanced surface tutorial](tutorials/surfaceplot-advanced.md) has a
complete example.

Snapshots need an OpenGL context. For remote or headless use, see
[PyVista installation](https://docs.pyvista.org/getting-started/installation.html)
and [Jupyter setup](https://docs.pyvista.org/user-guide/jupyter/).

## Display a static figure

`to_figure()` creates a Figure without displaying or saving it. The images in
the tutorials are previews of those Figures.

In a Python script, show the figure with Matplotlib:

```python
import matplotlib.pyplot as plt

plt.show()
```

In a notebook, use `display(fig)` to show a particular snapshot. Both work after
the anatplot scene has closed.

## Explore and choose a view

Call `plot.view()` while the plot is open to rotate the 3D scene. This view
does not include Matplotlib titles or colorbars. A live notebook viewer using
Trame server rendering can synchronize camera changes with Python. For a
repeatable static figure, set the camera in Python before `to_figure()`.

On the desktop, `plot.view()` opens a window and waits. Press **Q** in that
window to return to Python and continue saving or editing the scene. Closing
the window with its operating-system close button can destroy the graphics
context.

Named `SurfacePlot` views cover common cortical orientations. For camera
customization, `plot.plotter` is the built-in PyVista plotting object; see the
[advanced surface tutorial](tutorials/surfaceplot-advanced.md) or
[ROI camera example](tutorials/roiplot.md#3-choose-the-view). The initial content
fits the camera. Later additions preserve its framing. For `ROIPlot` or
`VolumePlot`, call `plot.plotter.reset_camera()` if new objects need to fit
within the view. `SurfacePlot` can have multiple views; PyVista camera changes
apply to the selected view.

## Share an interactive scene

Call `plot.export_html("brain.html")` while the plot is open. The standalone
HTML has its own camera: rotating it does not update the Python scene. It also
omits Matplotlib titles and colorbars.

Detailed parameters are in the [plotting API](api/plotting.md).
