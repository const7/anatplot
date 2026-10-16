# SurfacePlot: Advanced

Combine cortical maps in a Matplotlib figure or customize the brain's
appearance through PyVista. The examples below compare two surface shapes
and create a pale background for contacts.

## Setup

```python exec="on" source="above" html="on" session="surface-advanced" id="surface-advanced-setup"
import base64  # markdown-exec: hide
import io  # markdown-exec: hide
from pathlib import Path  # markdown-exec: hide
import matplotlib  # markdown-exec: hide

matplotlib.use("Agg")  # markdown-exec: hide
import matplotlib.pyplot as plt  # markdown-exec: hide

Path("outputs/surface").mkdir(parents=True, exist_ok=True)  # markdown-exec: hide


def _embed_static(figure):  # markdown-exec: hide
    image = io.BytesIO()  # markdown-exec: hide
    figure.savefig(image, format="png", dpi=300)  # markdown-exec: hide
    plt.close(figure)  # markdown-exec: hide
    encoded = base64.b64encode(image.getvalue()).decode("ascii")  # markdown-exec: hide
    print(  # markdown-exec: hide
        f'<img alt="{_stage}" src="data:image/png;base64,{encoded}" style="width:100%;max-width:1000px;height:auto">'  # markdown-exec: hide
    )  # markdown-exec: hide


from anatplot import SurfacePlot, datasets
```

## Compose two plots with titles

Compare the same association map on `inflated` and `veryinflated` surfaces,
stacked in two Matplotlib rows. The latter opens the folds further; both fsLR
32k meshes share vertex order, so the map values stay the same.

```python exec="on" source="above" html="on" session="surface-advanced" id="surface-composition" workdir="outputs/surface"
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    layout="constrained",
    gridspec_kw={"hspace": 0.2},  # Leave space between the two rows.
)
axes[0].set_title("A. Inflated")
axes[1].set_title("B. Very inflated")

left_map, right_map = datasets.fetch_dmn_map_fslr32k()
association = {
    "left": np.ma.masked_equal(left_map, 0),
    "right": np.ma.masked_equal(right_map, 0),
}

inflated = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
with SurfacePlot(**inflated, window_size=(1000, 350), zoom=1.5) as plot:
    plot.add_layer(
        association,
        cmap="YlOrRd",
        color_range=(0, 14),  # Use the same scale in both panels.
        cbar=True,
        cbar_label="Association Z",
        cbar_kws={"location": "right"},  # Matplotlib colorbar options.
    )
    plot.to_figure(ax=axes[0])  # Draw into an existing Matplotlib Axes.

veryinflated = datasets.fetch_surfaces("fsLR", surface="veryinflated", density="32k")
with SurfacePlot(**veryinflated, window_size=(1000, 350), zoom=1.5) as plot:
    plot.add_layer(
        association,
        cmap="YlOrRd",
        color_range=(0, 14),
        cbar=True,
        cbar_label="Association Z",
        cbar_kws={"location": "right"},
    )
    plot.to_figure(ax=axes[1])

fig.savefig("comparison.png", dpi=300)
_stage = "Inflated and veryinflated maps with titles and right hand colorbars"  # markdown-exec: hide
_embed_static(fig)  # markdown-exec: hide
```

Matplotlib controls the titles and spacing between rows. To adjust gaps
between brain views **within** one `SurfacePlot`, use
[`zoom`](surfaceplot-basics.md#brain-size-and-spacing).

## Optional: draw a soft pale surface

For a pale cortical background, start with a light surface color and soften
the shading through the underlying PyVista actors.

```python exec="on" source="above" html="on" session="surface-advanced" id="surface-pale"
surfaces = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
with SurfacePlot(
    **surfaces,
    views=["lateral", "medial"],
    window_size=(1000, 600),
    zoom=1.5,
    color="#f4f4f2",  # Brain color; background controls the canvas.
) as plot:
    # Adjust the surface material before adding contacts.
    for renderer in plot.plotter.renderers:
        for actor in renderer.actors.values():
            actor.prop.ambient = 0.60
            actor.prop.diffuse = 0.40
            actor.prop.specular = 0
    actors = plot.add_contacts([[-64, -13, 23]], radius=4)
    # Lift contact shadows and soften dark edges while retaining depth.
    for actor in actors:
        actor.prop.ambient = 0.5
        actor.prop.diffuse = 0.5
    fig = plot.to_figure(figsize=(10, 7))
_stage = "Soft pale fsLR surface with one contact"  # markdown-exec: hide
_embed_static(fig)  # markdown-exec: hide
```

These are native [PyVista material properties](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Property.html):

| PyVista property | Effect here |
| --- | --- |
| `ambient = 0.60` | More base light softens shadows |
| `diffuse = 0.40` | Less directional contrast |
| `specular = 0` | No bright highlights |

Lighting changes the appearance of surface colors; colorbars show the
underlying data-to-color mapping. Keep the material, lighting, and
`color_range` consistent when comparing figures.
