# SurfacePlot: Basics

This tutorial builds a cortical map with electrode contacts, then shows how
to display atlas labels on the same surfaces.

## 1. Fetch matching surfaces and shading

Fetch public fsLR 32k inflated surfaces for the visible brain and sulcal depth
from the same template and density for its shading. More built-in templates
and atlases are listed in [Datasets](../datasets.md).

```python exec="on" source="above" html="on" session="surface-tutorial" id="surface-step-1"
import base64  # markdown-exec: hide
import io  # markdown-exec: hide
from pathlib import Path  # markdown-exec: hide
import matplotlib  # markdown-exec: hide

matplotlib.use("Agg")  # markdown-exec: hide
import matplotlib.pyplot as plt  # markdown-exec: hide
import pyvista as pv  # markdown-exec: hide
from trame.app import get_server  # markdown-exec: hide

server = get_server(pv.global_theme.trame.jupyter_server_name)  # markdown-exec: hide
server.cli.set_defaults(nosignalhandlers=True)  # markdown-exec: hide
Path("outputs/surface").mkdir(parents=True, exist_ok=True)  # markdown-exec: hide
# Embed the completed Matplotlib Figure during the docs build.  # markdown-exec: hide


def _embed_static(figure):  # markdown-exec: hide
    image = io.BytesIO()  # markdown-exec: hide
    figure.savefig(image, format="png", dpi=300)  # markdown-exec: hide
    plt.close(figure)  # markdown-exec: hide
    encoded = base64.b64encode(image.getvalue()).decode("ascii")  # markdown-exec: hide
    print(  # markdown-exec: hide
        f'<img alt="{_stage}" src="data:image/png;base64,{encoded}" style="width:100%;max-width:1000px;height:auto">'  # markdown-exec: hide
    )  # markdown-exec: hide


import numpy as np

from anatplot import SurfacePlot, datasets

surfaces = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
sulc = datasets.fetch_sulcal_depth("fsLR", density="32k")
```

## 2. Add the map

Start with the Neurosynth association map for the “default mode” term,
using sulcal shading to make the cortical folds visible.

```python exec="on" source="above" html="on" session="surface-tutorial" id="surface-step-2"
left_map, right_map = datasets.fetch_dmn_map_fslr32k()
# Mask this map's zero background to leave the shading underneath visible.
association = {
    "left": np.ma.masked_equal(left_map, 0),
    "right": np.ma.masked_equal(right_map, 0),
}
plot = SurfacePlot(
    **surfaces,
    views=["lateral", "medial"],
    window_size=(1000, 600),
    zoom=1.5,
)
plot.add_layer(sulc, cmap="gray")  # Sulcal shading beneath the map.
# Give the map its own labelled colorbar.
plot.add_layer(association, cmap="YlOrRd", cbar=True, cbar_label="Association Z")
_stage = "Association map with sulcal shading on lateral and medial fsLR surfaces"  # markdown-exec: hide
fig = plot.to_figure(figsize=(10, 7))
_embed_static(fig)  # markdown-exec: hide
```

## 3. Add contacts to this map

Next, add four contacts to the map, using color to distinguish the two pairs.

```python exec="on" source="above" html="on" session="surface-tutorial" id="surface-step-3"
# Coordinates in the template's RAS millimetre frame.
xyz = [
    [-64, -13, 23],
    [-60, -42, 37],
    [66, -14, 22],
    [64, -37, 32],
]
plot.add_contacts(
    xyz,
    radius=4.0,
    color=["royalblue", "gold", "royalblue", "gold"],
)
_stage = "Association map and four contacts on lateral and medial fsLR surfaces"  # markdown-exec: hide
fig = plot.to_figure(figsize=(10, 7))
_embed_static(fig)  # markdown-exec: hide
```

!!! note "Contacts on inflated surfaces"

    Contacts must share the original pial or midthickness surface's RAS
    millimetre frame. The `projection_surface` pair returned by `fetch_surfaces()`
    lets `SurfacePlot` locate contacts on that anatomy and transfer them to the
    inflated display. Projection removes depth and skips contacts farther than
    6 mm by default; see [contact projection](../use-your-own-data.md#contacts-on-an-inflated-map).

## 4. Save the completed figure

Save the completed figure, then close the scene to release graphics resources.

```python exec="on" source="above" html="on" session="surface-tutorial" id="surface-step-4" workdir="outputs/surface"
fig.savefig("cortex.png", dpi=300)
plot.close()
```

For 3D viewing and HTML export, see [Save and display](../save-and-display.md).
To substitute your own maps or contacts, see [Use your own data](../use-your-own-data.md).

## Alternative: show atlas labels

To show regions instead of a continuous map, use the Schaefer 100-region,
seven-network atlas on the same fsLR 32k surfaces. `add_parcellation()` assigns
colors to labels without creating a colorbar.

=== "All parcels"

    ```python exec="on" source="above" html="on" session="surface-tutorial" id="surface-atlas-all"
    parcellation = datasets.fetch_parcellation("schaefer")
    # Hide the atlas background while preserving the original parcel IDs.
    labels = {
        "left": np.ma.masked_equal(parcellation["left"], 0),
        "right": np.ma.masked_equal(parcellation["right"], 0),
    }
    # A with block releases the scene's graphics resources when it ends.
    with SurfacePlot(
        **surfaces,
        views=["lateral", "medial"],
        window_size=(1000, 600),
        zoom=1.5,
    ) as atlas_plot:
        atlas_plot.add_layer(sulc, cmap="gray")
        atlas_plot.add_parcellation(labels)
        atlas_fig = atlas_plot.to_figure(figsize=(10, 7))
    _stage = "Schaefer atlas labels on both fsLR hemispheres"  # markdown-exec: hide
    _embed_static(atlas_fig)  # markdown-exec: hide
    ```

=== "Selected parcels"

    Highlight left-hemisphere parcels 17 and 18 while leaving the rest of the
    cortex visible.

    ```python exec="on" source="above" html="on" session="surface-tutorial" id="surface-atlas-selected"
    parcellation = datasets.fetch_parcellation("schaefer")
    left_labels = parcellation["left"]
    # Keep the selected IDs and hide all other parcels with NaN.
    selected = np.where(np.isin(left_labels, [17, 18]), left_labels, np.nan)
    with SurfacePlot(
        **surfaces,
        views=["lateral", "medial"],
        window_size=(1000, 600),
        zoom=1.5,
    ) as atlas_plot:
        atlas_plot.add_layer(sulc, cmap="gray")
        atlas_plot.add_parcellation(
            {"left": selected},
            cmap={17: "crimson", 18: "gold"},
        )
        atlas_fig = atlas_plot.to_figure(figsize=(10, 7))
    _stage = "Two left-hemisphere Schaefer parcels"  # markdown-exec: hide
    _embed_static(atlas_fig)  # markdown-exec: hide
    ```

Parcel names are available from the atlas label table, for example
`parcellation["labels"]["left"][17]["name"]`.

## Brain size and spacing

If the brains in either recipe need more or less room, adjust `zoom`. Within
one `SurfacePlot`, `layout` arranges the views; there is no separate gap
setting.

| Change | Effect |
| --- | --- |
| Increase `zoom` | Larger brains, smaller visible gaps |
| Decrease `zoom` | Smaller brains, larger visible gaps |

Panel positions stay fixed. Excessive zoom can crop a brain, and the
`window_size` width-to-height ratio also affects space available per view.
