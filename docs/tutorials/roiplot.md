# ROIPlot: Plot anatomical regions

Use `ROIPlot` to show selected regions from a labelled anatomical image, with
optional contacts in the same scene. This example selects the left and right
hippocampus and amygdala from a public atlas, then chooses a view for the
finished figure.

## 1. Select and draw regions

Load the [Harvard–Oxford label atlas](../datasets.md#provenance) with Nilearn.
Its lookup table connects region names to the integer labels in the image,
so the example can select regions by name and pass their IDs to `add_roi()`.

```python exec="on" source="above" html="on" session="roi-tutorial" id="roi-step-1"
import base64  # markdown-exec: hide
import io  # markdown-exec: hide
from pathlib import Path  # markdown-exec: hide
import matplotlib  # markdown-exec: hide

matplotlib.use("Agg")  # markdown-exec: hide
import matplotlib.pyplot as plt  # markdown-exec: hide
from html import escape  # markdown-exec: hide
import pyvista as pv  # markdown-exec: hide
from trame.app import get_server  # markdown-exec: hide

server = get_server(pv.global_theme.trame.jupyter_server_name)  # markdown-exec: hide
server.cli.set_defaults(nosignalhandlers=True)  # markdown-exec: hide
Path("outputs/roi").mkdir(parents=True, exist_ok=True)  # markdown-exec: hide
# Embed the completed Matplotlib Figure during the docs build.  # markdown-exec: hide


def _embed_static(figure):  # markdown-exec: hide
    image = io.BytesIO()  # markdown-exec: hide
    figure.savefig(image, format="png", dpi=200)  # markdown-exec: hide
    plt.close(figure)  # markdown-exec: hide
    encoded = base64.b64encode(image.getvalue()).decode("ascii")  # markdown-exec: hide
    print(  # markdown-exec: hide
        f'<img alt="{_stage}" src="data:image/png;base64,{encoded}" style="width:100%;max-width:900px;height:auto">'  # markdown-exec: hide
    )  # markdown-exec: hide


from nilearn.datasets import fetch_atlas_harvard_oxford

from anatplot import ROIPlot

# Fetch the label volume and its region lookup table.
atlas = fetch_atlas_harvard_oxford(
    "sub-maxprob-thr25-2mm",
    verbose=0,
)
labels_by_name = atlas.lut.set_index("name")["index"]

# Choose regions by name and assign their display colors.
regions = {
    "Left Hippocampus": "#D98C70",
    "Right Hippocampus": "#D98C70",
    "Left Amygdala": "#8EAFBB",
    "Right Amygdala": "#8EAFBB",
}
# Extract each region and add it to the shared anatomical scene.
plot = ROIPlot(atlas.maps)
for name, color in regions.items():
    label_id = int(labels_by_name.loc[name])
    plot.add_roi(labels=[label_id], color=color)

plot.plotter.reset_camera()  # Fit all four regions in the view.
_stage = "Bilateral hippocampus and amygdala"  # markdown-exec: hide
fig = plot.to_figure()
_embed_static(fig)  # markdown-exec: hide
```

## 2. Add contacts

Next, add contacts within the selected regions. Their coordinates must use
the atlas's RAS millimetre frame; `ROIPlot` keeps their supplied positions
without testing region membership.

```python exec="on" source="above" html="on" session="roi-tutorial" id="roi-step-2"
# Coordinates inside the four regions, in the same order.
xyz = [
    [-26, -10, -28],
    [26, -10, -22],
    [-22, -10, -12],
    [24, -10, -12],
]
plot.add_contacts(xyz, radius=1.0, color="crimson")
_stage = "Hippocampus and amygdala with four contacts"  # markdown-exec: hide
fig = plot.to_figure()
_embed_static(fig)  # markdown-exec: hide
```

## 3. Choose the view

With regions and contacts in place, look from above and slightly in front to
show both hippocampi along their length, with the amygdalae at the top.
Anatomical left appears on the left.

```python exec="on" source="above" html="on" session="roi-tutorial" id="roi-step-3"
# Look from anterior-superior, with anterior toward the top of the image.
plot.plotter.view_vector((0, 1, 3), viewup=(0, 1, 0))
plot.plotter.camera.parallel_scale = 32
_stage = "Hippocampus and amygdala with contacts from an anterior superior view"  # markdown-exec: hide
fig = plot.to_figure()
_embed_static(fig)  # markdown-exec: hide
```

| Camera setting | What it does |
| --- | --- |
| `view_vector((0, 1, 3))` | Looks from anterior and superior |
| `viewup=(0, 1, 0)` | Places anterior toward the image top |
| `parallel_scale = 32` | Shows a 64 mm image height; smaller values enlarge the regions |

Use `plot.view()` to explore other directions. For a reproducible final
figure, set the camera in Python before `to_figure()` and keep those settings
in your code.

## 4. Save and share

Finally, save the chosen view as a PNG and export the scene as interactive HTML.

```python exec="on" source="above" html="on" session="roi-tutorial" id="roi-step-4" workdir="outputs/roi"
fig.savefig("roi.png", dpi=300)
plot.export_html("roi.html")
scene = Path("roi.html").read_text(encoding="utf-8")  # markdown-exec: hide
print(  # markdown-exec: hide
    f'<iframe title="Interactive hippocampus and amygdala" srcdoc="{escape(scene, quote=True)}" style="width:100%;height:520px;border:0" loading="lazy"></iframe>'  # markdown-exec: hide
)  # markdown-exec: hide
plot.close()
```
