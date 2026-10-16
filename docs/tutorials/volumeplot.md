# VolumePlot: Whole-brain contacts

Use `VolumePlot` to see contacts in the context of the whole brain. Here,
translucent fsaverage pial surfaces provide the anatomical background.

## 1. Draw the background

Fetch both hemispheres to build the translucent background. `VolumePlot`
draws these surfaces, rather than image voxels.

```python exec="on" source="above" html="on" session="volume-tutorial" id="volume-step-1"
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
Path("outputs/volume").mkdir(parents=True, exist_ok=True)  # markdown-exec: hide
# Embed the completed Matplotlib Figure during the docs build.  # markdown-exec: hide


def _embed_static(figure):  # markdown-exec: hide
    image = io.BytesIO()  # markdown-exec: hide
    figure.savefig(image, format="png", dpi=200)  # markdown-exec: hide
    plt.close(figure)  # markdown-exec: hide
    encoded = base64.b64encode(image.getvalue()).decode("ascii")  # markdown-exec: hide
    print(  # markdown-exec: hide
        f'<img alt="{_stage}" src="data:image/png;base64,{encoded}" style="width:100%;max-width:900px;height:auto">'  # markdown-exec: hide
    )  # markdown-exec: hide


from anatplot import VolumePlot, datasets

surfaces = datasets.fetch_surfaces(
    template="fsaverage",
    surface="pial",
    density="10k",
)

plot = VolumePlot(
    left=surfaces["left"],
    right=surfaces["right"],
)
_stage = "Translucent fsaverage pial surfaces as a whole-brain background"  # markdown-exec: hide
fig = plot.to_figure()
_embed_static(fig)  # markdown-exec: hide
```

## 2. Add contacts

Next, add contacts on both sides of the brain, using red and blue to
distinguish the two groups.

```python exec="on" source="above" html="on" session="volume-tutorial" id="volume-step-2"
# Coordinates in the fsaverage surface's RAS millimetre frame.
xyz = [
    [-28, -18, -12],
    [-28, -22, -12],
    [-28, -26, -12],
    [28, -18, -12],
    [28, -22, -12],
    [28, -26, -12],
]
plot.add_contacts(
    xyz,
    radius=2.0,
    color=["crimson", "crimson", "crimson", "royalblue", "royalblue", "royalblue"],
)
_stage = "Translucent whole-brain surfaces with six contacts"  # markdown-exec: hide
fig = plot.to_figure()
_embed_static(fig)  # markdown-exec: hide
```

`VolumePlot` keeps the supplied coordinates rather than projecting them onto
the surface. If contacts lie outside the view, call
`plot.plotter.reset_camera()` before `to_figure()` to fit the whole scene.

## 3. Save and share

Finally, save the contact figure as a PNG and export the scene as interactive HTML.

```python exec="on" source="above" html="on" session="volume-tutorial" id="volume-step-3" workdir="outputs/volume"
fig.savefig("brain.png", dpi=300)
plot.export_html("brain.html")
scene = Path("brain.html").read_text(encoding="utf-8")  # markdown-exec: hide
print(  # markdown-exec: hide
    f'<iframe title="Interactive whole-brain background" srcdoc="{escape(scene, quote=True)}" style="width:100%;height:520px;border:0" loading="lazy"></iframe>'  # markdown-exec: hide
)  # markdown-exec: hide
plot.close()
```
