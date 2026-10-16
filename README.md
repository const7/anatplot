# anatplot

anatplot is a Python package for plotting data on brain surfaces and
visualizing anatomical regions in 3D. You can optionally add electrode
contacts to any plot, then save a Matplotlib figure or explore and export
the 3D scene.

> This project is under active development; some features may need further refinement.

Install the current development release from PyPI with Python 3.10 or newer:

```bash
uv add "anatplot==0.1.0.dev0"
# Or with pip:
python -m pip install "anatplot==0.1.0.dev0"
```

![Examples of SurfacePlot, ROIPlot, and VolumePlot](https://raw.githubusercontent.com/const7/anatplot/main/docs/assets/images/plot-types.png)

## Quick start

Plot a public Neurosynth default-mode association map and an example electrode
contact on inflated cortical surfaces. Dataset helpers download and cache the
public data on first use.

```python
import matplotlib.pyplot as plt
import numpy as np

from anatplot import SurfacePlot, datasets

# Fetch surfaces and a map with matching fsLR 32k vertex order.
surfaces = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
left, right = datasets.fetch_dmn_map_fslr32k()
association = {
    # Hide this map's zero background so the underlying surface stays visible.
    "left": np.ma.masked_equal(left, 0),
    "right": np.ma.masked_equal(right, 0),
}

# The with block closes the 3D scene and releases graphics resources.
with SurfacePlot(
    **surfaces,
    views=["lateral", "medial"],
    layout="grid",
    window_size=(1000, 600),
    zoom=1.5,
) as plot:
    plot.add_layer(association, cmap="YlOrRd", cbar=True, cbar_label="Association Z")
    # Use the original anatomy's RAS coordinates in millimetres;
    # the fetched projection surfaces transfer contacts to the inflated display.
    plot.add_contacts([[-64, -13, 23]], radius=4, color="royalblue")
    fig = plot.to_figure(figsize=(10, 7))

fig.savefig("surface-map.png", dpi=300)
plt.close(fig)
```

See the [documentation](https://chenkun.me/anatplot/) for more.

## Acknowledgements

`SurfacePlot` was inspired by [surfplot](https://github.com/danjgale/surfplot).
Public templates and atlases come from [neuromaps](https://netneurolab.github.io/neuromaps/),
[TemplateFlow](https://www.templateflow.org/), [Nilearn](https://nilearn.github.io/),
and [NetNeuroLab](https://osf.io/udpv8/). Cite the data used in your figure;
see [dataset sources](https://chenkun.me/anatplot/datasets/#provenance).

## Contributing

Issues and pull requests are welcome. Please see the
[contributing guide](https://chenkun.me/anatplot/development/contributing/) before getting started.
