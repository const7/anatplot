"""Render the three-plot overview from the public tutorial datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import fetch_atlas_harvard_oxford

from anatplot import ROIPlot, SurfacePlot, VolumePlot, datasets


def main() -> None:
    """Regenerate the three-plot overview.

    Returns
    -------
    None
        Downloads missing public data and writes plot-types.png to
        docs/assets/images, closing scenes after capture.
    """
    plt.switch_backend("Agg")
    destination = Path(__file__).resolve().parents[1] / "assets/images"
    surfaces = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
    left, right = datasets.fetch_dmn_map_fslr32k()
    association = {
        "left": np.ma.masked_equal(left, 0),
        "right": np.ma.masked_equal(right, 0),
    }

    with SurfacePlot(
        **surfaces,
        views=["lateral", "medial"],
        layout="grid",
        window_size=(1000, 600),
        zoom=1.5,
    ) as plot:
        plot.add_layer(
            association, cmap="YlOrRd", cbar=True, cbar_label="Association Z"
        )
        plot.add_contacts([[-64, -13, 23]], radius=4, color="royalblue")
        # Size typography for the overview's displayed width.
        with plt.rc_context({"font.size": 15}):
            overview, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
            for axis, title in zip(axes, ("SurfacePlot", "ROIPlot", "VolumePlot")):
                axis.set_axis_off()
                axis.set_title(title, fontsize=18)
            plot.to_figure(ax=axes[0])
            overview.axes[-1].set_xticks([0, 6, 12])

    atlas = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
    labels_by_name = atlas.lut.set_index("name")["index"]
    regions = {
        "Left Hippocampus": "#D98C70",
        "Right Hippocampus": "#D98C70",
        "Left Amygdala": "#8EAFBB",
        "Right Amygdala": "#8EAFBB",
    }
    with ROIPlot(atlas.maps) as plot:
        for name, color in regions.items():
            plot.add_roi([int(labels_by_name.loc[name])], color=color)
        plot.plotter.reset_camera()
        plot.add_contacts(
            [[-26, -10, -28], [26, -10, -22], [-22, -10, -12], [24, -10, -12]],
            radius=1,
            color="crimson",
        )
        plot.plotter.view_vector((0, 1, 3), viewup=(0, 1, 0))
        plot.plotter.camera.parallel_scale = 32
        plot.to_figure(ax=axes[1])

    pial = datasets.fetch_surfaces("fsaverage", surface="pial", density="10k")
    with VolumePlot(left=pial["left"], right=pial["right"]) as plot:
        plot.add_contacts(
            [[x, y, -12] for x in (-28, 28) for y in (-18, -22, -26)],
            radius=2,
            color=["crimson"] * 3 + ["royalblue"] * 3,
        )
        plot.to_figure(ax=axes[2])

    overview.savefig(destination / "plot-types.png", dpi=300)
    plt.close(overview)


if __name__ == "__main__":
    main()
