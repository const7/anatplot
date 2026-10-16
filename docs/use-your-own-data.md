# Use your own data

Use your own surface maps, contact coordinates, or label images with the
recipes below. Files combined in one scene must share a coordinate space;
anatplot does not register or resample them. [Datasets](datasets.md) lists
compatible public surfaces and atlases.

## Draw a cortical map

For a cortical map, supply one value per vertex of each display mesh. Match
the arrays and meshes by template, density, and **exact vertex order** within
each hemisphere; equal lengths do not establish correspondence. The recipes
below read `.npy` arrays and GIFTI meshes. anatplot does not apply embedded
GIFTI coordinate transforms, so check the source geometry before using them.

=== "Map only"

    ```python
    import numpy as np

    from anatplot import SurfacePlot

    left_values = np.load("left-values.npy")
    right_values = np.load("right-values.npy")

    with SurfacePlot(
        left="left.inflated.surf.gii",
        right="right.inflated.surf.gii",
    ) as plot:
        plot.add_layer({"left": left_values, "right": right_values}, cbar=True)
        fig = plot.to_figure()

    fig.savefig("my-map.png", dpi=300)
    ```

=== "Map and contacts"

    For contacts on an inflated map, `projection_surface` locates contacts on
    anatomical pial or midthickness meshes before displaying them on the
    inflated meshes. Each anatomical mesh must correspond vertex by vertex to
    its display mesh, with identical ordered triangles.

    Verify that contacts and anatomical meshes share one RAS millimetre frame.
    Each `xyz` row is `[R, A, S]`, with right, anterior, and superior positive;
    record the source transform and units. A shared label such as “MNI” does
    not establish alignment.

    ```python
    import numpy as np

    from anatplot import SurfacePlot

    left_values = np.load("left-values.npy")
    right_values = np.load("right-values.npy")
    xyz = np.load("contacts-ras-mm.npy")

    with SurfacePlot(
        left="left.inflated.surf.gii",
        right="right.inflated.surf.gii",
        projection_surface=(
            "left.midthickness.surf.gii",
            "right.midthickness.surf.gii",
        ),
    ) as plot:
        plot.add_layer({"left": left_values, "right": right_values}, cbar=True)
        plot.add_contacts(xyz, color="royalblue", radius=2)
        fig = plot.to_figure()

    fig.savefig("my-map-and-contacts.png", dpi=300)
    ```

Missing values (`NaN` or masked vertices) leave the surface underneath visible.
Zero remains visible; mask it only if it means background in your data.

When comparing the same statistic across plots, pass the same `color_range` to
each `add_layer()` call.

For discrete vertex labels, use
`plot.add_parcellation({"left": left_labels, "right": right_labels})` and mask a
background label if needed. Vertex colors blend across triangles, so their
edges are not exact parcel boundaries. The [surface tutorial](tutorials/surfaceplot-basics.md)
shows an atlas example.

### Contacts on an inflated map

With the anatomical `projection_surface` pair shown above, `SurfacePlot`
locates each contact on the nearest anatomical triangle and displays its marker
on the corresponding inflated triangle. Without that pair, it projects onto
the display meshes themselves, which does not recover anatomical distances
when the display is inflated.

Projection discards depth and does not establish cortical membership. Contacts
more than 6 mm from the localization surface are skipped with a warning by
default.

For deep contacts, `ROIPlot` or `VolumePlot` retains their supplied positions
when their anatomy shares the coordinate frame. On `SurfacePlot`,
`project=False` also keeps supplied coordinates and skips the distance filter.
It does not move contacts onto an inflated display surface, so use it with an
undeformed surface when you need their original positions. The
[surface tutorial](tutorials/surfaceplot-basics.md#3-add-contacts-to-this-map) uses a
matching pair supplied by the dataset fetcher.

## Show contacts within a whole-brain surface

`VolumePlot` gives contacts a translucent whole-brain surface background. It
displays surface meshes, not image voxels, and keeps contact positions as
supplied. Use anatomical pial meshes and contacts whose source records confirm
the same RAS millimetre frame:

```python
import numpy as np

from anatplot import VolumePlot

xyz = np.load("contacts-ras-mm.npy")

with VolumePlot(left="left.pial.surf.gii", right="right.pial.surf.gii") as plot:
    plot.add_contacts(xyz, radius=2, color="royalblue")
    fig = plot.to_figure()

fig.savefig("my-contacts.png", dpi=300)
```

Use a template surface only when its registration to your contacts is confirmed;
use subject-specific pial surfaces for contacts in subject space. If contacts
fall outside the initial camera frame, call
`plot.plotter.reset_camera()` before `to_figure()`. See the
[VolumePlot tutorial](tutorials/volumeplot.md) for a complete example.

## Draw regions from a label image

`ROIPlot` accepts a 3D NIfTI image whose **entire data array** contains finite,
nonnegative integers. Zero is background; choose positive IDs present in the
image. The NIfTI header must declare spatial units of `mm`, `meter`, or
`micron`—unknown units are rejected. Its affine and units place extracted
regions in RAS millimetres. This image differs from the one-dimensional
surface vertex labels used by `add_parcellation()`.

```python
import nibabel as nib
import numpy as np

from anatplot import ROIPlot

image = nib.load("my-labels.nii.gz")
ids = np.unique(np.asanyarray(image.dataobj))
print(ids[ids > 0])  # Choose positive IDs present in your image.

with ROIPlot(image) as plot:
    plot.add_roi([1, 2])  # Replace with IDs from the printed list.
    fig = plot.to_figure()

fig.savefig("my-regions.png", dpi=300)
```

Use the source atlas's label table to map IDs to anatomical names. Selected
IDs in one `add_roi()` call form a union without internal borders. If the
affine contains shear, use `add_roi(..., smoothing=False)` or resample the image
beforehand. Inspect thin regions after extraction. The [ROI tutorial](tutorials/roiplot.md)
shows how to find IDs by region name.

For figure size, 3D viewing, HTML, and camera choices, see
[Save and display](save-and-display.md). Full shapes and options are in the
[plotting API](api/plotting.md).
