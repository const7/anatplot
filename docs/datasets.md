# Datasets

Use these fetchers to obtain public display and anatomical surfaces, sulcal
shading, surface parcellations, and an example cortical map. Missing files
download on first use and are cached for later calls.

A surface supplies vertex positions and triangles; a parcellation supplies one
label per vertex. Match their template, density, and vertex order. Density is
mesh resolution, separate from the number of regions. See
[Use your own data](use-your-own-data.md) when matching your own maps and contacts to anatomy.

## Fetch matching surfaces

```python
from anatplot import datasets

surfaces = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
```

The result contains `left` and `right` display surfaces plus a corresponding
anatomical `projection_surface` pair. Pass the complete result to `SurfacePlot`
when showing contacts on an inflated surface. `VolumePlot` uses only `left` and
`right` from an anatomical fetch such as `surface="pial"` on fsaverage.

| Template | Available densities | Default density | Surface geometries |
| --- | --- | --- | --- |
| `fsaverage` | 3k, 10k, 41k, 164k | 10k | pial, white, inflated, sphere |
| `fsLR` | 4k, 8k, 32k, 164k | 32k | midthickness, inflated, veryinflated, sphere |
| `civet` | 41k, 164k | 41k | white, midthickness, inflated, veryinflated, sphere |

Density is the approximate vertex count per hemisphere. Inflated geometry
exposes folds; pial and midthickness retain anatomical shape. Sphere is
registration geometry. fsLR 4k/8k lack veryinflated surfaces.

## Add sulcal shading

```python
sulc = datasets.fetch_sulcal_depth("fsLR", density="32k")
```

The left/right arrays can be passed to `plot.add_layer(sulc, cmap="gray")`.
Use the same template and density as the surface.

## Choose a parcellation

`fetch_parcellation()` returns labels in a published surface space:

| Name | Regions or networks | fsLR density | fsaverage densities |
| --- | --- | --- | --- |
| `schaefer` | 100–1000 regions in steps of 100; 7- or 17-network ordering | 32k | 10k, 41k, 164k |
| `glasser` | 360 regions (HCP-MMP1.0) | 32k | — |
| `desikan` | Desikan–Killiany anatomical regions | — | 3k, 10k, 41k |
| `destrieux` | Destrieux 2009 anatomical regions | — | 3k, 10k, 41k, 164k |
| `yeo` | 7 or 17 functional networks | — | 10k, 41k, 164k |

A dash means unavailable. These parcellations do not have CIVET versions.

For example, fetch 400 Schaefer regions with seven-network ordering:

```python
atlas = datasets.fetch_parcellation(
    "schaefer", template="fsLR", density="32k", n_regions=400, networks=7
)
left_labels, right_labels = atlas["left"], atlas["right"]
```

Schaefer defaults to 100 regions and seven networks. Yeo uses `n_regions=7`
or `17`. See `datasets.available_parcellations()` for supported combinations.

## Find region names

Each hemisphere has its own label table. For left-hemisphere label 17:

```python
region = atlas["labels"]["left"][17]
print(region["name"])
```

IDs are specific to the atlas version and hemisphere; use the returned table,
not an ID alone, to identify a region.

The table also contains source colors, but `add_parcellation()` uses the
`tab20` palette by default and does not apply those colors automatically.
Mask zero when it denotes background. The [surface tutorial](tutorials/surfaceplot-basics.md)
shows how to highlight regions.

## Default-mode map and label volumes

`fetch_dmn_map_fslr32k()` returns the Neurosynth association map for the
“default mode” term, preprojected to fsLR 32k.

For label volumes, use [Nilearn](https://nilearn.github.io/stable/modules/datasets.html),
as in the [ROI tutorial](tutorials/roiplot.md). See the [dataset API](api/datasets.md)
for cache locations, return fields, and source metadata.

## Provenance

| Data | Source and citation |
| --- | --- |
| Cortical templates | [Template sources](https://netneurolab.github.io/neuromaps/user_guide/atlases.html); [neuromaps, Markello et al., 2022](https://doi.org/10.1038/s41592-022-01625-w) |
| Schaefer atlas | [CBIG parcellations](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations), [Schaefer et al., 2018](https://doi.org/10.1093/cercor/bhx179) |
| Glasser | [NetNeuroLab archive](https://osf.io/udpv8/), [Glasser et al., 2016](https://doi.org/10.1038/nature18933) |
| Desikan, Destrieux, Yeo | [TemplateFlow fsaverage archive](https://github.com/templateflow/tpl-fsaverage); original paper citations are returned under `source` |
| Default-mode map | [Neurosynth](https://doi.org/10.1038/nmeth.1635); fsLR projection distributed by [surfplot](https://surfplot.readthedocs.io/en/stable/generated/surfplot.datasets.load_example_data.html) |
| Harvard-Oxford atlas | [FSL atlas documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html), including its associated publications; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) under the [FSL atlas terms](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html) |

Cite the datasets used in your figures and follow their source licenses.
