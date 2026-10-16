"""Standard anatomy and attributed surface maps with persistent caches."""

from collections.abc import Sequence
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import TypeAlias, TypedDict

import nibabel as nib
import numpy as np
import pooch
from neuromaps.datasets import fetch_atlas
from neuromaps.images import dlabel_to_gifti
from numpy.typing import NDArray


class _SurfacePaths(TypedDict):
    """Left/right display paths and the matching anatomical projection pair."""

    left: Path
    right: Path
    projection_surface: tuple[Path, Path]


_LabelTable: TypeAlias = dict[int, dict[str, str | tuple[float, float, float, float]]]


class _ParcellationChoice(TypedDict):
    """Published template/density choices, region counts, provider, and citation."""

    templates: dict[str, tuple[str, ...]]
    default: tuple[str, str]
    n_regions: tuple[int, ...]
    networks: tuple[int, ...]
    provider: str
    citation: str


class _Parcellation(TypedDict):
    """Hemisphere vertex labels, original label tables, selection, and provenance."""

    left: NDArray[np.int32]
    right: NDArray[np.int32]
    labels: dict[str, _LabelTable]
    name: str
    template: str
    density: str
    n_regions: int | None
    networks: int | None
    source: dict[str, str | tuple[Path, ...] | tuple[str, ...]]


_TEMPLATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "fsaverage": ("10k", ("white", "pial", "inflated", "sphere")),
    "fsLR": ("32k", ("midthickness", "inflated", "veryinflated", "sphere")),
    "civet": ("41k", ("white", "midthickness", "inflated", "veryinflated", "sphere")),
}
_VERTEX_COUNTS = {
    "3k": 2562,
    "4k": 4002,
    "8k": 7842,
    "10k": 10242,
    "32k": 32492,
    "41k": 40962,
    "164k": 163842,
}


def _fetch_template(
    template: str,
    density: str,
    data_dir: str | Path | None,
) -> dict[str, tuple[Path, Path]]:
    """Fetch one neuromaps template archive using the shared surface-data cache.

    Parameters
    ----------
    template : str
        Validated cortical template name from _TEMPLATES.
    density : str
        Requested vertex density. neuromaps validates the available densities.
    data_dir : str | pathlib.Path | None
        Cache root. None uses ~/.cache/anatplot/neuromaps outside the checkout.

    Returns
    -------
    atlas : dict[str, tuple[pathlib.Path, pathlib.Path]]
        Provider paths keyed by geometry or auxiliary map, in left/right order.
        neuromaps downloads missing files and reuses existing ones. Paths refer
        to cached source files; no file content is modified or copied.
    """
    cache = (
        Path.home() / ".cache" / "anatplot" / "neuromaps"
        if data_dir is None
        else Path(data_dir)
    )
    return fetch_atlas(template, density, data_dir=cache, verbose=0)


def fetch_surfaces(
    template: str = "fsaverage",
    *,
    surface: str = "inflated",
    density: str | None = None,
    data_dir: str | Path | None = None,
) -> _SurfacePaths:
    """Fetch a standard left/right surface pair ready for a plotting constructor.

    Parameters
    ----------
    template : str, default="fsaverage"
        Surface template: "fsaverage", "fsLR", or "civet". This chooses actual
        data, not a declaration of registration with another image or subject.
    surface : str, default="inflated"
        Geometry to select. All templates provide inflated and sphere surfaces.
        fsaverage also provides pial and white; fsLR provides midthickness;
        CIVET provides white and midthickness. fsLR and CIVET provide
        veryinflated, except fsLR at 4k/8k. Use anatomical geometry such as pial
        for a whole-brain background; sphere is registration geometry.
    density : str | None, default=None
        Approximate vertices per hemisphere. None selects 10k for fsaverage,
        32k for fsLR, or 41k for CIVET. Supported densities are fsaverage:
        3k/10k/41k/164k; fsLR: 4k/8k/32k/164k; CIVET: 41k/164k.
        Density validation is delegated to neuromaps.
    data_dir : str | pathlib.Path | None, default=None
        Persistent dataset directory. None uses ~/.cache/anatplot/neuromaps,
        outside the checkout. neuromaps downloads missing files and reuses
        existing files. The first call requires network access and fetches the
        template/density archive, including other geometries and auxiliary maps.

    Returns
    -------
    surfaces : dict
        Display GIFTI paths under "left" and "right", plus "projection_surface",
        a (left, right) path tuple for anatomical contact localization. Pass
        the complete dictionary to SurfacePlot; VolumePlot uses only left and
        right. SurfacePlot uses projection_surface when adding contacts.
        Localization always prefers pial when the template provides it:
        fsaverage uses pial; fsLR/CIVET provide no pial and use midthickness.
        This choice is independent of display geometry. No geometry is changed.

    Raises
    ------
    ValueError
        If the template, surface, or density is unsupported. Download errors
        propagate from neuromaps; no substitute template is silently selected.

    Notes
    -----
    Data acquisition, integrity checks, extraction, and caching belong to
    neuromaps. No Workbench installation is needed to fetch and display these
    surfaces. Cite the template source and neuromaps when using downloaded data.

    Examples
    --------
    ```python
    from anatplot import SurfacePlot, datasets

    surfaces = datasets.fetch_surfaces("fsLR", surface="inflated", density="32k")
    plot = SurfacePlot(**surfaces)
    fig = plot.to_figure()
    fig.savefig("cortex.png")
    plot.close()

    ```
    """
    if template not in _TEMPLATES:
        raise ValueError(f"template must be one of {tuple(_TEMPLATES)}")
    default_density, kinds = _TEMPLATES[template]
    density = default_density if density is None else density
    if surface not in kinds or (
        template == "fsLR" and density in ("4k", "8k") and surface == "veryinflated"
    ):
        raise ValueError(f"surface {surface!r} is unavailable for {template} {density}")
    atlas = _fetch_template(template, density, data_dir)
    left, right = atlas[surface]
    reference = "pial" if "pial" in kinds else "midthickness"
    source_left, source_right = atlas[reference]
    return {
        "left": Path(left),
        "right": Path(right),
        "projection_surface": (Path(source_left), Path(source_right)),
    }


def fetch_sulcal_depth(
    template: str = "fsaverage",
    *,
    density: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Fetch sulcal-depth vertex arrays for a standard cortical template.

    Parameters
    ----------
    template : str, default="fsaverage"
        Surface template: "fsaverage", "fsLR", or "civet". Use the same template
        and density as fetch_surfaces to retain source vertex correspondence.
    density : str | None, default=None
        Approximate vertices per hemisphere. None selects 10k for fsaverage,
        32k for fsLR, or 41k for CIVET. Supported densities match fetch_surfaces;
        neuromaps validates the available choices.
    data_dir : str | pathlib.Path | None, default=None
        Same template cache root as fetch_surfaces. None uses
        ~/.cache/anatplot/neuromaps. The first call downloads the template archive
        if needed; later surface and sulcal-depth requests reuse those files.

    Returns
    -------
    values : dict[str, numpy.ndarray]
        Independent float64 arrays under "left" and "right", each with shape
        (n_vertices,) in the selected template's full hemisphere vertex order.
        Pass the dictionary directly to SurfacePlot.add_layer. Source values,
        signs, and units are retained without normalization or resampling.

    Raises
    ------
    ValueError
        If the template or density is unsupported, or a source array has the
        wrong vertex count. Download errors propagate from neuromaps.

    Notes
    -----
    These maps provide an anatomical display background. Fetching them does not
    establish correspondence with an arbitrary user surface. Cite the template
    source and neuromaps when using downloaded data.
    """
    if template not in _TEMPLATES:
        raise ValueError(f"template must be one of {tuple(_TEMPLATES)}")
    density = _TEMPLATES[template][0] if density is None else density
    atlas = _fetch_template(template, density, data_dir)
    values = {
        side: np.array(nib.load(path).agg_data(), dtype=np.float64, copy=True)
        for side, path in zip(("left", "right"), atlas["sulc"])
    }
    for side, data in values.items():
        if data.shape != (_VERTEX_COUNTS[density],):
            raise ValueError(f"{side} sulcal depth does not match {template} {density}")
    return values


_PARCELLATIONS: dict[str, _ParcellationChoice] = {
    "schaefer": {
        "templates": {"fsaverage": ("10k", "41k", "164k"), "fsLR": ("32k",)},
        "default": ("fsLR", "32k"),
        "n_regions": tuple(range(100, 1001, 100)),
        "networks": (7, 17),
        "provider": "netneurotools",
        "citation": "https://doi.org/10.1093/cercor/bhx179",
    },
    "glasser": {
        "templates": {"fsLR": ("32k",)},
        "default": ("fsLR", "32k"),
        "n_regions": (360,),
        "networks": (),
        "provider": "netneurotools",
        "citation": "https://doi.org/10.1038/nature18933",
    },
    "desikan": {
        "templates": {"fsaverage": ("3k", "10k", "41k")},
        "default": ("fsaverage", "10k"),
        "n_regions": (),
        "networks": (),
        "provider": "templateflow",
        "citation": "https://doi.org/10.1016/j.neuroimage.2006.01.021",
    },
    "destrieux": {
        "templates": {"fsaverage": ("3k", "10k", "41k", "164k")},
        "default": ("fsaverage", "10k"),
        "n_regions": (),
        "networks": (),
        "provider": "templateflow",
        "citation": "https://doi.org/10.1016/j.neuroimage.2010.06.010",
    },
    "yeo": {
        "templates": {"fsaverage": ("10k", "41k", "164k")},
        "default": ("fsaverage", "10k"),
        "n_regions": (7, 17),
        "networks": (),
        "provider": "templateflow",
        "citation": "https://doi.org/10.1152/jn.00338.2011",
    },
}


def available_parcellations() -> dict[str, _ParcellationChoice]:
    """Return the supported published surface parcellation combinations.

    Returns
    -------
    choices : dict
        Keys are names accepted by fetch_parcellation. Each value contains
        templates mapped to density tuples, the default (template, density),
        selectable n_regions and networks, provider, and original citation.
        Empty option tuples mean the corresponding parameter must remain None.
        The returned dictionary can be edited without changing package defaults.

    Notes
    -----
    This lists existing provider files, not combinations generated by resampling.
    Calling this function performs no downloads.
    """
    return deepcopy(_PARCELLATIONS)


def _gifti_labels(
    image: nib.gifti.GiftiImage,
) -> tuple[NDArray[np.integer], _LabelTable]:
    """Read one hemisphere's GIFTI labels and original label table.

    Parameters
    ----------
    image : nibabel.gifti.GiftiImage
        Provider image with one integer label array in original vertex order.

    Returns
    -------
    data : numpy.ndarray
        Integer labels of shape (n_vertices,), retaining source order and IDs.
        The array may share storage with image and is not modified here.
    labels : dict[int, dict]
        Original ID-to-name and RGBA table. Each entry has a string name and
        a four-float color tuple with channels in [0, 1].
    """
    return image.agg_data(), {
        label.key: {"name": label.label, "color": label.rgba}
        for label in image.labeltable.labels
    }


def _read_parcellation(
    paths: Path | Sequence[str | Path],
) -> list[tuple[NDArray[np.integer], _LabelTable]]:
    """Read provider parcellation files without resampling their vertices.

    Parameters
    ----------
    paths : pathlib.Path | collections.abc.Sequence[str | pathlib.Path]
        One surface CIFTI dlabel path, or left/right GIFTI label or FreeSurfer
        annotation paths. A pair follows left-then-right hemisphere order.

    Returns
    -------
    hemispheres : list[tuple[numpy.ndarray, dict[int, dict]]]
        Label arrays and original label tables in file order, or CIFTI BrainModel
        order. Each integer array has shape (n_vertices,) for its hemisphere.
        CIFTI cortical indices are expanded to full vertex arrays with zero for
        absent vertices. FreeSurfer packed source IDs are retained.

    Notes
    -----
    Reads source files without rewriting them. NiBabel diagnostics from CIFTI
    loading remain visible to the caller.
    """
    if isinstance(paths, Path):
        # Surface CIFTI labels use BrainModel vertex indices, not voxel spacing.
        return [_gifti_labels(image) for image in dlabel_to_gifti(paths)]
    hemispheres: list[tuple[NDArray[np.integer], _LabelTable]] = []
    for path in paths:
        path = Path(path)
        if path.suffix == ".annot":
            # Packed FreeSurfer IDs retain the original label-table correspondence.
            data, colors, names = nib.freesurfer.read_annot(path, orig_ids=True)
            labels: _LabelTable = {
                int(row[4]): {
                    "name": name.decode("utf-8"),
                    "color": (*tuple(row[:3] / 255), 1 - row[3] / 255),
                }
                for row, name in zip(colors, names)
            }
            hemispheres.append((data, labels))
        else:
            hemispheres.append(_gifti_labels(nib.load(path)))
    return hemispheres


# Fixed public archives listed in NetNeuroLab's dataset manifest. Pooch owns
# download integrity, caching, and extraction; no netneurotools import is needed.
_PARCELLATION_MANIFEST = (
    "https://github.com/netneurolab/netneurotools/blob/"
    "3995ce898666f7667e66816a21b6e738868a62d2/"
    "netneurotools/datasets/datasets.json"
)
_PARCELLATION_ARCHIVES: dict[tuple[str, str], tuple[str, str]] = {
    ("schaefer", "10k"): (
        "673273c550aeaa53a10dded5",
        "d0fe0525b83553279d750e38f2d1dd6194a22c190371dbdf6778bdde414d23b3",
    ),
    ("schaefer", "41k"): (
        "673273c72e830a5cbd0dde84",
        "1abab92972e249579ac3e29da676d2a9a5394269042916d1815dc5618331b211",
    ),
    ("schaefer", "164k"): (
        "673273c54b79ea9a5062f4b3",
        "21b061d5948bddd652baef204380a7e8ec20c8aba20c0391aa1176f85d203adc",
    ),
    ("schaefer", "32k"): (
        "673273c790023be9c44d6b5a",
        "f3911adaa6266077e69f953483e9440813f2bc7967701c2e102c4660f21d1058",
    ),
    ("glasser", "32k"): (
        "6732741eb9a26212244d6bbb",
        "ba87e86f5aca7b2584a70ca56663ba0d2702643bded73efbbc94d57291d0bbec",
    ),
}


def _fetch_parcellation_files(
    name: str,
    density: str,
    n_regions: int | None,
    networks: int | None,
    cache: Path,
) -> tuple[Path | tuple[Path, Path], dict[str, str]]:
    """Retrieve selected Schaefer or Glasser files from fixed public archives.

    Parameters
    ----------
    name : str
        Validated atlas name, schaefer or glasser.
    density : str
        Validated surface density from _PARCELLATION_ARCHIVES.
    n_regions : int | None
        Selected number of Schaefer parcels; unused for Glasser.
    networks : int | None
        Selected Schaefer network ordering; unused for Glasser.
    cache : pathlib.Path
        Parcellation cache root. Archives and selected files are stored under
        its netneurotools subdirectory, referring to the data source.

    Returns
    -------
    paths : pathlib.Path | tuple[pathlib.Path, pathlib.Path]
        One CIFTI dlabel path, or left/right annotation or GIFTI label paths.
        File contents, label IDs, and vertex ordering are unchanged.
    source : dict[str, str]
        Archive URL, verified archive SHA256, and fixed source manifest URL.
        Pooch downloads missing archives and extracts only the selected files.
    """
    space = {
        "10k": "fsaverage5",
        "41k": "fsaverage6",
        "164k": "fsaverage",
        "32k": "fslr32k",
    }[density]
    if name == "schaefer":
        suffix = f"_desc-{n_regions}Parcels{networks}Networks_deterministic"
        hemispheres = ("LR",) if space == "fslr32k" else ("L", "R")
        extension = ".dlabel.nii" if space == "fslr32k" else ".annot"
        members = [
            f"{space}/atl-Schaefer2018_space-{space}_hemi-{hemi}{suffix}{extension}"
            for hemi in hemispheres
        ]
    else:
        members = [
            f"{space}/atl-MMPAll_space-{space}_hemi-{hemi}_deterministic.label.gii"
            for hemi in ("L", "R")
        ]
    file_id, checksum = _PARCELLATION_ARCHIVES[name, density]
    url = f"https://files.osf.io/v1/resources/udpv8/providers/osfstorage/{file_id}"
    root = cache / "netneurotools"
    stem = f"{name}-{density}"
    pooch.retrieve(
        url=url,
        known_hash=f"sha256:{checksum}",
        fname=f"{stem}.tar.gz",
        path=root,
        processor=pooch.Untar(members=members, extract_dir=stem),
        progressbar=False,
    )
    paths = [root / stem / member for member in members]
    return paths[0] if len(paths) == 1 else (paths[0], paths[1]), {
        "url": url,
        "archive_sha256": checksum,
        "manifest": _PARCELLATION_MANIFEST,
    }


def fetch_parcellation(
    name: str,
    *,
    template: str | None = None,
    density: str | None = None,
    n_regions: int | None = None,
    networks: int | None = None,
    data_dir: str | Path | None = None,
) -> _Parcellation:
    """Fetch a published surface parcellation and its original label tables.

    Parameters
    ----------
    name : {"schaefer", "glasser", "desikan", "destrieux", "yeo"}
        Schaefer 2018, HCP-MMP1.0, Desikan-Killiany, Destrieux 2009, or Yeo 2011.
    template : str | None, default=None
        Surface space. None selects fsLR for Schaefer/Glasser, fsaverage for
        the others. Consult available_parcellations for supported combinations.
    density : str | None, default=None
        Vertices per hemisphere, approximately. None follows fetch_surfaces:
        32k for fsLR and 10k for fsaverage. No registration or resampling occurs.
    n_regions : int | None, default=None
        Schaefer: 100 to 1000 in steps of 100 (default 100); Glasser: 360;
        Yeo: 7 or 17 networks (default 7). Desikan/Destrieux have fixed schemes
        and require None. This selects published data, not runtime subdivision.
    networks : int | None, default=None
        Schaefer network ordering: 7 or 17, default 7. Leave None for other
        parcellations; for Yeo select the network count through n_regions.
    data_dir : str | pathlib.Path | None, default=None
        Provider cache root. None uses ~/.cache/anatplot/parcellations. Provider
        downloads are explicit on first use and cached for subsequent calls.

    Returns
    -------
    parcellation : dict
        "left" and "right" contain integer arrays in the requested vertex order.
        "labels" maps each hemisphere to {ID: {name, color}}, where color is
        source RGBA in [0, 1]. Original IDs are retained, including hemisphere-
        local FreeSurfer IDs; interpret each array with its own label table.
        Zero is background/medial wall and should be masked for add_parcellation.
        "name", "template", "density", "n_regions", and "networks" describe
        the selected variant. "source" records the data provider, source paths,
        file SHA256 hashes, and original paper citation. Direct OSF downloads also
        record their archive URL, archive SHA256, and fixed manifest URL;
        TemplateFlow downloads record the provider package version.

    Raises
    ------
    ValueError
        If a requested combination is unavailable, before any download starts,
        or a provider file has a vertex count inconsistent with the selection.

    Notes
    -----
    Pooch retrieves the same public OSF archives listed by netneurotools,
    without importing that package. TemplateFlow supplies the other atlases.
    No Workbench or FreeSurfer installation is required. Format reading restores
    CIFTI BrainModel vertex indices without transforming coordinates. Existing provider versions may
    themselves have been resampled upstream; cite the source and inspect maps.
    Surface CIFTI files can contain zero voxel spacings in their NIfTI header.
    Their vertex labels do not use these fields; NiBabel may report this header
    diagnostic during loading. Source files are unchanged.
    Desikan 164k is excluded because its published files contain vertex IDs
    absent from their label tables.
    Array length cannot establish registration with an arbitrary user surface.
    """
    if name not in _PARCELLATIONS:
        raise ValueError(f"name must be one of {tuple(_PARCELLATIONS)}")
    choice = _PARCELLATIONS[name]
    template = choice["default"][0] if template is None else template
    if template not in choice["templates"]:
        raise ValueError(f"{name} templates: {tuple(choice['templates'])}")
    density = _TEMPLATES[template][0] if density is None else density
    if density not in choice["templates"][template]:
        raise ValueError(
            f"{name} {template} densities: {choice['templates'][template]}"
        )
    if choice["n_regions"] and n_regions is None:
        n_regions = choice["n_regions"][0]
    if n_regions is not None and n_regions not in choice["n_regions"]:
        raise ValueError(f"{name} n_regions choices: {choice['n_regions'] or (None,)}")
    if name == "schaefer" and networks is None:
        networks = 7
    if networks is not None and networks not in choice["networks"]:
        raise ValueError(f"{name} networks choices: {choice['networks'] or (None,)}")
    cache = (
        Path.home() / ".cache/anatplot/parcellations"
        if data_dir is None
        else Path(data_dir)
    )
    provider = choice["provider"]
    if provider == "netneurotools":
        paths, source = _fetch_parcellation_files(
            name, density, n_regions, networks, cache
        )
    else:
        from templateflow import TemplateFlowClient

        source = {"version": version(provider)}
        client = TemplateFlowClient(root=cache / provider)
        atlas = {
            "desikan": "Desikan2006",
            "destrieux": "Destrieux2009",
            "yeo": "Yeo2011",
        }[name]
        seg = (
            "aparc" if name == "desikan" else f"{n_regions}n" if name == "yeo" else None
        )
        paths = tuple(
            client.get(
                template,
                atlas=atlas,
                density=density,
                hemi=hemi,
                segmentation=seg,
                desc=None,
                suffix="dseg",
                extension=".label.gii",
                raise_empty=True,
            )
            for hemi in ("L", "R")
        )
    values = _read_parcellation(paths)
    if len(values) != 2:
        raise ValueError("parcellation must provide both cortical hemispheres")
    result: dict[str, NDArray[np.int32]] = {}
    labels = {}
    for side, (data, table) in zip(("left", "right"), values):
        if data.shape != (_VERTEX_COUNTS[density],):
            raise ValueError(f"{side} parcellation does not match {template} {density}")
        if not set(np.unique(data)) <= table.keys():
            raise ValueError(
                f"{side} parcellation contains IDs absent from its label table"
            )
        result[side] = np.asarray(data, dtype=np.int32)
        labels[side] = table
    files = (paths,) if isinstance(paths, Path) else tuple(Path(path) for path in paths)
    return {
        "left": result["left"],
        "right": result["right"],
        "labels": labels,
        "name": name,
        "template": template,
        "density": density,
        "n_regions": n_regions,
        "networks": networks,
        "source": {
            "provider": provider,
            **source,
            "files": files,
            "sha256": tuple(sha256(path.read_bytes()).hexdigest() for path in files),
            "citation": choice["citation"],
        },
    }


# Immutable upstream revisions; Pooch verifies every downloaded file.
_MAP_URL = (
    "https://raw.githubusercontent.com/danjgale/surfplot/"
    "5ad3068ce50b5fcd02ca37528ecd5ccff1ed1d29/surfplot/data/"
)
_MAP_HASHES = {
    "lh_default_mode_example.tsv": "fa79f892b58bdfb5b948b34c1937551b49f9d6477f93c15c02ed8ccdecb48894",
    "rh_default_mode_example.tsv": "cf5f6fb6ea265388ed40be18388a291240bb6e1d8ca66b784bf097741266275e",
}


def fetch_dmn_map_fslr32k(
    *,
    data_dir: str | Path | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fetch the Neurosynth default-mode association map on fsLR 32k.

    Parameters
    ----------
    data_dir : str | pathlib.Path | None, default=None
        Persistent cache directory. None uses Pooch's platform cache for
        anatplot. Downloads use fixed upstream revisions and SHA256 checks.

    Returns
    -------
    maps : tuple[numpy.ndarray, numpy.ndarray]
        Left and right float arrays, each of shape (32492,), in fsLR 32k
        vertex order. Zero denotes absent association and can be masked to
        retain the anatomical background.

    Notes
    -----
    This is a meta-analytic association map, not subject-level activation.
    Neurosynth provides the underlying map (Yarkoni et al., 2011,
    doi:10.1038/nmeth.1635); surfplot distributes its neuromaps projection to
    fsLR 32k. Fetching these attributed data files does not require surfplot
    or BrainSpace. No registration or surface transformation occurs here.
    """
    cache = (
        pooch.os_cache("anatplot") / "surface-maps" if data_dir is None else data_dir
    )
    maps = []
    for name, checksum in _MAP_HASHES.items():
        path = pooch.retrieve(
            url=_MAP_URL + name,
            known_hash="sha256:" + checksum,
            fname=name,
            path=cache,
            progressbar=False,
        )
        maps.append(np.loadtxt(path))
    return maps[0], maps[1]
