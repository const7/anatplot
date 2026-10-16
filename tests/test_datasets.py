from hashlib import sha256
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from anatplot import datasets
from anatplot.datasets import fetch_surfaces


@pytest.mark.parametrize(
    "template,density", [("fsaverage", "10k"), ("fsLR", "32k"), ("civet", "41k")]
)
def test_default_selection_retains_left_right_and_cache(
    monkeypatch, tmp_path, template, density
):
    left, right = tmp_path / "L.surf.gii", tmp_path / "R.surf.gii"
    requests = []

    def provider(atlas, requested_density, *, data_dir, verbose):
        requests.append((atlas, requested_density, data_dir))
        reference = "pial" if template == "fsaverage" else "midthickness"
        return {
            "inflated": (left, right),
            reference: (tmp_path / "source-L.gii", tmp_path / "source-R.gii"),
        }

    monkeypatch.setattr(datasets, "fetch_atlas", provider)
    surfaces = fetch_surfaces(template, data_dir=tmp_path)
    assert surfaces == {
        "left": left,
        "right": right,
        "projection_surface": (tmp_path / "source-L.gii", tmp_path / "source-R.gii"),
    }
    assert requests == [(template, density, tmp_path)]


@pytest.mark.parametrize(
    "options",
    [
        {"template": "MNI152"},
        {"template": "fsLR", "surface": "pial"},
        {"template": "fsLR", "density": "8k", "surface": "veryinflated"},
    ],
)
def test_invalid_choices_do_not_start_a_download(monkeypatch, options):
    def unexpected_fetch(*args, **kwargs):
        pytest.fail("invalid geometry should be rejected before downloading")

    monkeypatch.setattr(datasets, "fetch_atlas", unexpected_fetch)
    with pytest.raises(ValueError):
        fetch_surfaces(**options)


def test_parcellation_restores_brain_model_vertex_indices_and_label_table(
    monkeypatch, tmp_path
):
    from nibabel.cifti2 import BrainModelAxis, LabelAxis

    left = BrainModelAxis.from_surface([3, 0], 32492, name="CortexLeft")
    right = BrainModelAxis.from_surface([2, 1], 32492, name="CortexRight")
    table = {i: (str(i), (1, 0, 0, 1)) for i in (0, 1, 2, 51, 52)}
    header = nib.Cifti2Header.from_axes((LabelAxis(["atlas"], [table]), left + right))
    image = nib.Cifti2Image(np.array([[1, 2, 51, 52]], dtype=np.int32), header)
    path = tmp_path / "atlas.dlabel.nii"
    nib.save(image, path)
    # Reproduce the published Schaefer NIfTI header without changing CIFTI XML.
    offset = nib.Nifti2Header.template_dtype.fields["pixdim"][1]
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(np.zeros(8, dtype="<f8").tobytes())
    source_bytes = path.read_bytes()

    def fetcher(name, density, n_regions, networks, cache):
        assert (name, density, n_regions, networks) == ("schaefer", "32k", 400, 17)
        assert cache == tmp_path
        return path, {"url": "https://example.org/archive"}

    monkeypatch.setattr(datasets, "_fetch_parcellation_files", fetcher)
    result = datasets.fetch_parcellation(
        "schaefer", n_regions=400, networks=17, data_dir=tmp_path
    )
    np.testing.assert_array_equal(result["left"][:4], [2, 0, 0, 1])
    np.testing.assert_array_equal(result["right"][:4], [0, 52, 51, 0])
    assert not result["left"][4:].any()
    assert result["labels"]["right"][52]["name"] == "52"
    assert result["labels"]["left"][1]["color"] == (1, 0, 0, 1)
    assert (result["template"], result["density"]) == ("fsLR", "32k")
    assert (result["n_regions"], result["networks"]) == (400, 17)
    assert result["source"]["files"] == (path,)
    assert result["source"]["sha256"] == (sha256(source_bytes).hexdigest(),)
    assert path.read_bytes() == source_bytes


def test_dmn_map_fslr32k_retains_hemisphere_and_vertex_order(monkeypatch, tmp_path):
    for side, values in (("lh", [0, 3, 1]), ("rh", [9, 0, 2])):
        np.savetxt(tmp_path / f"{side}_default_mode_example.tsv", values)
    monkeypatch.setattr(
        datasets.pooch, "retrieve", lambda **kwargs: tmp_path / kwargs["fname"]
    )
    left, right = datasets.fetch_dmn_map_fslr32k()
    np.testing.assert_array_equal(left, [0, 3, 1])
    np.testing.assert_array_equal(right, [9, 0, 2])


@pytest.mark.parametrize(
    "options",
    [
        {"name": "unknown"},
        {"name": "schaefer", "template": "civet"},
        {"name": "schaefer", "template": "fsLR", "density": "164k"},
        {"name": "schaefer", "n_regions": 150},
        {"name": "schaefer", "networks": 8},
        {"name": "desikan", "n_regions": 100},
        {"name": "destrieux", "networks": 7},
    ],
)
def test_invalid_parcellation_combinations_do_not_download(monkeypatch, options):
    import templateflow

    def unexpected(*args, **kwargs):
        pytest.fail("Unavailable combinations must be rejected before downloading")

    monkeypatch.setattr(datasets, "_fetch_parcellation_files", unexpected)
    monkeypatch.setattr(templateflow, "TemplateFlowClient", unexpected)
    with pytest.raises(ValueError):
        datasets.fetch_parcellation(**options)


@pytest.mark.parametrize(
    "name,atlas,segmentation",
    [
        ("desikan", "Desikan2006", "aparc"),
        ("destrieux", "Destrieux2009", None),
        ("yeo", "Yeo2011", "7n"),
    ],
)
def test_templateflow_selection_and_source_ids(
    monkeypatch, tmp_path, name, atlas, segmentation
):
    import templateflow

    paths = {}
    for side, key in (("L", 17), ("R", 48)):
        data = np.zeros(10242, dtype=np.int32)
        data[3] = key
        table = nib.gifti.GiftiLabelTable()
        for index, label_name in ((0, "background"), (key, f"{side}-region")):
            label = nib.gifti.GiftiLabel(index, 1, 0, 0, 1)
            label.label = label_name
            table.labels.append(label)
        image = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(data, intent="NIFTI_INTENT_LABEL")],
            labeltable=table,
        )
        paths[side] = tmp_path / f"{side}.label.gii"
        nib.save(image, paths[side])

    class Client:
        def __init__(self, *, root):
            assert root == tmp_path / "templateflow"

        def get(self, template, **query):
            assert template == "fsaverage"
            assert query["atlas"] == atlas
            assert query["density"] == "10k"
            assert query["segmentation"] == segmentation
            assert query["desc"] is None
            return paths[query["hemi"]]

    monkeypatch.setattr(templateflow, "TemplateFlowClient", Client)
    result = datasets.fetch_parcellation(name, data_dir=tmp_path)
    assert result["left"][3] == 17
    assert result["right"][3] == 48
    assert result["labels"]["left"][17]["name"] == "L-region"
    assert result["source"]["provider"] == "templateflow"


def test_fsaverage_annotations_preserve_packed_ids(monkeypatch, tmp_path):
    paths = []
    table = np.array([[0, 0, 0, 255], [10, 20, 30, 0]])
    data = np.zeros(10242, dtype=np.int32)
    data[4] = 1
    for side in ("L", "R"):
        path = tmp_path / f"{side}.annot"
        nib.freesurfer.write_annot(path, data, table, [b"background", b"parcel"])
        paths.append(path)
    monkeypatch.setattr(
        datasets,
        "_fetch_parcellation_files",
        lambda *args: (tuple(paths), {}),
    )
    result = datasets.fetch_parcellation("schaefer", template="fsaverage")
    key = 10 + 20 * 256 + 30 * 65536
    assert result["left"][4] == key
    assert result["left"][0] == 0
    assert result["labels"]["left"][key]["name"] == "parcel"
    assert result["labels"]["left"][key]["color"][3] == 1


def test_parcellation_rejects_vertex_ids_absent_from_label_table(monkeypatch, tmp_path):
    data = np.zeros(32492, dtype=np.int32)
    data[0] = -1
    table = nib.gifti.GiftiLabelTable()
    label = nib.gifti.GiftiLabel(0, 0, 0, 0, 0)
    label.label = "background"
    table.labels.append(label)
    image = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(data, intent="NIFTI_INTENT_LABEL")],
        labeltable=table,
    )
    path = tmp_path / "missing-label.label.gii"
    nib.save(image, path)
    monkeypatch.setattr(
        datasets, "_fetch_parcellation_files", lambda *args: ((path, path), {})
    )
    with pytest.raises(ValueError):
        datasets.fetch_parcellation("glasser")


@pytest.mark.parametrize(
    "template,density,n_vertices",
    [
        ("fsaverage", "10k", 10242),
        ("fsLR", "4k", 4002),
        ("fsLR", "8k", 7842),
        ("fsLR", "32k", 32492),
        ("civet", "41k", 40962),
    ],
)
def test_sulcal_depth_retains_hemisphere_vertex_order_and_source(
    monkeypatch, tmp_path, template, density, n_vertices
):
    paths = []
    original = ([0.5, -1.5, 2.0], [-2.5, 1.0, 0.0])
    for side, values in zip(("left", "right"), original):
        path = tmp_path / f"{side}.shape.gii"
        data = np.zeros(n_vertices, dtype=np.float32)
        data[: len(values)] = values
        image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data)])
        nib.save(image, path)
        paths.append(path)
    source_bytes = [path.read_bytes() for path in paths]
    requests = []

    def provider(requested_template, requested_density, *, data_dir, verbose):
        requests.append((requested_template, requested_density, data_dir))
        return {"sulc": tuple(paths)}

    monkeypatch.setattr(datasets, "fetch_atlas", provider)
    result = datasets.fetch_sulcal_depth(template, density=density, data_dir=tmp_path)
    for side, expected in zip(("left", "right"), original):
        np.testing.assert_array_equal(result[side][: len(expected)], expected)
        assert result[side].shape == (n_vertices,)
        assert result[side].dtype == np.float64
    result["left"][:] = 100
    reread = datasets.fetch_sulcal_depth(template, density=density, data_dir=tmp_path)
    np.testing.assert_array_equal(reread["left"][: len(original[0])], original[0])
    assert requests == [(template, density, tmp_path)] * 2
    assert [path.read_bytes() for path in paths] == source_bytes


def test_sulcal_depth_rejects_wrong_vertex_count(monkeypatch, tmp_path):
    path = tmp_path / "short.shape.gii"
    nib.save(
        nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(np.zeros(3, dtype=np.float32))]
        ),
        path,
    )
    monkeypatch.setattr(
        datasets, "fetch_atlas", lambda *args, **kwargs: {"sulc": (path, path)}
    )
    with pytest.raises(ValueError, match="sulcal depth does not match"):
        datasets.fetch_sulcal_depth(data_dir=tmp_path)


@pytest.mark.parametrize(
    "name,density,space,hemispheres,extension",
    [
        ("schaefer", "10k", "fsaverage5", ("L", "R"), ".annot"),
        ("schaefer", "32k", "fslr32k", ("LR",), ".dlabel.nii"),
        ("glasser", "32k", "fslr32k", ("L", "R"), ".label.gii"),
    ],
)
def test_direct_archive_selection_and_integrity(
    monkeypatch, tmp_path, name, density, space, hemispheres, extension
):
    from io import BytesIO
    from shutil import copyfile
    from tarfile import TarInfo
    from tarfile import open as open_tar

    import pooch.downloaders

    prefix = "Schaefer2018" if name == "schaefer" else "MMPAll"
    description = "_desc-400Parcels17Networks" if name == "schaefer" else ""
    members = [
        f"{space}/atl-{prefix}_space-{space}_hemi-{side}"
        f"{description}_deterministic{extension}"
        for side in hemispheres
    ]
    archive = tmp_path / "source.tar.gz"
    with open_tar(archive, "w:gz") as stream:
        # Deliberately reverse archive order; output must retain L/R ordering.
        for member in reversed(members):
            payload = member.encode()
            entry = TarInfo(member)
            entry.size = len(payload)
            stream.addfile(entry, BytesIO(payload))
    checksum = sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setitem(
        datasets._PARCELLATION_ARCHIVES, (name, density), ("file-id", checksum)
    )
    requests = []

    def download(url, output_file, pooch):
        requests.append(url)
        copyfile(archive, output_file)

    monkeypatch.setattr(pooch.downloaders, "HTTPDownloader", lambda **kwargs: download)
    cache = tmp_path / "cache"
    selected, source = datasets._fetch_parcellation_files(name, density, 400, 17, cache)
    paths = (selected,) if isinstance(selected, Path) else selected
    assert [path.read_text() for path in paths] == members
    assert source["archive_sha256"] == checksum
    assert source["url"] == requests[0]
    assert len(requests) == 1
    if name == "schaefer" and density == "10k":
        monkeypatch.setitem(
            datasets._PARCELLATION_ARCHIVES, (name, density), ("file-id", "0" * 64)
        )
        with pytest.raises(ValueError):
            datasets._fetch_parcellation_files(
                name, density, 400, 17, tmp_path / "bad-cache"
            )
