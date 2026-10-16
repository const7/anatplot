import nibabel as nib
import numpy as np
import pytest
import pyvista as pv

from anatplot.anatomy import load_surface, roi_surface


def label_image(affine=None, unit="mm"):
    data = np.zeros((8, 9, 10), dtype=np.uint8)
    data[2:5, 3:6, 4:7] = 1
    data[5:7, 3:6, 4:7] = 2
    image = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units(unit)
    return image


def test_surface_read_preserves_geometry_and_owns_copy(tmp_path):
    original = pv.Sphere()
    original.point_data["index"] = np.arange(original.n_points)
    path = tmp_path / "surface.vtp"
    original.save(path)
    for source in (original, path):
        mesh = load_surface(source)
        np.testing.assert_array_equal(mesh.points, original.points)
        np.testing.assert_array_equal(mesh.faces, original.faces)
        mesh.points[:] = 0
        assert original.points.max() > 0


@pytest.mark.parametrize("extension", [".gii", ".gii.gz"])
def test_gifti_preserves_vertex_order(tmp_path, extension):
    original = pv.Sphere()
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                original.points.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
            ),
            nib.gifti.GiftiDataArray(
                original.regular_faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"
            ),
        ]
    )
    path = tmp_path / f"surface.surf{extension}"
    nib.save(image, path)
    mesh = load_surface(path)
    np.testing.assert_array_equal(mesh.points, original.points)
    np.testing.assert_array_equal(mesh.faces, original.faces)


@pytest.mark.parametrize("source", [pv.PolyData(), pv.PolyData([[0.0, 0, 0]])])
def test_reject_non_surface(source):
    with pytest.raises(ValueError, match="polygonal"):
        load_surface(source)


def test_reject_nonfinite_surface_coordinates():
    mesh = pv.PolyData.from_regular_faces(
        np.array([[0.0, 0, 0], [1, np.nan, 0], [0, 1, 0]]), [[0, 1, 2]]
    )
    with pytest.raises(ValueError, match="finite"):
        load_surface(mesh)


@pytest.mark.parametrize(
    "linear",
    [
        np.diag([2, 3, 4]),
        np.array([[0, -3, 0], [2, 0, 0], [0, 0, 4]]),
        np.diag([-2, 3, 4]),
        np.array([[1, 0.3, 0], [0, 2, 0], [0, 0, 3]]),
    ],
)
def test_roi_applies_full_affine_without_moving_voxel_boundaries(linear):
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = [17, -23, 41]
    reference = roi_surface(label_image(), [1], smoothing=False)
    result = roi_surface(label_image(affine), [1], smoothing=False)
    expected = reference.points @ linear.T + affine[:3, 3]
    np.testing.assert_allclose(result.points, expected, atol=1e-5)
    assert result.n_open_edges == 0
    # Reflected affines must retain outward winding.
    triangles = result.points[result.regular_faces]
    signed_volume = (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )
    assert signed_volume > 0


@pytest.mark.parametrize("unit,factor", [("mm", 1), ("meter", 1000), ("micron", 0.001)])
def test_units_convert_geometry_to_mm(unit, factor):
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    expected = roi_surface(label_image(affine), [1], smoothing=False)
    affine[:3] /= factor
    result = roi_surface(label_image(affine, unit), [1], smoothing=False)
    np.testing.assert_allclose(result.points, expected.points, atol=1e-4)


def test_union_removes_internal_interface_and_preserves_source(tmp_path):
    image = label_image()
    data = np.asarray(image.dataobj).copy()
    path = tmp_path / "labels.nii.gz"
    nib.save(image, path)
    union = roi_surface(path, [1, 2], smoothing=False)
    binary = nib.Nifti1Image((data > 0).astype(np.uint8), image.affine, image.header)
    expected = roi_surface(binary, [1], smoothing=False)
    np.testing.assert_allclose(union.points, expected.points)
    np.testing.assert_array_equal(union.faces, expected.faces)
    smoothed = roi_surface(image, [1, 2])
    assert smoothed.faces.size > 0 and np.isfinite(smoothed.points).all()
    np.testing.assert_array_equal(image.dataobj, data)


def test_padding_closes_region_touching_image_boundary():
    image = nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.uint8), np.eye(4))
    image.header.set_xyzt_units("mm")
    mesh = roi_surface(image, [1], smoothing=False)
    assert mesh.n_open_edges == 0
    np.testing.assert_allclose(mesh.bounds, [-0.5, 2.5, -0.5, 3.5, -0.5, 4.5])


@pytest.mark.parametrize("labels", [[], [0], [3], [1.5], [float("nan")]])
def test_invalid_label_selection(labels):
    with pytest.raises(ValueError, match="positive integer"):
        roi_surface(label_image(), labels)


def test_reject_unknown_units_probability_maps_and_smoothed_shear():
    with pytest.raises(ValueError, match="units"):
        roi_surface(label_image(unit="unknown"), [1])
    image = nib.Nifti1Image(np.full((3, 3, 3), 0.5), np.eye(4))
    with pytest.raises(ValueError, match="integer label"):
        roi_surface(image, [1])
    affine = np.eye(4)
    affine[0, 1] = 0.2
    with pytest.raises(ValueError, match="sheared"):
        roi_surface(label_image(affine), [1])
