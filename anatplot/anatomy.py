"""Read surface geometry and derive display surfaces from discrete volumes."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike, NDArray


def load_surface(source: pv.PolyData | str | Path) -> pv.PolyData:
    """Read or copy a polygonal surface without changing its coordinates.

    Parameters
    ----------
    source : pyvista.PolyData | str | pathlib.Path
        Mesh or local surface file. GIFTI files must contain one POINTSET and
        one TRIANGLE array; other formats are delegated to PyVista. Coordinates
        must already be RAS in millimetres. GIFTI coordinate transforms are not
        applied automatically.

    Returns
    -------
    mesh : pyvista.PolyData
        Independent surface retaining vertex order and connectivity. Modifying
        it does not modify an input mesh. No registration or rescaling occurs.

    Raises
    ------
    ValueError
        If geometry is empty, nonpolygonal, or has nonfinite coordinates.
    """
    if isinstance(source, pv.PolyData):
        mesh = source.copy(deep=True)
    elif Path(source).name.endswith((".gii", ".gii.gz")):
        image = nib.load(source)
        points = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        triangles = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
        if len(points) != 1 or len(triangles) != 1:
            raise ValueError("GIFTI requires one POINTSET and one TRIANGLE array")
        mesh = pv.PolyData.from_regular_faces(points[0].data, triangles[0].data)
    else:
        mesh = pv.read(source)
    if (
        not isinstance(mesh, pv.PolyData)
        or not mesh.faces.size
        or mesh.n_lines
        or mesh.n_verts
        or mesh.n_strips
        or not np.isfinite(mesh.points).all()
    ):
        raise ValueError("expected a nonempty finite polygonal surface")
    return mesh


def roi_surface(
    image: nib.Nifti1Image | nib.Nifti2Image | str | Path,
    labels: ArrayLike,
    *,
    smoothing: bool = True,
) -> pv.PolyData:
    """Extract the outer surface of a selected union of discrete labels.

    Parameters
    ----------
    image : nibabel.Nifti1Image | nibabel.Nifti2Image | str | pathlib.Path
        Three-dimensional NIfTI image or local path with finite, nonnegative
        integer-valued labels. Zero is background. Spatial units must be declared
        as mm, meter, or micron; unknown units are rejected. Probability and
        statistical maps need an explicit upstream conversion.
    labels : collections.abc.Sequence[int]
        Nonempty positive IDs present in the image. Selected labels form one
        region; their internal interfaces are omitted. Use separate calls for
        separately colored regions.
    smoothing : bool, default=True
        Use PyVista SurfaceNets smoothing with its default settings. Smoothing
        changes display geometry only. Sheared affines require False or explicit
        upstream resampling, because smoothing distances use orthogonal axes.

    Returns
    -------
    mesh : pyvista.PolyData
        Triangulated display surface in RAS millimetres, with the full image
        affine applied. The input image and its labels are unchanged.

    Raises
    ------
    ValueError
        If labels, units, or the affine violate the input contract.

    Notes
    -----
    Uses ``ImageData.contour_labels`` on a binary selection, with background
    padding. Boundaries lie between voxel centres. Thin or corner-connected
    regions need visual inspection; manifold geometry is not guaranteed.
    """
    if isinstance(image, (str, Path)):
        image = nib.load(image)
    data = np.asanyarray(image.dataobj)
    if (
        data.ndim != 3
        or not data.size
        or not np.isfinite(data).all()
        or np.any(data < 0)
        or not np.array_equal(data, np.rint(data))
    ):
        raise ValueError("expected a 3-D nonnegative integer label image")
    ids = np.asarray(labels)
    if (
        ids.ndim != 1
        or not ids.size
        or not np.isfinite(ids).all()
        or np.any(ids <= 0)
        or not np.array_equal(ids, np.rint(ids))
        or not np.isin(ids, data).all()
    ):
        raise ValueError("labels must be positive integer IDs present in the image")
    unit = image.header.get_xyzt_units()[0]
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors:
        raise ValueError("NIfTI must declare spatial units: mm, meter, or micron")
    affine = np.array(image.affine, dtype=float, copy=True)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
        or np.linalg.matrix_rank(affine[:3, :3]) != 3
    ):
        raise ValueError("expected an invertible homogeneous voxel-to-world affine")
    affine[:3] *= factors[unit]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = affine[:3, :3] / spacing
    if smoothing and not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
        raise ValueError("sheared affine requires smoothing=False or resampling")
    grid = pv.ImageData(dimensions=data.shape, spacing=spacing)
    grid.point_data["region"] = np.isin(data, ids).astype(np.uint8).ravel(order="F")
    mesh = grid.contour_labels(
        scalars="region",
        smoothing=smoothing,
        output_mesh_type="triangles",
        boundary_style="external",
    )
    transform = affine.copy()
    transform[:3, :3] = direction
    mesh.transform(transform, inplace=True)
    # Reflections reverse face winding; restore outward orientation explicitly.
    if np.linalg.det(direction) < 0:
        mesh.flip_faces(inplace=True)
    return mesh


def _project_contacts(
    source: pv.PolyData,
    target: pv.PolyData,
    cells: NDArray[np.integer],
    projected: NDArray[np.floating],
    *,
    offset: float,
) -> NDArray[np.floating]:
    """Map localized triangle positions onto corresponding display faces.

    Parameters
    ----------
    source : pyvista.PolyData
        Anatomical triangle mesh in RAS millimetres, with outward face winding.
    target : pyvista.PolyData
        Display triangle mesh in RAS millimetres. Ordered faces and vertex IDs
        must correspond to source; display deformation is allowed.
    cells : numpy.ndarray
        Integer source face IDs of shape (n_contacts,), in contact row order.
    projected : numpy.ndarray
        Closest anatomical positions of shape (n_contacts, 3), in RAS millimetres.
        Each position lies on the source triangle selected by cells.
    offset : float
        Outward display offset in millimetres along each target face's unit normal.

    Returns
    -------
    centers : numpy.ndarray
        Floating array of shape (n_contacts, 3) in contact order. Barycentric
        positions are transferred to target and offset; inputs are unchanged.

    Raises
    ------
    ValueError
        If a selected source or display triangle is degenerate.
    """
    triangles = source.points[source.regular_faces[cells]].astype(
        np.float64, copy=False
    )
    origin = triangles[:, 0]
    u = triangles[:, 1] - origin
    v = triangles[:, 2] - origin
    delta = projected - origin
    source_normals = np.cross(u, v)
    squared_norms = np.einsum("ij,ij->i", source_normals, source_normals)
    if np.any(squared_norms == 0):
        raise ValueError("contact projection requires nondegenerate source triangles")
    b = np.einsum("ij,ij->i", np.cross(delta, v), source_normals) / squared_norms
    c = np.einsum("ij,ij->i", np.cross(u, delta), source_normals) / squared_norms
    weights = np.column_stack((1 - b - c, b, c))
    display = target.points[target.regular_faces[cells]].astype(np.float64, copy=False)
    normals = np.cross(display[:, 1] - display[:, 0], display[:, 2] - display[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("contact projection requires nondegenerate display triangles")
    return (
        np.einsum("ij,ijk->ik", weights, display) + offset * normals / lengths[:, None]
    )
