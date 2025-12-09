"""
Geometry and mesh utilities for VTK examples
"""

import vtk
from vtk.util import numpy_support
import numpy as np
import trueform as tf


class MeshData:
    """Container for mesh rendering and interaction data"""
    def __init__(self, mesh, actor, matrix):
        self.mesh = mesh        # tf.Mesh
        self.actor = actor      # vtkActor
        self.matrix = matrix    # vtkMatrix4x4


def random_rotation_matrix(dtype=np.float32):
    """
    Generate a random 3x3 rotation matrix using uniform quaternion sampling

    Uses uniform sampling on SO(3) via quaternion generation.
    See: http://planning.cs.uiuc.edu/node198.html

    Parameters
    ----------
    dtype : numpy dtype
        Data type for the output matrix (default: np.float32)

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    u = np.random.uniform(0, 1, 3)

    # Generate quaternion
    q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
    q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
    q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
    q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])

    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ], dtype=dtype)

    return R


def numpy_to_polydata(points: np.ndarray, faces) -> vtk.vtkPolyData:
    """
    Convert numpy arrays to VTK PolyData (VTK 9+ format)

    Parameters
    ----------
    points : np.ndarray
        (m, 3) float array of point coordinates
    faces : np.ndarray or tf.OffsetBlockedArray
        (n, 3) int array of vertex indices (0-based), or OffsetBlockedArray for dynamic meshes

    Returns
    -------
    vtk.vtkPolyData
        VTK polydata object
    """
    # Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_support.numpy_to_vtk(
            num_array=points,
            deep=True,
            array_type=vtk.VTK_FLOAT if points.dtype == np.float32 else vtk.VTK_DOUBLE,
        )
    )

    # CellArray - handle both numpy arrays and OffsetBlockedArray
    if isinstance(faces, tf.OffsetBlockedArray):
        # Dynamic mesh - use offsets and data directly
        connectivity = faces.data.astype(np.int64).ravel()
        offsets = faces.offsets.astype(np.int64).ravel()
    else:
        # Fixed-size faces (numpy array)
        n_faces = faces.shape[0]
        connectivity = faces.astype(np.int64).ravel()
        face_size = faces.shape[1]
        offsets = np.arange(0, face_size * n_faces + 1, face_size, dtype=np.int64)

    vtk_conn = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)
    vtk_offs = numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True)

    cell_array = vtk.vtkCellArray()
    cell_array.SetData(vtk_offs, vtk_conn)

    # PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cell_array)
    return poly


def compute_centering_and_scaling_transform(points: np.ndarray, target_radius: float = 10.0) -> np.ndarray:
    """
    Compute 4x4 transformation matrix to center and scale mesh

    Mimics C++ center_and_scale() function:
    - Centers mesh at origin
    - Scales to fit in sphere of given radius

    Parameters
    ----------
    points : np.ndarray
        (n, 3) array of point coordinates
    target_radius : float
        Target bounding sphere radius

    Returns
    -------
    np.ndarray
        4x4 transformation matrix
    """
    # Compute AABB
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt) / 2.0

    # Compute bounding sphere radius (diagonal / 2)
    diagonal = max_pt - min_pt
    radius = np.linalg.norm(diagonal) / 2.0

    # Scale factor to fit in target radius
    scale = target_radius / radius

    # Create transform: first translate to origin, then scale
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] *= scale
    T[0:3, 3] = -center * scale

    return T


def curves_to_polydata(paths, points: np.ndarray) -> vtk.vtkPolyData:
    """
    Convert isocontour curves to VTK PolyData

    Parameters
    ----------
    paths : OffsetBlockedArray
        Paths as indices into points array (from tf.isocontours)
    points : np.ndarray
        (n, 3) array of curve point coordinates

    Returns
    -------
    vtk.vtkPolyData
        VTK polydata with polylines
    """
    # Convert points to VTK
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_support.numpy_to_vtk(
            num_array=points,
            deep=False,
            array_type=vtk.VTK_FLOAT if points.dtype == np.float32 else vtk.VTK_DOUBLE,
        )
    )

    # Create lines from paths using offset+connectivity arrays directly (VTK 9+ format)
    # paths.offsets and paths.data are already in the correct format!
    connectivity = paths.data.astype(np.int64).ravel()
    offsets = paths.offsets.astype(np.int64).ravel()
    vtk_conn = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)
    vtk_offs = numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True)

    lines = vtk.vtkCellArray()
    lines.SetData(vtk_offs, vtk_conn)

    # Create polydata
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(lines)
    return poly


def create_tube_filter(radius: float = 0.05, number_of_sides: int = 20) -> vtk.vtkTubeFilter:
    """
    Create a tube filter for curve visualization

    Parameters
    ----------
    radius : float
        Tube radius
    number_of_sides : int
        Number of sides for tube cross-section

    Returns
    -------
    vtk.vtkTubeFilter
        Configured tube filter
    """
    tubes = vtk.vtkTubeFilter()
    tubes.SetRadius(radius)
    tubes.SetNumberOfSides(number_of_sides)
    return tubes


def load_mesh(filename, position, target_radius=10.0, random_rotation=True, dynamic=False):
    """
    Load mesh from file, center, scale, position, and optionally apply random rotation

    Parameters
    ----------
    filename : str
        Path to mesh file (STL format)
    position : tuple
        (x, y, z) position in world coordinates
    target_radius : float
        Target bounding sphere radius for scaling
    random_rotation : bool
        Whether to apply random rotation
    dynamic : bool
        If True, use OffsetBlockedArray for faces (dynamic mesh)

    Returns
    -------
    MeshData
        Container with configured mesh, actor, and matrix
    """
    # Read STL with trueform
    faces, points = tf.read_stl(filename)

    # Optionally convert to dynamic mesh
    if dynamic:
        faces = tf.as_offset_blocked(faces)

    # Compute centering and scaling transform
    center_scale_transform = compute_centering_and_scaling_transform(points, target_radius=target_radius)

    # Create rotation transform
    if random_rotation:
        R = random_rotation_matrix(dtype=np.float32)
        rotation_transform = np.eye(4, dtype=np.float32)
        rotation_transform[:3, :3] = R
    else:
        rotation_transform = np.eye(4, dtype=np.float32)

    # Create position transform (translation)
    position_transform = np.eye(4, dtype=np.float32)
    position_transform[0, 3] = position[0]  # X offset
    position_transform[1, 3] = position[1]  # Y offset
    position_transform[2, 3] = position[2]  # Z offset

    # Combine: first center/scale, then rotate, then position
    combined_transform = position_transform @ rotation_transform @ center_scale_transform

    # Create trueform mesh with transformation
    mesh = tf.Mesh(faces, points)
    mesh.transformation = combined_transform

    # Create VTK visualization (uses original points)
    polydata = numpy_to_polydata(points, faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Set VTK transformation matrix
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, combined_transform[i, j])
    actor.SetUserMatrix(matrix)

    return MeshData(mesh, actor, matrix)


def load_mesh_shared(source_mesh_data, position, random_rotation=True):
    """
    Create a shared view of an existing mesh with a different transformation.

    Uses shared_view() to share geometry and cached structures (tree, face_membership,
    manifold_edge_link) with the source mesh, avoiding duplicate memory and computation.

    Parameters
    ----------
    source_mesh_data : MeshData
        Source mesh data to share geometry from
    position : tuple
        (x, y, z) position in world coordinates
    random_rotation : bool
        Whether to apply random rotation

    Returns
    -------
    MeshData
        Container with shared mesh view, new actor, and matrix
    """
    # Create shared view - shares geometry and structures
    shared_mesh = source_mesh_data.mesh.shared_view()

    # Get points from original mesh for visualization
    points = source_mesh_data.mesh.points
    faces = source_mesh_data.mesh.faces

    # Compute centering and scaling transform (same as original)
    center_scale_transform = compute_centering_and_scaling_transform(points, target_radius=10.0)

    # Create rotation transform
    if random_rotation:
        R = random_rotation_matrix(dtype=np.float32)
        rotation_transform = np.eye(4, dtype=np.float32)
        rotation_transform[:3, :3] = R
    else:
        rotation_transform = np.eye(4, dtype=np.float32)

    # Create position transform (translation)
    position_transform = np.eye(4, dtype=np.float32)
    position_transform[0, 3] = position[0]
    position_transform[1, 3] = position[1]
    position_transform[2, 3] = position[2]

    # Combine: first center/scale, then rotate, then position
    combined_transform = position_transform @ rotation_transform @ center_scale_transform

    # Set transformation on the shared view
    shared_mesh.transformation = combined_transform

    # Create VTK visualization (uses original points)
    polydata = numpy_to_polydata(points, faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Set VTK transformation matrix
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, combined_transform[i, j])
    actor.SetUserMatrix(matrix)

    return MeshData(shared_mesh, actor, matrix)
