"""
CsgGraph: build one arrangement, answer many CSG queries.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import numpy as np

from .. import _trueform

from .._core import OffsetBlockedArray
from .._dispatch.meta import extract_meta
from .._dispatch.suffix import build_suffix
from .expr import Expr, _as_expr
from .index_maps import DomainsIndexMap, MeshIndexMap

_MODE_MAP = {"sos": 1, "primitives": 2}
_RESOLVE_CROSSINGS = 4
_RESOLVE_SELF_CROSSINGS = 8

_TRIANGULATION_MAP = {"cdt": 0, "refined_cdt": 1}

_EXCLUDE_OUTER_SHELL = 1
_IGNORE_OPEN_FRAGMENTS = 2


class CsgGraph:
    """
    The arrangement of N meshes with its domain classification, built
    once; any boolean expression over the operands is then answered as
    a cheap query against the same build.

    The graph holds references to the input meshes (accessible as
    ``.forms``); the underlying native engine is opaque.

    Parameters
    ----------
    meshes : list of Mesh
        Two or more closed 3D triangle meshes (triangulate n-gon meshes
        first). All must share index and real dtypes.
    sheets : list of int, optional
        Indices of operands declared as oriented open sheets: they cut
        volumes through the same algebra without enclosing one.
    mode : str, default "primitives"
        Intersection mode. "sos" or "primitives".
    tolerance : float, default 0.0
        World-coordinate distance band for predicate tolerance (0 = exact).
    resolve_crossings : bool, default True
        Resolve crossings between different contours on the same face.
    triangulation : str, default "cdt"
        Cut-surface triangulation: "cdt" (plain constrained Delaunay per
        cut loop) or "refined_cdt" (quality refinement of the cut
        surface; shared boundaries stay watertight by construction).

    Examples
    --------
    >>> graph = tf.CsgGraph([a, b, c], triangulation="refined_cdt")
    >>> faces_points = graph.mesh(tf.op(0) - tf.op(1))
    >>> cells, ids = graph.domains()
    """

    def __init__(
        self,
        meshes,
        *,
        sheets=None,
        mode: str = "primitives",
        tolerance: float = 0.0,
        resolve_crossings: bool = True,
        triangulation: str = "cdt",
    ):
        if len(meshes) < 2:
            raise ValueError("CsgGraph needs at least two meshes")
        meta = extract_meta(meshes[0])
        for m in meshes:
            m_meta = extract_meta(m)
            if (
                m_meta.index_dtype != meta.index_dtype
                or m_meta.real_dtype != meta.real_dtype
                or m_meta.dims != meta.dims
            ):
                raise ValueError(
                    "all meshes must share index dtype, real dtype, and dims"
                )
            if m.is_dynamic:
                raise ValueError(
                    "CsgGraph takes triangle meshes; call triangulated() on "
                    "n-gon meshes first"
                )
        if meta.dims != 3:
            raise ValueError("CsgGraph requires 3D meshes")

        if mode not in _MODE_MAP:
            raise ValueError(f"mode must be 'sos' or 'primitives', got '{mode}'")
        if triangulation not in _TRIANGULATION_MAP:
            raise ValueError(
                "triangulation must be 'cdt' or 'refined_cdt', "
                f"got '{triangulation}'"
            )
        sheet_list = [int(s) for s in (sheets or [])]
        for s in sheet_list:
            if s < 0 or s >= len(meshes):
                raise ValueError(f"sheet index {s} out of range")

        mode_int = _MODE_MAP[mode]
        if resolve_crossings:
            mode_int |= _RESOLVE_CROSSINGS

        self._forms = list(meshes)
        self._sheets = tuple(sheet_list)
        self._mode = mode
        self._tolerance = tolerance
        self._resolve_crossings = resolve_crossings
        self._triangulation = triangulation
        self._created_points = None

        suffix = build_suffix(meta)
        cls = getattr(_trueform.csg, f"CsgGraph_{suffix}", None)
        if cls is None:
            raise TypeError(
                f"unsupported mesh type combination for CsgGraph: {suffix}"
            )
        self._wrapper = cls(
            [m._wrapper for m in meshes],
            sheet_list,
            mode_int,
            tolerance,
            _TRIANGULATION_MAP[triangulation],
        )

    # -- remembered construction state (no native calls) -----------------
    @property
    def forms(self):
        """The input meshes, as passed."""
        return list(self._forms)

    @property
    def sheets(self):
        """Operand indices declared as sheets, as passed."""
        return self._sheets

    @property
    def mode(self):
        """Intersection mode this graph was built with."""
        return self._mode

    @property
    def tolerance(self):
        """Predicate tolerance this graph was built with."""
        return self._tolerance

    @property
    def triangulation(self):
        """Cut-surface triangulation this graph was built with."""
        return self._triangulation

    @property
    def created_points(self):
        """
        Points the arrangement created (intersections; plus refinement
        points under ``triangulation="refined_cdt"``), shape (K, dims),
        in the input real dtype.
        """
        if self._created_points is None:
            self._created_points = self._wrapper.created_points()
        return self._created_points

    def intersection_curves(self):
        """
        The intersection polylines of the arrangement — where two operand
        surfaces cross (coincident walls excluded).

        Returns
        -------
        paths : OffsetBlockedArray
            Polylines as blocks of point indices into ``curve_points``.
        curve_points : np.ndarray
            Curve point coordinates, shape (K, dims).
        """
        (off, data), pts = self._wrapper.intersection_curves()
        return OffsetBlockedArray(off, data), pts

    # -- queries ----------------------------------------------------------
    def mesh(self, expr=None, *, return_source_ids: bool = False,
             return_index_map: bool = False):
        """
        The boolean result mesh for ``expr``; with no expression, the full
        arrangement mesh (every input face, cut at intersections).

        Parameters
        ----------
        expr : Expr or int, optional
            Boolean expression over operand indices (``tf.op(0) - tf.op(1)``),
            or a single operand index. Omit for the full arrangement mesh
            (``return_index_map`` requires an expression).
        return_source_ids : bool, default False
            Also return per-face provenance: ``tag_labels[f]`` is the input
            form of output face ``f``, ``face_labels[f]`` the original face
            id within it.
        return_index_map : bool, default False
            Also return a :class:`MeshIndexMap` with the full output <->
            input identity maps (points and faces, forward and inverse).

        Returns
        -------
        (faces, points) : tuple of np.ndarray
            The result mesh.
        tag_labels, face_labels : np.ndarray
            Only when ``return_source_ids=True``.
        index_map : MeshIndexMap
            Only when ``return_index_map=True``.
        """
        if return_source_ids and return_index_map:
            raise ValueError(
                "return_source_ids and return_index_map are exclusive; the "
                "index map already carries the face labels")
        program = [] if expr is None else _as_expr(expr).program()
        if return_index_map:
            (mesh, ptl, pl, ftl, fl, (pf_off, pf_data), n_op, n_of, n_tags,
             n_out) = self._wrapper.mesh_with_index_map(program)
            return mesh, MeshIndexMap(
                point_tag_labels=ptl, point_labels=pl, face_tag_labels=ftl,
                face_labels=fl,
                point_f=OffsetBlockedArray(pf_off, pf_data),
                n_original_points=n_op, n_original_faces=n_of,
                n_tags=n_tags, n_output_points=n_out)
        if return_source_ids:
            return self._wrapper.mesh_with_labels(program)
        return self._wrapper.mesh(program)

    def domains(
        self,
        expr=None,
        *,
        exclude_outer_shell: bool = True,
        ignore_open_fragments: bool = True,
        return_source_ids: bool = False,
        return_index_map: bool = False,
    ):
        """
        Every kept volumetric domain as its own watertight mesh.

        Parameters
        ----------
        expr : Expr or int, optional
            Restrict to domains inside the expression's selection; with no
            expression, every kept domain of the partition is returned.
        exclude_outer_shell : bool, default True
            Drop the unbounded outside domain.
        ignore_open_fragments : bool, default True
            Fuse open fragments (fins, damage) instead of letting them
            partition volumes.
        return_source_ids : bool, default False
            Also return per-cell face provenance as two OffsetBlockedArrays
            parallel to the cell list.
        return_index_map : bool, default False
            Also return a :class:`DomainsIndexMap` with per-cell face AND
            point identity maps.

        Returns
        -------
        cells : list of (faces, points)
            One mesh per kept domain.
        ids : np.ndarray
            ``ids[k]`` is the coarse domain id of cell ``k``.
        tag_blocks, face_blocks : OffsetBlockedArray
            Only when ``return_source_ids=True``.
        """
        program = [] if expr is None else _as_expr(expr).program()
        config = 0
        if exclude_outer_shell:
            config |= _EXCLUDE_OUTER_SHELL
        if ignore_open_fragments:
            config |= _IGNORE_OPEN_FRAGMENTS
        if return_source_ids and return_index_map:
            raise ValueError(
                "return_source_ids and return_index_map are exclusive; the "
                "index map already carries the face blocks")
        if return_index_map:
            (cells, ids, (ft_off, ft_data), (f_off, f_data),
             (pt_off, pt_data), (p_off, p_data), n_op, n_tags, n_out,
             inclusion) = (
                self._wrapper.domains_with_index_map(program, config)
            )
            return cells, ids, DomainsIndexMap(
                face_tag_blocks=OffsetBlockedArray(ft_off, ft_data),
                face_blocks=OffsetBlockedArray(f_off, f_data),
                point_tag_blocks=OffsetBlockedArray(pt_off, pt_data),
                point_blocks=OffsetBlockedArray(p_off, p_data),
                n_original_points=n_op, n_tags=n_tags, n_output_points=n_out,
                inclusion=np.asarray(inclusion))
        if return_source_ids:
            cells, ids, (t_off, t_data), (f_off, f_data) = (
                self._wrapper.domains_with_labels(program, config)
            )
            return (
                cells,
                ids,
                OffsetBlockedArray(t_off, t_data),
                OffsetBlockedArray(f_off, f_data),
            )
        cells, ids = self._wrapper.domains(program, config)
        return cells, ids

    def __repr__(self):
        return (
            f"tf.CsgGraph(n_forms={len(self._forms)}, sheets={self._sheets}, "
            f"triangulation='{self._triangulation}')"
        )
