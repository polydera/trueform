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
_WITHIN = 24  # self_intersections | resolve_self_crossing_contours

_TRIANGULATION_MAP = {"cdt": 0, "refined_cdt": 1}

_EXCLUDE_OUTER_SHELL = 1
_IGNORE_OPEN_FRAGMENTS = 2

_SELECTION_BOUNDARY = 0
_SELECTION_INSIDE = 1


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
        One or more closed 3D meshes, all triangle or all dynamic (a
        mixed set needs one representation before the build). All must
        share index and real dtypes. A single mesh is
        its own self arrangement: its self-intersections are the whole
        cut, and ``domains()`` classifies its overlap pockets.
    sheets : list of int, optional
        Indices of operands declared as oriented open sheets: they cut
        volumes through the same algebra without enclosing one.
    mode : str, default "primitives"
        Intersection mode. "sos" or "primitives".
    tolerance : float, default 0.0
        World-coordinate distance an input vertex may move to reach the lattice
        (0 = exact).
    resolve_crossings : bool, default True
        Resolve crossings between different contours on the same face.
    within : bool, default False
        Also intersect each operand with itself. Required when an operand
        can self-overlap, e.g. meshes concatenated into one operand;
        ``domains()`` then classifies the overlap pockets structurally.
        Boolean expressions still require solid, non-self-overlapping
        operands.
    triangulation : str, default "cdt"
        Cut-surface triangulation: "cdt" (plain constrained Delaunay per
        cut loop) or "refined_cdt" (quality refinement of the cut
        surface; shared boundaries stay watertight by construction).

    Examples
    --------
    >>> graph = tf.CsgGraph([a, b, c], triangulation="refined_cdt")
    >>> faces_points = graph.mesh(tf.op(0) - tf.op(1))
    >>> imprinted = graph.mesh(selection=[0])   # a's surface, cut by b, c
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
        within: bool = False,
        triangulation: str = "cdt",
    ):
        if not meshes:
            raise ValueError("CsgGraph needs at least one mesh")
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
            if not m.is_dynamic and m.ngon != 3:
                raise ValueError(
                    "CsgGraph takes triangle or dynamic meshes; call "
                    "triangulated() on fixed n-gon meshes first"
                )
            if m_meta.ngon != meta.ngon:
                raise ValueError(
                    "CsgGraph operands must be all triangle or all dynamic; "
                    "convert the triangle operands to dynamic faces (or "
                    "triangulated() the dynamic ones) first"
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
        if within:
            mode_int |= _WITHIN

        self._forms = list(meshes)
        self._is_dynamic = meshes[0].is_dynamic
        self._sheets = tuple(sheet_list)
        self._mode = mode
        self._tolerance = tolerance
        self._resolve_crossings = resolve_crossings
        self._within = within
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

    def _wrap_mesh(self, mesh_pair):
        """Native dynamic faces cross as (offsets, data); rewrap them."""
        if not self._is_dynamic:
            return mesh_pair
        (offsets, data), points = mesh_pair
        return OffsetBlockedArray(offsets, data), points

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
        """Input placement tolerance this graph was built with."""
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

    def outer_shell(self):
        """
        The outer shell of the arrangement: the boundary between the
        unbounded universe and everything the operands enclose, oriented
        outward.

        Structural read — no expression, no options. Internal structure
        (overlap membranes, buried faces, enclosed cavities) has the same
        domain on both sides and never reaches the boundary; open
        fragments bound no volume and cannot survive into the shell.

        Returns
        -------
        (faces, points) : tuple
            The shell mesh, in the graph's index and real dtypes;
            ``faces`` is an ndarray for a triangle graph, an
            :class:`OffsetBlockedArray` for a dynamic one.
        """
        return self._wrap_mesh(self._wrapper.outer_shell())

    # -- queries ----------------------------------------------------------
    def _selection_tags(self, selection):
        """Validate a surface restriction into a plain list of operands."""
        if selection is None:
            return None
        tags = [int(t) for t in selection]
        for t in tags:
            if t < 0 or t >= len(self._forms):
                raise ValueError(f"selection index {t} out of range")
        return tags

    def _restriction(self, selection, inside, has_expr):
        """Validate the surface restriction into (tags or None, kind)."""
        if selection is not None and inside is not None:
            raise ValueError("selection and inside are exclusive")
        if inside is None:
            return self._selection_tags(selection), _SELECTION_BOUNDARY
        if not has_expr:
            raise ValueError("inside requires an expression")
        return self._selection_tags(inside), _SELECTION_INSIDE

    def mesh(self, expr=None, *, selection=None, inside=None,
             return_source_ids: bool = False,
             return_index_map: bool = False):
        """
        The boolean result mesh for ``expr``; with no expression, the full
        arrangement mesh (every input face, cut at intersections).

        Parameters
        ----------
        expr : Expr or int, optional
            Boolean expression over operand indices (``tf.op(0) - tf.op(1)``),
            or a single operand index. Omit for the full arrangement mesh
            (``return_index_map`` requires an expression or a selection).
        selection : sequence of int, optional
            Restrict the emitted surface to the faces these operands
            contributed; ``None`` (default) emits every operand's. With an
            expression it splits one boolean result by provenance; without
            one it is the embedded read — the named operands' surfaces cut
            by everything, original winding, no classification.
        inside : sequence of int, optional
            Emit the faces these operands contributed that lie INSIDE the
            expression's region — both sides in it — with the operand's
            stored winding, where ``selection`` emits the faces bounding
            it. Requires ``expr``; exclusive with ``selection``. A sheet's
            surface inside a solid: ``graph.mesh(tf.op(solid),
            inside=[sheet])``.
        return_source_ids : bool, default False
            Also return per-face provenance: ``tag_labels[f]`` is the input
            form of output face ``f``, ``face_labels[f]`` the original face
            id within it.
        return_index_map : bool, default False
            Also return a :class:`MeshIndexMap` with the full output <->
            input identity maps (points and faces, forward and inverse).

        Returns
        -------
        (faces, points) : tuple
            The result mesh; ``faces`` is an ndarray for a triangle
            graph, an :class:`OffsetBlockedArray` for a dynamic one.
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
        tags, kind = self._restriction(selection, inside, expr is not None)
        if return_index_map:
            (mesh, ptl, pl, ftl, fl, (pf_off, pf_data), uncut, n_op, n_tags,
             n_out) = self._wrapper.mesh_with_index_map(program, tags, kind)
            return self._wrap_mesh(mesh), MeshIndexMap(
                point_tag_labels=ptl, point_labels=pl, face_tag_labels=ftl,
                face_labels=fl,
                point_f=OffsetBlockedArray(pf_off, pf_data),
                uncut_faces=uncut,
                n_original_points=n_op,
                n_tags=n_tags, n_output_points=n_out)
        if return_source_ids:
            mesh, tag_labels, face_labels = self._wrapper.mesh_with_labels(
                program, tags, kind)
            return self._wrap_mesh(mesh), tag_labels, face_labels
        return self._wrap_mesh(self._wrapper.mesh(program, tags, kind))

    def domains(
        self,
        expr=None,
        *,
        selection=None,
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
        selection : sequence of int, optional
            Restrict each cell's walls to the faces these operands
            contributed; ``None`` (default) emits every operand's. The kept
            domains are unchanged — only whose surface reaches the output.
        exclude_outer_shell : bool, default True
            Drop every domain no operand claims — the unbounded outside
            among them.
        ignore_open_fragments : bool, default True
            Fuse the open regions a sheet's dangling parts bound back into
            their surroundings. Sheets only: a volume's open fragments are
            always fused, before any read.
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
        tags = self._selection_tags(selection)
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
                self._wrapper.domains_with_index_map(program, config, tags)
            )
            return [self._wrap_mesh(c) for c in cells], ids, DomainsIndexMap(
                face_tag_blocks=OffsetBlockedArray(ft_off, ft_data),
                face_blocks=OffsetBlockedArray(f_off, f_data),
                point_tag_blocks=OffsetBlockedArray(pt_off, pt_data),
                point_blocks=OffsetBlockedArray(p_off, p_data),
                n_original_points=n_op, n_tags=n_tags, n_output_points=n_out,
                inclusion=np.asarray(inclusion))
        if return_source_ids:
            cells, ids, (t_off, t_data), (f_off, f_data) = (
                self._wrapper.domains_with_labels(program, config, tags)
            )
            return (
                [self._wrap_mesh(c) for c in cells],
                ids,
                OffsetBlockedArray(t_off, t_data),
                OffsetBlockedArray(f_off, f_data),
            )
        cells, ids = self._wrapper.domains(program, config, tags)
        return [self._wrap_mesh(c) for c in cells], ids

    def __repr__(self):
        return (
            f"tf.CsgGraph(n_forms={len(self._forms)}, sheets={self._sheets}, "
            f"triangulation='{self._triangulation}')"
        )
