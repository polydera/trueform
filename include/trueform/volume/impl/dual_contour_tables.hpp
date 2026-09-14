/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once
#include "./flying_edges_tables.hpp"
#include <array>
#include <cstddef>

// The patch topology of a marching-cubes case: which crossing edges each
// surface component owns, the cyclic boundary link of that component, and the
// contour arcs its boundary draws on each cell face.
//
// A dual edge is named by the pair of patches it joins. Two neighbouring cells
// share exactly one face, and that face may carry TWO arcs — two distinct dual
// edges between the same pair of patches. Nothing but the arc distinguishes
// them, so the arc has to be an identity before anything is emitted.
//
// Everything here is derived from the one marching-cubes case authority
// (`fe_tables`), once at first use: the derivation is too long to evaluate in
// every translation unit that includes it, and past MSVC's default constexpr
// budget. Signs and the table decide, positions never do.

namespace tf {
namespace volume_detail {

// The cell's corner and edge numbering is the extractors' shared one.
inline constexpr auto &k_corner_offset = fe_corner_offset;
inline constexpr auto &k_cell_edge = fe_edge_corner;

// A face of the cell is f = axis * 2 + side, side 0 low and 1 high. Its four
// edges are listed in the CANONICAL ordinal order the two cells sharing that
// face agree on: the two edges along the first spanned axis (low then high on
// the second), then the two along the second axis.
constexpr std::array<std::array<int, 4>, 6> k_face_edges = {{
    {4, 6, 8, 10},  // x low
    {5, 7, 9, 11},  // x high
    {0, 2, 8, 9},   // y low
    {1, 3, 10, 11}, // y high
    {0, 1, 4, 5},   // z low
    {2, 3, 6, 7},   // z high
}};

// The four corners of each face, in the order the two sharing cells agree on.
constexpr std::array<std::array<int, 4>, 6> k_face_corners = {{
    {0, 2, 4, 6}, // x low
    {1, 3, 5, 7}, // x high
    {0, 1, 4, 5}, // y low
    {2, 3, 6, 7}, // y high
    {0, 1, 2, 3}, // z low
    {4, 5, 6, 7}, // z high
}};

constexpr auto face_ordinal_of(int face, int edge) -> int {
  for (int i = 0; i < 4; ++i)
    if (k_face_edges[std::size_t(face)][std::size_t(i)] == edge)
      return i;
  return -1;
}

// One vertex per surface COMPONENT within a cell (Nielson): edges are grouped
// by connectivity in the MC triangle table, so the output is the dual of
// consistent marching cubes.
struct component_tables {
  std::array<std::array<unsigned char, 12>, 256> comp{};
  std::array<unsigned char, 256> count{};
};

inline auto make_component_tables() -> component_tables {
  component_tables t{};
  for (std::size_t cs = 0; cs < 256; ++cs) {
    const auto &tc = fe_tables().edge_cases[cs];
    int parent[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    bool used[12] = {};
    const int tri_n = tc[0];
    for (int i = 0; i < tri_n; ++i) {
      const int e[3] = {tc[1 + 3 * i], tc[2 + 3 * i], tc[3 + 3 * i]};
      for (int k = 0; k < 3; ++k)
        used[e[k]] = true;
      for (int k = 1; k < 3; ++k) {
        int a = e[0], b = e[k];
        while (parent[a] != a)
          a = parent[a];
        while (parent[b] != b)
          b = parent[b];
        if (a != b)
          parent[b] = a;
      }
    }
    int comp_of_root[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    unsigned char next = 0;
    for (int e = 0; e < 12; ++e) {
      t.comp[cs][e] = 0xff;
      if (!used[e])
        continue;
      int r = e;
      while (parent[r] != r)
        r = parent[r];
      if (comp_of_root[r] < 0)
        comp_of_root[r] = next++;
      t.comp[cs][e] = static_cast<unsigned char>(comp_of_root[r]);
    }
    t.count[cs] = next;
  }
  return t;
}

// Both tables are derived from the case authority at first use, for the same
// reason it is.
inline auto components() -> const component_tables & {
  static const component_tables tables = make_component_tables();
  return tables;
}

constexpr int k_max_components = 4;

// The boundary link of every component and the arcs it draws on every face.
struct patch_tables {
  // the component's boundary cycle, as crossing edge ids in cyclic order
  std::array<std::array<std::array<unsigned char, 12>, k_max_components>, 256>
      link{};
  std::array<std::array<unsigned char, k_max_components>, 256> link_len{};
  // per face, up to two arcs: the canonical ordinal pair (low << 2 | high)
  // and the component whose boundary draws it
  std::array<std::array<std::array<unsigned char, 2>, 6>, 256> arc_pair{};
  std::array<std::array<std::array<unsigned char, 2>, 6>, 256> arc_comp{};
  std::array<std::array<unsigned char, 6>, 256> arc_count{};
  // set when a case's patches are not all disks, which the audit must reject
  bool malformed = false;
};

inline auto make_patch_tables() -> patch_tables {
  patch_tables t{};
  for (std::size_t cs = 0; cs < 256; ++cs) {
    const auto &tc = fe_tables().edge_cases[cs];
    const int tri_n = tc[0];
    for (int comp = 0; comp < components().count[cs]; ++comp) {
      // the patch boundary: directed triangle sides that no reversed side
      // cancels
      bool side[12][12] = {};
      for (int i = 0; i < tri_n; ++i) {
        const int e[3] = {tc[1 + 3 * i], tc[2 + 3 * i], tc[3 + 3 * i]};
        if (components().comp[cs][e[0]] != comp)
          continue;
        for (int k = 0; k < 3; ++k) {
          const int a = e[k], b = e[(k + 1) % 3];
          if (side[b][a])
            side[b][a] = false;
          else
            side[a][b] = true;
        }
      }
      int start = -1;
      for (int a = 0; a < 12 && start < 0; ++a)
        for (int b = 0; b < 12; ++b)
          if (side[a][b]) {
            start = a;
            break;
          }
      if (start < 0)
        continue;
      // one outgoing side per boundary crossing is what makes the walk a cycle
      int len = 0;
      int a = start;
      for (int guard = 0; guard < 12; ++guard) {
        int next = -1;
        for (int b = 0; b < 12; ++b)
          if (side[a][b]) {
            if (next >= 0)
              t.malformed = true;
            next = b;
          }
        if (next < 0) {
          t.malformed = true;
          break;
        }
        t.link[cs][std::size_t(comp)][std::size_t(len++)] =
            static_cast<unsigned char>(a);
        side[a][next] = false;
        a = next;
        if (a == start)
          break;
      }
      t.link_len[cs][std::size_t(comp)] = static_cast<unsigned char>(len);
      for (int b = 0; b < 12; ++b)
        for (int d = 0; d < 12; ++d)
          if (side[b][d])
            t.malformed = true; // a second boundary cycle: not a disk

      // each consecutive pair of the link shares one face: that is an arc
      for (int i = 0; i < len; ++i) {
        const int e0 = t.link[cs][std::size_t(comp)][std::size_t(i)];
        const int e1 =
            t.link[cs][std::size_t(comp)][std::size_t((i + 1) % len)];
        int face = -1;
        for (int f = 0; f < 6; ++f) {
          const int o0 = face_ordinal_of(f, e0);
          const int o1 = face_ordinal_of(f, e1);
          if (o0 >= 0 && o1 >= 0) {
            if (face >= 0)
              t.malformed = true;
            face = f;
          }
        }
        if (face < 0) {
          t.malformed = true;
          continue;
        }
        int o0 = face_ordinal_of(face, e0);
        int o1 = face_ordinal_of(face, e1);
        if (o0 > o1) {
          const int tmp = o0;
          o0 = o1;
          o1 = tmp;
        }
        auto &count = t.arc_count[cs][std::size_t(face)];
        if (count >= 2) {
          t.malformed = true;
          continue;
        }
        t.arc_pair[cs][std::size_t(face)][count] =
            static_cast<unsigned char>((o0 << 2) | o1);
        t.arc_comp[cs][std::size_t(face)][count] =
            static_cast<unsigned char>(comp);
        ++count;
      }
    }
    // canonical arc order, so both cells sharing a face enumerate alike
    for (int f = 0; f < 6; ++f)
      if (t.arc_count[cs][std::size_t(f)] == 2 &&
          t.arc_pair[cs][std::size_t(f)][0] >
              t.arc_pair[cs][std::size_t(f)][1]) {
        const auto p = t.arc_pair[cs][std::size_t(f)][0];
        t.arc_pair[cs][std::size_t(f)][0] = t.arc_pair[cs][std::size_t(f)][1];
        t.arc_pair[cs][std::size_t(f)][1] = p;
        const auto c = t.arc_comp[cs][std::size_t(f)][0];
        t.arc_comp[cs][std::size_t(f)][0] = t.arc_comp[cs][std::size_t(f)][1];
        t.arc_comp[cs][std::size_t(f)][1] = c;
      }
  }
  return t;
}

inline auto patches() -> const patch_tables & {
  static const patch_tables tables = make_patch_tables();
  return tables;
}

/// @brief Which arc of @p face carries crossing ordinal @p ordinal, or -1.
inline auto arc_of_ordinal(unsigned char cs, int face, int ordinal) -> int {
  for (int i = 0; i < patches().arc_count[cs][std::size_t(face)]; ++i) {
    const auto p = patches().arc_pair[cs][std::size_t(face)][std::size_t(i)];
    if ((p >> 2) == ordinal || (p & 3) == ordinal)
      return i;
  }
  return -1;
}

/// @brief Which of a cell's 12 crossing edges own a dual quad: the quad needs
/// all four cells around its grid edge to exist, so the domain boundary layer
/// retains fewer than all twelve.
constexpr auto retained_edges(int x, int y, int z, int cx, int cy, int cz)
    -> unsigned {
  unsigned mask = 0;
  const int g0[3] = {x, y, z};
  const int n[3] = {cx, cy, cz};
  for (int e = 0; e < 12; ++e) {
    const auto &a =
        k_corner_offset[std::size_t(k_cell_edge[std::size_t(e)][0])];
    const auto &b =
        k_corner_offset[std::size_t(k_cell_edge[std::size_t(e)][1])];
    int axis = 0;
    for (int d = 0; d < 3; ++d)
      if (a[std::size_t(d)] != b[std::size_t(d)])
        axis = d;
    bool ok = true;
    for (int d = 0; d < 3; ++d) {
      if (d == axis)
        continue;
      const int g = g0[d] + a[std::size_t(d)];
      ok = ok && g >= 1 && g <= n[d] - 1;
    }
    if (ok)
      mask |= 1u << e;
  }
  return mask;
}

constexpr unsigned k_all_edges = 0xfffu;

// Removing quads cuts a component's boundary link into intervals; each
// surviving interval is its own output vertex, so that a cell whose link is cut
// does not pinch two sheets through one identity.
inline auto link_position(unsigned char cs, int comp, int edge) -> int {
  const int len = patches().link_len[cs][std::size_t(comp)];
  for (int i = 0; i < len; ++i)
    if (patches().link[cs][std::size_t(comp)][std::size_t(i)] == edge)
      return i;
  return -1;
}

inline auto interval_starts(unsigned char cs, int comp, unsigned retained,
                            int *starts) -> int {
  const int len = patches().link_len[cs][std::size_t(comp)];
  int n = 0, present = 0;
  for (int i = 0; i < len; ++i)
    present +=
        (retained >> patches().link[cs][std::size_t(comp)][std::size_t(i)]) &
        1u;
  if (present == 0)
    return 0;
  if (present == len) {
    starts[0] = 0; // a whole cycle: one interval, starting anywhere
    return 1;
  }
  for (int i = 0; i < len; ++i) {
    const int e = patches().link[cs][std::size_t(comp)][std::size_t(i)];
    const int p =
        patches().link[cs][std::size_t(comp)][std::size_t((i + len - 1) % len)];
    if (((retained >> e) & 1u) && !((retained >> p) & 1u))
      starts[n++] = i;
  }
  return n;
}

inline auto interval_count(unsigned char cs, int comp, unsigned retained)
    -> int {
  int starts[12] = {};
  return interval_starts(cs, comp, retained, starts);
}

/// @brief Output vertices a cell owns: one per component, or one per surviving
/// interval of a component whose link the domain boundary cut.
inline auto cell_vertex_count(unsigned char cs, unsigned retained) -> int {
  if (retained == k_all_edges)
    return components().count[cs];
  int n = 0;
  for (int comp = 0; comp < components().count[cs]; ++comp)
    n += interval_count(cs, comp, retained);
  return n;
}

/// @brief First cell-local output index owned by a component.
inline auto component_id_base(unsigned char cs, unsigned retained, int comp)
    -> int {
  if (retained == k_all_edges)
    return comp;
  int base = 0;
  for (int c = 0; c < comp; ++c)
    base += interval_count(cs, c, retained);
  return base;
}

/// @brief How many output indices a component owns: one, or one per surviving
/// interval of a link the domain boundary cut.
inline auto component_id_copies(unsigned char cs, unsigned retained, int comp)
    -> int {
  return retained == k_all_edges ? 1 : interval_count(cs, comp, retained);
}

/// @brief The cell-local output index of the vertex a quad on crossing @p edge
/// reads: the component, or the interval of that component containing the edge.
inline auto cell_vertex_index(unsigned char cs, unsigned retained, int edge)
    -> int {
  const int comp = components().comp[cs][std::size_t(edge)];
  if (retained == k_all_edges)
    return comp;
  int base = 0;
  for (int c = 0; c < comp; ++c)
    base += interval_count(cs, c, retained);
  int starts[12] = {};
  const int n = interval_starts(cs, comp, retained, starts);
  if (n <= 1)
    return base;
  const int len = patches().link_len[cs][std::size_t(comp)];
  int at = link_position(cs, comp, edge);
  if (at < 0)
    return base;
  // walk back to this interval's start, then order intervals by that start's
  // edge id so every reader of the cell agrees
  for (int guard = 0; guard < len; ++guard) {
    const int p = (at + len - 1) % len;
    if (!((retained >> patches().link[cs][std::size_t(comp)][std::size_t(p)]) &
          1u))
      break;
    at = p;
  }
  const int my = patches().link[cs][std::size_t(comp)][std::size_t(at)];
  int rank = 0;
  for (int i = 0; i < n; ++i) {
    const int s = patches().link[cs][std::size_t(comp)][std::size_t(starts[i])];
    if (s < my)
      ++rank;
  }
  return base + rank;
}

// A dual quad is owned by a crossing grid edge and walks the four cells around
// it. Side s joins its corners s and (s+1)%4; these state, per quad axis, the
// face axis that side crosses, the owned edge's canonical ordinal on that
// face, and which of the four corner cells is the lower cell that owns it.
struct quad_side {
  unsigned char face_axis;
  unsigned char ordinal;
  unsigned char owner_corner;
};

constexpr std::array<std::array<quad_side, 4>, 3> k_quad_sides = {{
    // x quad: cells (y-1,z-1), (y,z-1), (y,z), (y-1,z)
    {{{1, 1, 0}, {2, 0, 1}, {1, 0, 3}, {2, 1, 0}}},
    // y quad: cells (x-1,z-1), (x-1,z), (x,z), (x,z-1)
    {{{2, 3, 0}, {0, 0, 1}, {2, 2, 3}, {0, 1, 0}}},
    // z quad: cells (x-1,y-1), (x,y-1), (x,y), (x-1,y)
    {{{0, 3, 0}, {1, 2, 1}, {0, 2, 3}, {1, 3, 0}}},
}};

// Where each quad corner reads its cell: the row walked in lockstep (0 is
// (y-1,z-1), 1 is (y,z-1), 2 is (y-1,z), 3 is (y,z)), the x slot (0 is x-1,
// 1 is x) and the grid edge's id local to that cell.
struct quad_corner {
  unsigned char row;
  unsigned char slot;
  unsigned char edge;
};

constexpr std::array<std::array<quad_corner, 4>, 3> k_quad_corners = {{
    {{{0, 1, 3}, {1, 1, 2}, {3, 1, 0}, {2, 1, 1}}},   // x quad
    {{{1, 0, 7}, {3, 0, 5}, {3, 1, 4}, {1, 1, 6}}},   // y quad
    {{{2, 0, 11}, {2, 1, 10}, {3, 1, 8}, {3, 0, 9}}}, // z quad
}};

} // namespace volume_detail
} // namespace tf
