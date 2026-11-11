/*
 * Copyright (c) 2026 XLAB
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
#include "../../../core/config/verify.hpp"
#include "../../../exact/meta.hpp"
#include "../../../exact/vertex.hpp"
#include "./tet_orient3d_below.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// How a triangle and a segment meet.
enum class tet_contact : std::uint8_t {
  disjoint,
  share_vertex,
  share_edge,
  touch_edge,
  touch_face,
  across_vertex,
  across_edge,
  across_face
};

/// The contact of a triangle `A B C` and a segment `P Q`, exact: up to two
/// contacts, each a kind and two positions, the first on the triangle — a
/// corner 0, 1, 2 for A, B, C; an edge 0, 1, 2 for A B, B C, C A; 3 for the
/// interior — and the second on the segment — 0 for P, 1 for Q, 0 for the
/// segment itself under an `across` kind. Every orientation of the coplanar
/// case is the exact orientation within the plane, read through the plane's
/// normal's largest axis, and both of its contacts are stated. Three
/// collinear corners meet nothing.
struct tet_triangle_edge_contact {
  tet_contact type[2];
  std::uint8_t pos[4];
};

template <typename Int, typename Coord>
auto classify_tet_triangle_edge_coplanar(const tf::exact::pt3<Coord> &A,
                                         const tf::exact::pt3<Coord> &B,
                                         const tf::exact::pt3<Coord> &C,
                                         const tf::exact::pt3<Coord> &P,
                                         const tf::exact::pt3<Coord> &Q,
                                         tet_triangle_edge_contact &out)
    -> int {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using pt = tf::exact::pt3<Coord>;
  constexpr auto share_vertex = tet_contact::share_vertex;
  constexpr auto share_edge = tet_contact::share_edge;
  constexpr auto touch_edge = tet_contact::touch_edge;
  constexpr auto touch_face = tet_contact::touch_face;
  constexpr auto across_vertex = tet_contact::across_vertex;
  constexpr auto across_edge = tet_contact::across_edge;
  T1 u[3], v[3];
  for (std::size_t k = 0; k < 3; ++k) {
    u[k] = T1(B[k]) - T1(A[k]);
    v[k] = T1(C[k]) - T1(A[k]);
  }
  const T2 n[3] = {T2(u[1]) * v[2] - T2(u[2]) * v[1],
                   T2(u[2]) * v[0] - T2(u[0]) * v[2],
                   T2(u[0]) * v[1] - T2(u[1]) * v[0]};
  std::size_t k = 0;
  for (std::size_t t = 1; t < 3; ++t) {
    const T2 at = n[t] < 0 ? -n[t] : n[t];
    const T2 best = n[k] < 0 ? -n[k] : n[k];
    if (at > best)
      k = t;
  }
  if (n[k] == 0)
    return 0;
  const int nk = n[k] > 0 ? 1 : -1;
  const std::size_t i = (k + 1) % 3, j = (k + 2) % 3;
  const auto lifted = [&](const pt &X, const pt &Y, const pt &Z) {
    const T1 yi = T1(Y[i]) - T1(X[i]), yj = T1(Y[j]) - T1(X[j]);
    const T1 zi = T1(Z[i]) - T1(X[i]), zj = T1(Z[j]) - T1(X[j]);
    const T2 m = T2(yi) * zj - T2(yj) * zi;
    return m > 0 ? nk : m < 0 ? -nk : 0;
  };
  const int sA = lifted(P, Q, A);
  const int sB = lifted(P, Q, B);
  const int sC = lifted(P, Q, C);
  const pt *U[3];
  const pt *V[3];
  int pu[3], pv[3];
  int z1;
  const auto set = [](auto *dst, auto x, auto y, auto z) {
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
  };
  if (sA < 0) {
    if (sB < 0) {
      if (sC < 0)
        return 0;
      set(U, &A, &B, &C);
      set(V, &P, &Q, &P);
      set(pu, 0, 1, 2);
      set(pv, 0, 1, 2);
      z1 = sC > 0 ? 0 : 1;
    } else {
      if (sB > 0) {
        if (sC < 0) {
          set(U, &C, &A, &B);
          set(V, &P, &Q, &P);
          set(pu, 2, 0, 1);
          set(pv, 0, 1, 2);
          z1 = 0;
        } else if (sC > 0) {
          set(U, &B, &C, &A);
          set(V, &Q, &P, &P);
          set(pu, 1, 2, 0);
          set(pv, 1, 0, 2);
          z1 = 0;
        } else {
          set(U, &C, &A, &B);
          set(V, &P, &Q, &P);
          set(pu, 2, 0, 1);
          set(pv, 0, 1, 2);
          z1 = 2;
        }
      } else {
        if (sC < 0) {
          set(U, &C, &A, &B);
          set(V, &P, &Q, &P);
          set(pu, 2, 0, 1);
          set(pv, 0, 1, 2);
          z1 = 1;
        } else if (sC > 0) {
          set(U, &B, &C, &A);
          set(V, &Q, &P, &P);
          set(pu, 1, 2, 0);
          set(pv, 1, 0, 2);
          z1 = 2;
        } else {
          set(U, &B, &C, &A);
          set(V, &Q, &P, &P);
          set(pu, 1, 2, 0);
          set(pv, 1, 0, 2);
          z1 = 3;
        }
      }
    }
  } else if (sA > 0) {
    if (sB < 0) {
      if (sC < 0) {
        set(U, &B, &C, &A);
        set(V, &P, &Q, &P);
        set(pu, 1, 2, 0);
        set(pv, 0, 1, 2);
        z1 = 0;
      } else if (sC > 0) {
        set(U, &C, &A, &B);
        set(V, &Q, &P, &P);
        set(pu, 2, 0, 1);
        set(pv, 1, 0, 2);
        z1 = 0;
      } else {
        set(U, &C, &A, &B);
        set(V, &Q, &P, &P);
        set(pu, 2, 0, 1);
        set(pv, 1, 0, 2);
        z1 = 2;
      }
    } else if (sB > 0) {
      if (sC < 0) {
        set(U, &A, &B, &C);
        set(V, &Q, &P, &P);
        set(pu, 0, 1, 2);
        set(pv, 1, 0, 2);
        z1 = 0;
      } else if (sC > 0) {
        return 0;
      } else {
        set(U, &A, &B, &C);
        set(V, &Q, &P, &P);
        set(pu, 0, 1, 2);
        set(pv, 1, 0, 2);
        z1 = 1;
      }
    } else {
      if (sC < 0) {
        set(U, &B, &C, &A);
        set(V, &P, &Q, &P);
        set(pu, 1, 2, 0);
        set(pv, 0, 1, 2);
        z1 = 2;
      } else if (sC > 0) {
        set(U, &C, &A, &B);
        set(V, &Q, &P, &P);
        set(pu, 2, 0, 1);
        set(pv, 1, 0, 2);
        z1 = 1;
      } else {
        set(U, &B, &C, &A);
        set(V, &P, &Q, &P);
        set(pu, 1, 2, 0);
        set(pv, 0, 1, 2);
        z1 = 3;
      }
    }
  } else {
    if (sB < 0) {
      if (sC < 0) {
        set(U, &B, &C, &A);
        set(V, &P, &Q, &P);
        set(pu, 1, 2, 0);
        set(pv, 0, 1, 2);
        z1 = 1;
      } else if (sC > 0) {
        set(U, &A, &B, &C);
        set(V, &P, &Q, &P);
        set(pu, 0, 1, 2);
        set(pv, 0, 1, 2);
        z1 = 2;
      } else {
        set(U, &C, &A, &B);
        set(V, &Q, &P, &P);
        set(pu, 2, 0, 1);
        set(pv, 1, 0, 2);
        z1 = 3;
      }
    } else if (sB > 0) {
      if (sC < 0) {
        set(U, &A, &B, &C);
        set(V, &Q, &P, &P);
        set(pu, 0, 1, 2);
        set(pv, 1, 0, 2);
        z1 = 2;
      } else if (sC > 0) {
        set(U, &B, &C, &A);
        set(V, &Q, &P, &P);
        set(pu, 1, 2, 0);
        set(pv, 1, 0, 2);
        z1 = 1;
      } else {
        set(U, &C, &A, &B);
        set(V, &P, &Q, &P);
        set(pu, 2, 0, 1);
        set(pv, 0, 1, 2);
        z1 = 3;
      }
    } else {
      if (sC < 0) {
        set(U, &A, &B, &C);
        set(V, &Q, &P, &P);
        set(pu, 0, 1, 2);
        set(pv, 1, 0, 2);
        z1 = 3;
      } else if (sC > 0) {
        set(U, &A, &B, &C);
        set(V, &P, &Q, &P);
        set(pu, 0, 1, 2);
        set(pv, 0, 1, 2);
        z1 = 3;
      } else {
        set(U, &A, &B, &C);
        set(V, &P, &Q, &P);
        set(pu, 0, 1, 2);
        set(pv, 0, 1, 2);
        z1 = 4;
      }
    }
  }
  const int s1 = lifted(*U[0], *U[2], *V[1]);
  const int s2 = lifted(*U[1], *U[2], *V[0]);
  if (s1 > 0)
    return 0;
  if (s2 < 0)
    return 0;
  if constexpr (tf::core::verification)
    tf::core::verify(z1 != 4, "a contact the orientations reach is classified");
  const auto one = [&out](tet_contact t0, int p0, int p1) {
    out.type[0] = t0;
    out.pos[0] = std::uint8_t(p0);
    out.pos[1] = std::uint8_t(p1);
    out.type[1] = tet_contact::disjoint;
  };
  const auto two = [&out](tet_contact t0, int p0, int p1, tet_contact t1,
                          int p2, int p3) {
    out.type[0] = t0;
    out.pos[0] = std::uint8_t(p0);
    out.pos[1] = std::uint8_t(p1);
    out.type[1] = t1;
    out.pos[2] = std::uint8_t(p2);
    out.pos[3] = std::uint8_t(p3);
  };
  if (z1 == 1) {
    if (s1 == 0)
      one(share_vertex, pu[2], pv[1]);
    else if (s2 == 0)
      one(share_vertex, pu[2], pv[0]);
    else
      one(across_vertex, pu[2], pv[0]);
    return 4;
  }
  const int s3 = lifted(*U[0], *U[2], *V[0]);
  const int s4 = lifted(*U[1], *U[2], *V[1]);
  if (z1 == 0) {
    if (s1 < 0) {
      if (s3 > 0) {
        if constexpr (tf::core::verification)
          tf::core::verify(
              s2 > 0, "a negative first orientation makes the second strict");
        if (s4 > 0)
          two(across_edge, pu[2], pv[0], touch_face, 3, pv[1]);
        else if (s4 == 0)
          two(across_edge, pu[2], pv[0], touch_edge, pu[1], pv[1]);
        else
          two(across_edge, pu[2], pv[0], across_edge, pu[1], pv[0]);
      } else if (s3 == 0) {
        if constexpr (tf::core::verification)
          tf::core::verify(
              s2 > 0, "a negative first orientation makes the second strict");
        if (s4 > 0)
          two(touch_edge, pu[2], pv[0], touch_face, 3, pv[1]);
        else if (s4 == 0)
          two(touch_edge, pu[2], pv[0], touch_edge, pu[1], pv[1]);
        else
          two(touch_edge, pu[2], pv[0], across_edge, pu[1], pv[0]);
      } else {
        if (s2 > 0) {
          if (s4 > 0)
            two(touch_face, 3, pv[0], touch_face, 3, pv[1]);
          else if (s4 == 0)
            two(touch_face, 3, pv[0], touch_edge, pu[1], pv[1]);
          else
            two(touch_face, 3, pv[0], across_edge, pu[1], pv[0]);
        } else
          one(touch_edge, pu[1], pv[0]);
      }
    } else
      one(touch_edge, pu[2], pv[1]);
  } else if (z1 == 2) {
    if (s1 < 0) {
      if (s3 > 0) {
        if constexpr (tf::core::verification)
          tf::core::verify(
              s2 > 0, "a negative first orientation makes the second strict");
        if (s4 > 0)
          two(across_vertex, pu[0], pv[0], touch_face, 3, pv[1]);
        else if (s4 == 0)
          two(across_vertex, pu[0], pv[0], touch_edge, pu[1], pv[1]);
        else
          two(across_vertex, pu[0], pv[0], across_edge, pu[1], pv[0]);
      } else if (s3 == 0) {
        if constexpr (tf::core::verification)
          tf::core::verify(
              s2 > 0, "a negative first orientation makes the second strict");
        if (s4 > 0)
          two(share_vertex, pu[0], pv[0], touch_face, 3, pv[1]);
        else if (s4 == 0)
          two(share_vertex, pu[0], pv[0], touch_edge, pu[1], pv[1]);
        else
          two(share_vertex, pu[0], pv[0], across_edge, pu[1], pv[0]);
      } else {
        if (s2 > 0) {
          if (s4 > 0)
            two(touch_face, 3, pv[0], touch_face, 3, pv[1]);
          else if (s4 == 0)
            two(touch_face, 3, pv[0], touch_edge, pu[1], pv[1]);
          else
            two(touch_face, 3, pv[0], across_edge, pu[1], pv[0]);
        } else
          one(touch_edge, pu[1], pv[0]);
      }
    } else
      one(share_vertex, pu[0], pv[1]);
  } else {
    if (s1 < 0) {
      if (s3 > 0) {
        if constexpr (tf::core::verification)
          tf::core::verify(
              s2 > 0, "a negative first orientation makes the second strict");
        if (s4 > 0)
          two(across_vertex, pu[0], pv[0], touch_edge, pu[0], pv[1]);
        else if (s4 == 0)
          two(across_vertex, pu[0], pv[0], share_vertex, pu[1], pv[1]);
        else
          two(across_vertex, pu[0], pv[0], across_vertex, pu[1], pv[0]);
      } else if (s3 == 0) {
        if constexpr (tf::core::verification)
          tf::core::verify(
              s2 > 0, "a negative first orientation makes the second strict");
        if (s4 > 0)
          two(share_vertex, pu[0], pv[0], touch_edge, pu[0], pv[1]);
        else if (s4 == 0)
          one(share_edge, pu[0], pv[0]);
        else
          two(share_vertex, pu[0], pv[0], across_vertex, pu[1], pv[0]);
      } else {
        if (s2 > 0) {
          if (s4 > 0)
            two(touch_edge, pu[0], pv[0], touch_edge, pu[0], pv[1]);
          else if (s4 == 0)
            two(touch_edge, pu[0], pv[0], share_vertex, pu[1], pv[1]);
          else
            two(touch_edge, pu[0], pv[0], across_vertex, pu[1], pv[0]);
        } else
          one(share_vertex, pu[1], pv[0]);
      }
    } else
      one(share_vertex, pu[0], pv[1]);
  }
  return 4;
}

/// The contact of `A B C` and `P Q`: 0 disjoint, 2 one point off the
/// plane, 4 coplanar (one point or a segment, the second contact saying
/// which).
template <typename Int, typename Coord>
auto classify_tet_triangle_edge(const tf::exact::pt3<Coord> &A,
                                const tf::exact::pt3<Coord> &B,
                                const tf::exact::pt3<Coord> &C,
                                const tf::exact::pt3<Coord> &P,
                                const tf::exact::pt3<Coord> &Q,
                                tet_triangle_edge_contact &out) -> int {
  using pt = tf::exact::pt3<Coord>;
  const auto below = [](const pt &a, const pt &b, const pt &c, const pt &d) {
    return tet_orient3d_below<Int>(a, b, c, d);
  };
  const int sP = below(A, B, C, P);
  const int sQ = below(A, B, C, Q);
  const pt *U[3];
  const pt *V[2];
  int pu[3], pv[2];
  int z1;
  if (sP < 0) {
    if (sQ < 0)
      return 0;
    U[0] = &A, U[1] = &B, U[2] = &C;
    V[0] = &P, V[1] = &Q;
    pu[0] = 0, pu[1] = 1, pu[2] = 2;
    pv[0] = 0, pv[1] = 1;
    z1 = sQ > 0 ? 0 : 1;
  } else if (sP > 0) {
    if (sQ < 0) {
      U[0] = &A, U[1] = &B, U[2] = &C;
      V[0] = &Q, V[1] = &P;
      pu[0] = 0, pu[1] = 1, pu[2] = 2;
      pv[0] = 1, pv[1] = 0;
      z1 = 0;
    } else if (sQ > 0)
      return 0;
    else {
      U[0] = &B, U[1] = &A, U[2] = &C;
      V[0] = &P, V[1] = &Q;
      pu[0] = 1, pu[1] = 0, pu[2] = 2;
      pv[0] = 0, pv[1] = 1;
      z1 = 1;
    }
  } else {
    if (sQ < 0) {
      U[0] = &A, U[1] = &B, U[2] = &C;
      V[0] = &Q, V[1] = &P;
      pu[0] = 0, pu[1] = 1, pu[2] = 2;
      pv[0] = 1, pv[1] = 0;
      z1 = 1;
    } else if (sQ > 0) {
      U[0] = &B, U[1] = &A, U[2] = &C;
      V[0] = &Q, V[1] = &P;
      pu[0] = 1, pu[1] = 0, pu[2] = 2;
      pv[0] = 1, pv[1] = 0;
      z1 = 1;
    } else
      return classify_tet_triangle_edge_coplanar<Int>(A, B, C, P, Q, out);
  }
  const int s1 = below(*U[0], *U[1], *V[0], *V[1]);
  if (s1 < 0)
    return 0;
  const int s2 = below(*U[1], *U[2], *V[0], *V[1]);
  if (s2 < 0)
    return 0;
  const int s3 = below(*U[2], *U[0], *V[0], *V[1]);
  if (s3 < 0)
    return 0;
  out.type[1] = tet_contact::disjoint;
  const auto one = [&out](tet_contact t0, int p0, int p1) {
    out.type[0] = t0;
    out.pos[0] = std::uint8_t(p0);
    out.pos[1] = std::uint8_t(p1);
  };
  if (z1 == 0) {
    if (s1 > 0) {
      if (s2 > 0) {
        if (s3 > 0)
          one(tet_contact::across_face, 3, 0);
        else
          one(tet_contact::across_edge, pu[2], 0);
      } else {
        if (s3 > 0)
          one(tet_contact::across_edge, pu[1], 0);
        else
          one(tet_contact::across_vertex, pu[2], 0);
      }
    } else {
      if (s2 > 0) {
        if (s3 > 0)
          one(tet_contact::across_edge, pu[0], 0);
        else
          one(tet_contact::across_vertex, pu[0], 0);
      } else {
        if constexpr (tf::core::verification)
          tf::core::verify(s3 > 0,
                           "a zero second orientation makes the third strict");
        one(tet_contact::across_vertex, pu[1], 0);
      }
    }
  } else {
    if (s1 > 0) {
      if (s2 > 0) {
        if (s3 > 0)
          one(tet_contact::touch_face, 0, pv[1]);
        else
          one(tet_contact::touch_edge, pu[2], pv[1]);
      } else {
        if (s3 > 0)
          one(tet_contact::touch_edge, pu[1], pv[1]);
        else
          one(tet_contact::share_vertex, pu[2], pv[1]);
      }
    } else {
      if (s2 > 0) {
        if (s3 > 0)
          one(tet_contact::touch_edge, pu[0], pv[1]);
        else
          one(tet_contact::share_vertex, pu[0], pv[1]);
      } else {
        if constexpr (tf::core::verification)
          tf::core::verify(s3 > 0,
                           "a zero second orientation makes the third strict");
        one(tet_contact::share_vertex, pu[1], pv[1]);
      }
    }
  }
  return 2;
}

} // namespace tf::topology::cdt::dt3
