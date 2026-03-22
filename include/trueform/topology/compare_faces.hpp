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

#include "../core/static_size.hpp"
#include <type_traits>

namespace tf {

/// @ingroup topology_analysis
/// @brief Compare two faces as cyclic vertex sequences.
///
/// Returns:
///   +1 if equal with same winding (aligned)
///   -1 if equal with reversed winding (opposing)
///    0 if not equal
///
/// Faces are cyclic sequences. {0,1,2} and {1,2,0} are aligned (+1).
/// {0,1,2} and {0,2,1} are opposing (-1).
template <typename Face1, typename Face2>
auto compare_faces(const Face1 &face, const Face2 &neighbor) -> int {
  constexpr auto N1 = tf::static_size_v<Face1>;
  constexpr auto N2 = tf::static_size_v<Face2>;
  using vertex_t = std::decay_t<decltype(face[0])>;

  if constexpr (N1 == 3 && N2 == 3) {
    vertex_t v0 = face[0], v1 = face[1], v2 = face[2];
    vertex_t n0 = neighbor[0], n1 = neighbor[1], n2 = neighbor[2];
    // Aligned
    if ((v0 == n0 && v1 == n1 && v2 == n2) ||
        (v0 == n1 && v1 == n2 && v2 == n0) ||
        (v0 == n2 && v1 == n0 && v2 == n1))
      return 1;
    // Opposing
    if ((v0 == n0 && v1 == n2 && v2 == n1) ||
        (v0 == n1 && v1 == n0 && v2 == n2) ||
        (v0 == n2 && v1 == n1 && v2 == n0))
      return -1;
    return 0;
  } else {
    auto N = face.size();
    if (static_cast<decltype(N)>(neighbor.size()) != N)
      return 0;

    // Find face[0] in neighbor
    auto start = N;
    for (decltype(N) i = 0; i < N; ++i) {
      if (neighbor[i] == face[0]) {
        start = i;
        break;
      }
    }
    if (start == N)
      return 0;

    // Check forward (aligned)
    if (neighbor[(start + 1) % N] == face[1]) {
      for (decltype(N) k = 2; k < N; ++k) {
        if (face[k] != neighbor[(start + k) % N])
          return 0;
      }
      return 1;
    }

    // Check backward (opposing)
    if (neighbor[(start + N - 1) % N] == face[1]) {
      for (decltype(N) k = 2; k < N; ++k) {
        if (face[k] != neighbor[(start + N - k) % N])
          return 0;
      }
      return -1;
    }

    return 0;
  }
}

} // namespace tf
