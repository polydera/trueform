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
/// @brief Checks if two faces are equal with the same winding order.
///
/// Faces are cyclic sequences of vertex indices. Two faces are considered
/// equal with same orientation if they have identical vertices in the same
/// cyclic order (possibly rotated). For example, {0,1,2} equals {1,2,0}
/// but not {0,2,1}.
///
/// @tparam Face1 The first face type.
/// @tparam Face2 The second face type.
/// @param face The first face.
/// @param neighbor The second face.
/// @return True if faces have identical vertices in the same cyclic order.
/// @see tf::are_faces_equal() for orientation-independent comparison.
template <typename Face1, typename Face2>
auto are_oriented_faces_equal(const Face1 &face, const Face2 &neighbor) -> bool {
  constexpr auto N1 = tf::static_size_v<Face1>;
  constexpr auto N2 = tf::static_size_v<Face2>;
  using Index = std::decay_t<decltype(face[0])>;

  if constexpr (N1 == 3 && N2 == 3) {
    // Fast path for triangles
    return face[0] == neighbor[0] ? (face[1] == neighbor[1] && face[2] == neighbor[2])
         : face[0] == neighbor[1] ? (face[1] == neighbor[2] && face[2] == neighbor[0])
         : face[0] == neighbor[2] ? (face[1] == neighbor[0] && face[2] == neighbor[1])
         : false;
  } else {
    Index N = face.size();
    if (static_cast<Index>(neighbor.size()) != N)
      return false;

    // Find face[0] in neighbor
    Index start = N;
    for (Index i = 0; i < N; ++i) {
      if (neighbor[i] == face[0]) {
        start = i;
        break;
      }
    }
    if (start == N)
      return false;

    // Check forward match
    for (Index k = 1; k < N; ++k) {
      if (face[k] != neighbor[(start + k) % N])
        return false;
    }
    return true;
  }
}
} // namespace tf
