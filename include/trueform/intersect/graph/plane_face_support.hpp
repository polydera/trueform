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

#include "../../exact/plane_support.hpp"
#include "../../exact/vertex.hpp"

#include <cstddef>

namespace tf::intersect::graph {

/// The support a face's carrier stands on, in FACE terms: its corners are
/// offered to @ref tf::exact::plane_support in the ORIGINAL winding, so the
/// plane they state is the face's own. The scan's own cross comes back with
/// it, so a reader that needs the normal takes the one already computed.
template <typename Int, typename Face, typename GetVertex>
auto plane_face_support(const Face &face, const GetVertex &get_vertex)
    -> tf::exact::plane_support<Int> {
  tf::exact::plane_support<Int> support;
  const auto n = face.size();
  for (std::size_t corner = 0; corner < n && support.size < 3; ++corner)
    support.offer(get_vertex(corner));
  return support;
}

/// That support as the three points it kept.
///
/// The answer is collinear exactly when the face HAS no plane, and then it
/// states the line the face collapsed to: `b` is the first corner off `a`
/// and `c` repeats `b`. A reader may therefore take a degenerate answer as
/// the verdict itself.
template <typename Int, typename Face, typename GetVertex>
auto plane_face_support(const Face &face, const GetVertex &get_vertex,
                        tf::exact::pt3<Int> &a, tf::exact::pt3<Int> &b,
                        tf::exact::pt3<Int> &c) -> bool {
  auto support = plane_face_support<Int>(face, get_vertex);
  if (support.size == 0)
    return false;
  a = support.point[0];
  b = support.size > 1 ? support.point[1] : a;
  c = support.size > 2 ? support.point[2] : b;
  return support.size == 3;
}

} // namespace tf::intersect::graph
