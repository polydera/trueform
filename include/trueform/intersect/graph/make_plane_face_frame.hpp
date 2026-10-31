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

#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/plane_frame.hpp"
#include "./plane_face_support.hpp"

namespace tf::intersect::graph {

/// THE LINE LAW: a face's carrier is the support of its CORNERS
/// (@ref plane_face_support) and nothing else, for every tier that founds a
/// plane on a face. Its name (@ref name_plane_face) and its frame are two
/// readings of that one support, so a void name and a line frame are one
/// verdict. A face whose corners are collinear is a LINE CARRIER whatever its
/// definitions hold: a definition endpoint is a created point rounded onto
/// the lattice, and a plane such a point spans is not one the face stands on.
template <typename Int, typename Face, typename GetVertex>
auto make_plane_face_frame(const Face &face, const GetVertex &get_vertex)
    -> tf::exact::plane_frame<Int> {
  return tf::exact::make_supported_plane_frame<Int>(
      plane_face_support<Int>(face, get_vertex));
}

} // namespace tf::intersect::graph
