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
#include "../../topology/face_membership.hpp"
#include "../../topology/structures/compute_face_link_per_edge.hpp"
#include "./cut_faces_base.hpp"

namespace tf::loop {
template <typename Index, typename ObjectKey>
class cut_faces : public cut_faces_base<Index, ObjectKey> {
  using base_t = cut_faces_base<Index, ObjectKey>;

public:
  auto connectivity_per_face_edge() const {
    return tf::make_offset_block_range(base_t::_loop_offsets, _ob);
  }

  auto clear() {
    base_t::clear();
    _ob.clear();
    _fm.clear();
  }

protected:
  template <typename Range, typename Policy1, typename F0, typename F1,
            typename F2>
  auto build(const Range &intersections,
             const tf::points<Policy1> &intersection_points,
             const F0 &apply_to_polygons, const F1 &handle_id,
             const F2 &get_flat_id) {
    clear();
    Index total_vertices =
        base_t::build(intersections, intersection_points, apply_to_polygons,
                      handle_id, get_flat_id);

    _fm.build(tf::make_faces(base_t::loops()), total_vertices,
              base_t::_loop_vertices.size());
    tf::topology::compute_face_link_per_edge(base_t::loops(), _fm, _ob);
  }

  tf::offset_block_buffer<Index, Index> _ob;
  tf::face_membership<Index> _fm;
};
} // namespace tf::loop
