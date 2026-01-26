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
#include "../clean/soup/segments.hpp"
#include "../core/is_soup.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../topology/planar_embedding.hpp"
#include "./planar_overlay.hpp"

namespace tf {

/// @ingroup cut_planar
/// @brief Compute 2D planar arrangements from line segments.
///
/// Subdivides the plane according to a set of line segments,
/// creating faces with optional holes. Access results via inherited
/// methods from @ref tf::planar_embedding (faces, holes) and
/// @ref tf::planar_overlay (edges, points).
///
/// @tparam Index The index type.
/// @tparam RealType The coordinate type.
template <typename Index, typename RealType>
class planar_arrangements : public tf::planar_embedding<Index, RealType>,
                            public tf::planar_overlay<Index, RealType, 2> {
  using pe_base_t = tf::planar_embedding<Index, RealType>;
  using is_base_t = tf::planar_overlay<Index, RealType, 2>;

public:
  template <typename Policy> auto build(const tf::segments<Policy> &segments) {
    clear();

    auto build_intersections = [&](auto &&segments) {
      _em.build(segments);
      _tree.build(segments, tf::config_tree(4, 4));
      _si.build(segments | tf::tag(_em), _tree);
      is_base_t::build(segments, _si);
    };
    if constexpr (tf::is_soup<Policy>) {
      tf::clean::segment_soup<Index, RealType, 2> _cs;
      _cs.build(segments);
      build_intersections(_cs.segments());
    } else
      build_intersections(segments);
    _work_buffer.allocate(is_base_t::edges().size() * 4);
    tf::parallel_for_each(
        tf::zip(is_base_t::edges(), tf::make_blocked_range<4>(_work_buffer)),
        [](auto pair) {
          auto &&[_in, _out] = pair;
          _out[0] = _in[0];
          _out[1] = _in[1];
          _out[2] = _in[1];
          _out[3] = _in[0];
        });
    pe_base_t::build(tf::make_edges(tf::make_blocked_range<2>(_work_buffer)),
                     is_base_t::points());
  }

  auto clear() {
    _em.clear();
    _tree.clear();
    _si.clear();
    is_base_t::clear();
    pe_base_t::clear();
  }

private:
  tf::edge_membership<Index> _em;
  tf::aabb_tree<Index, RealType, 2> _tree;
  tf::intersections_within_segments<Index, RealType, 2> _si;
  tf::buffer<Index> _work_buffer;
};
} // namespace tf
