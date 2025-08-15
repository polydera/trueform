/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../intersect/base/simple_intersections.hpp"
#include "./intersection_loops.hpp"

#include "../../topology/face_membership.hpp"
#include "../../topology/structures/compute_face_link_per_edge.hpp"
namespace tf::loop {
template <typename Index, typename RealT>
class simple_intersection_loops
    : public intersection_loops<Index, Index> {
  using base_t = intersection_loops<Index, Index>;

public:
  template <typename Policy, std::size_t Dims>
  auto build(const tf::polygons<Policy> &polygons,
             const tf::intersect::simple_intersections<Index, RealT, Dims> &tgs) {
    clear();
    _own_map.reserve(tgs.intersections().size() * 3);
    _map_offset = tgs.intersection_points().size();
    auto apply_f = [&](auto, const auto &f) {
        f(polygons);
    };
    auto handle_id_f = [this](auto , auto v) { return this->handle_id(v); };

    base_t::build(tgs.intersections(),
                  tf::make_points(tgs.intersection_points()), apply_f,
                  handle_id_f,
                  [&tgs](const auto &x) { return tgs.get_flat_index(x); });

    _fm.build(tf::make_faces(base_t::loops()), base_t::_vertices.size(),
              base_t::_loop_vertices.size());
    tf::topology::compute_face_link_per_edge(base_t::loops(), _fm, _ob);
  }

  auto clear() {
    _map_offset = 0;
    _own_map.clear();
    base_t::clear();
  }

private:
  auto handle_id(tf::loop::vertex<Index> v) {
    if (v.source() == vertex_source::created)
      return std::make_pair(false, v.id);
    else {
      auto it = _own_map.find(v.id);
      if (it == _own_map.end()) {
        _own_map[v.id] = _map_offset;
        return std::make_pair(true, _map_offset++);
      } else
        return std::make_pair(false, it->second);
    }
  }

  tf::offset_block_buffer<Index, Index> _ob;
  tf::face_membership<Index> _fm;
  Index _map_offset;
  tf::hash_map<Index, Index, Index> _own_map;
};
} // namespace tf::loop

