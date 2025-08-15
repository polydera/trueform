/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/views/drop.hpp"
#include "../../core/views/take.hpp"
#include "../../intersect/tagged_intersections.hpp"
#include "./descriptor.hpp"
#include "./intersection_loops.hpp"

#include "../../topology/face_membership.hpp"
#include "../../topology/structures/compute_face_link_per_edge.hpp"
namespace tf::loop {
template <typename Index, typename RealT>
class tagged_intersection_loops
    : public intersection_loops<Index, descriptor<Index>> {
  using base_t = intersection_loops<Index, descriptor<Index>>;

public:
  template <typename Policy0, typename Policy1, std::size_t Dims>
  auto build(const tf::polygons<Policy0> &polygons0,
             const tf::polygons<Policy1> &polygons1,
             const tf::tagged_intersections<Index, RealT, Dims> &tgs) {
    clear();
    _own_map.reserve(tgs.intersections().size() * 3);
    _map_offset = tgs.intersection_points().size();
    auto apply_f = [&](auto intersection, const auto &f) {
      if (intersection.tag == 0)
        f(polygons0);
      else
        f(polygons1);
    };
    auto handle_id_f = [this](auto d, auto v) { return this->handle_id(d, v); };

    base_t::build(tgs.intersections(),
                  tf::make_points(tgs.intersection_points()), apply_f,
                  handle_id_f,
                  [&tgs](const auto &x) { return tgs.get_flat_index(x); });

    _partition_id =
        std::upper_bound(
            base_t::descriptors().begin(), base_t::descriptors().end(), 0,
            [](const auto &value, const auto &r1) { return value < r1.tag; }) -
        base_t::descriptors().begin();

    _fm.build(tf::make_faces(base_t::loops()), base_t::_vertices.size(),
              base_t::_loop_vertices.size());
    tf::topology::compute_face_link_per_edge(base_t::loops(), _fm, _ob);
  }

  auto clear() {
    _map_offset = 0;
    _partition_id = 0;
    _own_map.clear();
    base_t::clear();
  }

  auto descriptors0() const {
    return tf::take(base_t::descriptors(), _partition_id);
  }

  auto descriptors1() const {
    return tf::drop(base_t::descriptors(), _partition_id);
  }

  auto loops0() const { return tf::take(base_t::loops(), _partition_id); }

  auto loops1() const { return tf::drop(base_t::loops(), _partition_id); }

  auto mapped_loops0() const {
    return tf::take(base_t::mapped_loops(), _partition_id);
  }

  auto mapped_loops1() const {
    return tf::drop(base_t::mapped_loops(), _partition_id);
  }

private:
  auto handle_id(descriptor<Index> d, tf::loop::vertex<Index> v) {
    if (v.source() == vertex_source::created)
      return std::make_pair(false, v.id);
    else {
      auto key = std::array<Index, 2>{d.tag, v.id};
      auto it = _own_map.find(key);
      if (it == _own_map.end()) {
        _own_map[key] = _map_offset;
        return std::make_pair(true, _map_offset++);
      } else
        return std::make_pair(false, it->second);
    }
  }

  tf::offset_block_buffer<Index, Index> _ob;
  tf::face_membership<Index> _fm;
  Index _map_offset;
  Index _partition_id;
  tf::hash_map<std::array<Index, 2>, Index, tf::array_hash<Index, 2>> _own_map;
};
} // namespace tf::loop
