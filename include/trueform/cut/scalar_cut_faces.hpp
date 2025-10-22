/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../intersect/types/simple_intersections.hpp"
#include "./loop/cut_faces_base.hpp"
#include "./loop/descriptor.hpp"

namespace tf {
template <typename Index>
class scalar_cut_faces
    : public loop::cut_faces_base<Index, loop::descriptor<Index>> {
  using base_t = loop::cut_faces_base<Index, loop::descriptor<Index>>;

public:
  template <typename Policy, typename RealT, std::size_t Dims>
  auto
  build(const tf::polygons<Policy> &_polygons,
        const tf::intersect::simple_intersections<Index, RealT, Dims> &tgs) {
    clear();
    auto polygons = tf::wrap_map(_polygons, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });

    _own_map.reserve(tgs.intersections().size() * 3);
    _map_offset = tgs.intersection_points().size();
    auto handle_id_f = [this](auto, auto v) { return this->handle_id(v); };

    base_t::build(
        tgs.intersections(), tf::make_points(tgs.intersection_points()),
        [&](const auto &, auto &&f) { f(polygons); }, handle_id_f,
        [&tgs](const auto &x) { return tgs.get_flat_index(x); });
  }

  auto clear() {
    _map_offset = 0;
    _own_map.clear();
    base_t::clear();
  }

private:
  auto handle_id(tf::loop::vertex<Index> v) {
    if (v.source == loop::vertex_source::created)
      return std::make_pair(false, v.id);
    else {
      auto key = v.id;
      auto it = _own_map.find(key);
      if (it == _own_map.end()) {
        _own_map[key] = _map_offset;
        return std::make_pair(true, _map_offset++);
      } else
        return std::make_pair(false, it->second);
    }
  }

  Index _map_offset;
  tf::hash_map<Index, Index> _own_map;
};
} // namespace tf
