/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../intersect/intersections_within_polygons.hpp"
#include "./loop/cut_faces.hpp"
#include "./loop/descriptor.hpp"

namespace tf {
template <typename Index>
class cut_faces : public loop::cut_faces<Index, loop::descriptor<Index>> {
  using base_t = loop::cut_faces<Index, loop::descriptor<Index>>;

public:
  template <typename Policy, typename RealT, std::size_t Dims>
  auto
  build(const tf::polygons<Policy> &_polygons,
        const tf::intersections_within_polygons<Index, RealT, Dims> &tgs) {
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
        [&](const auto &, auto &&f) { f(polygons, polygons); }, handle_id_f,
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
