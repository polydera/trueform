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
#include "../core/views/drop.hpp"
#include "../core/views/take.hpp"
#include "../intersect/types/tagged_intersections.hpp"
#include "../topology/topo_type.hpp"
#include "./loop/cut_faces.hpp"
#include "./loop/tagged_descriptor.hpp"

namespace tf {

/// @ingroup cut_data
/// @brief Low-level two-mesh face cutting for boolean operations.
///
/// Splits faces from two meshes along their intersection curves.
/// Used internally by @ref tf::make_boolean, @ref tf::make_boolean_pair,
/// and @ref tf::make_mesh_arrangements.
///
/// @tparam Index The index type.
template <typename Index>
class tagged_cut_faces
    : public loop::cut_faces<Index, loop::tagged_descriptor<Index>> {
  using base_t = loop::cut_faces<Index, loop::tagged_descriptor<Index>>;

public:
  template <typename Policy0, typename Policy1, typename RealT,
            std::size_t Dims>
  auto
  build(const tf::polygons<Policy0> &_polygons0,
        const tf::polygons<Policy1> &_polygons1,
        const tf::intersect::tagged_intersections<Index, RealT, Dims> &tgs) {
    clear();
    auto make_polygons = [](const auto &polygons) {
      return tf::wrap_map(polygons, [](auto &&x) {
        return tf::core::make_polygons(x.faces(),
                                       x.points().template as<RealT>());
      });
    };
    const auto &polygons0 = make_polygons(_polygons0);
    const auto &polygons1 = make_polygons(_polygons1);
    _own_map.reserve(tgs.intersections().size() * 3);
    _map_offset = tgs.intersection_points().size();
    auto apply_f = [&](auto intersection, const auto &f) {
      if (intersection.tag == 0)
        f(polygons0, polygons1);
      else
        f(polygons1, polygons0);
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
  }

  auto clear() {
    _map_offset = 0;
    _partition_id = 0;
    _own_map.clear();
    base_t::clear();
  }

  auto partition_id() const { return _partition_id; }

  auto connectivity_per_face_edge0() const {
    return tf::take(base_t::connectivity_per_face_edge(), _partition_id);
  }

  auto connectivity_per_face_edge1() const {
    return tf::drop(base_t::connectivity_per_face_edge(), _partition_id);
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

  /// Check if a created edge lies on an original edge of the polygon
  template <typename RealT, std::size_t Dims, typename Policy>
  auto is_created_edge_original(
      loop::vertex<Index> v0,
      loop::vertex<Index> v1,
      const tf::intersect::tagged_intersections<Index, RealT, Dims> &tgs,
      const tf::polygons<Policy> &polygons) const -> bool {

    // Both must be created vertices
    if (v0.source != tf::loop::vertex_source::created ||
        v1.source != tf::loop::vertex_source::created)
      return false;

    const auto &ins0 = tgs.flat_intersections()[v0.intersection_index];
    const auto &ins1 = tgs.flat_intersections()[v1.intersection_index];

    // Must be same tag and same object (same polygon)
    if (ins0.tag != ins1.tag || ins0.object != ins1.object)
      return false;

    // Both must be at vertices
    if (ins0.target.label != tf::topo_type::vertex ||
        ins1.target.label != tf::topo_type::vertex)
      return false;

    // Check if local indices are adjacent in polygon
    Index idx0 = ins0.target.id;
    Index idx1 = ins1.target.id;
    Index polygon_size = polygons.faces()[ins0.object].size();

    Index diff = (idx1 > idx0) ? (idx1 - idx0) : (idx0 - idx1);
    return diff == 1 || diff == polygon_size - 1;
  }

private:
  auto handle_id(loop::tagged_descriptor<Index> d, tf::loop::vertex<Index> v) {
    if (v.source == loop::vertex_source::created)
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

  Index _map_offset;
  Index _partition_id;
  tf::hash_map<std::array<Index, 2>, Index, tf::array_hash<Index, 2>> _own_map;
};
} // namespace tf
