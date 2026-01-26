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
#include "../core/algorithm/block_reduce.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../intersect/types/simple_intersections.hpp"
#include "./loop/descriptor.hpp"
#include "./loop/loop_extractor.hpp"
#include "./loop/vertex.hpp"

namespace tf {

/// @ingroup cut_data
/// @brief Low-level scalar field face cutting.
///
/// Splits faces along scalar field isocurves.
/// Used internally by @ref tf::embedded_isocurves and @ref tf::make_isobands.
///
/// @tparam Index The index type.
template <typename Index> class scalar_cut_faces {
public:
  auto mapped_loops() const {
    return tf::make_offset_block_range(_offsets, _vertices);
  }

  auto descriptors() const { return tf::make_range(_descriptors); }

  auto intersection_edges() const {
    return tf::make_range(_intersection_edges);
  }

  template <typename Policy, typename RealT, std::size_t Dims>
  auto
  build(const tf::polygons<Policy> &_polygons,
        const tf::intersect::simple_intersections<Index, RealT, Dims> &sfi) {
    clear();
    auto polygons = tf::wrap_map(_polygons, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });

    tf::blocked_reduce(
        sfi.intersections(),
        std::tie(_vertices, _offsets, _intersection_edges, _descriptors),
        std::make_tuple(
            tf::buffer<tf::loop::vertex<Index>>{}, tf::buffer<Index>{},
            tf::blocked_buffer<tf::loop::vertex<Index>, 2>{},
            tf::buffer<tf::loop::descriptor<Index>>{},
            tf::loop::loop_extractor<Index, tf::coordinate_type<Policy>>{}),
        [&](const auto &inters, auto &tup) {
          for (const auto &ins : inters) {
            auto &[vertices, offsets, edges, descriptors, extractor] = tup;
            Index n_loops = extractor.build(
                polygons.faces()[ins.front().object],
                tf::make_points(sfi.intersection_points()), polygons.points(),
                ins, [&](const auto &x) { return sfi.get_flat_index(x); },
                offsets, vertices);
            std::copy(extractor.intersection_edges().begin(),
                      extractor.intersection_edges().end(),
                      std::back_inserter(edges));
            for (Index i = 0; i < n_loops; ++i)
              descriptors.push_back({ins.front().object});
          }
        },
        [](const auto &l_result, auto &result) {
          const auto &[l_vs, l_offs, l_edges, l_ds, _] = l_result;
          auto &[vs, offs, edges, ds] = result;
          auto old_vs_size = vs.size();
          vs.reallocate(old_vs_size + l_vs.size());
          std::copy(l_vs.begin(), l_vs.end(), vs.begin() + old_vs_size);
          auto old_ds_size = ds.size();
          ds.reallocate(old_ds_size + l_ds.size());
          std::copy(l_ds.begin(), l_ds.end(), ds.begin() + old_ds_size);
          auto old_edges_size = edges.size();
          edges.reallocate(old_edges_size + l_edges.size());
          std::copy(l_edges.begin(), l_edges.end(),
                    edges.begin() + old_edges_size);
          auto old_offs_size = offs.size();
          offs.reallocate(old_offs_size + l_offs.size());
          auto it = offs.begin() + old_offs_size;
          for (auto o : l_offs)
            *it++ = o + old_vs_size;
        });
    if (_offsets.size())
      _offsets.push_back(_vertices.size());
  }

  auto clear() {
    _vertices.clear();
    _offsets.clear();
    _descriptors.clear();
    _intersection_edges.clear();
  }

private:
  tf::buffer<tf::loop::vertex<Index>> _vertices;
  tf::buffer<Index> _offsets;
  tf::buffer<tf::loop::descriptor<Index>> _descriptors;
  tf::blocked_buffer<tf::loop::vertex<Index>, 2> _intersection_edges;
};
} // namespace tf
