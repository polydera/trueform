/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_copy_blocked.hpp"
#include "../core/index_map.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/views/block_indirect_range.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {
template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3, typename Policy1>
auto reindexed(const tf::polygons<Policy> &polygons,
               const tf::index_map<Range0, Range1> &face_im,
               const tf::index_map<Range2, Range3> &point_im,
               tf::polygons<Policy1> &out) {
  tf::parallel_copy_blocked(
      tf::make_indirect_range(
          face_im.kept_ids(),
          tf::make_block_indirect_range(polygons.faces(), point_im.f())),
      out.faces());
  tf::parallel_copy(
      tf::make_indirect_range(point_im.kept_ids(), polygons.points()),
      out.points());
}

template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3, typename Index, typename RealT, std::size_t Dims,
          std::size_t Ngons>
auto reindexed(const tf::polygons<Policy> &polygons,
               const tf::index_map<Range0, Range1> &face_im,
               const tf::index_map<Range2, Range3> &point_im,
               tf::polygons_buffer<Index, RealT, Dims, Ngons> &out) {
  out.faces_buffer().allocate(face_im.kept_ids().size());
  out.points_buffer().allocate(point_im.kept_ids().size());
  auto out_s = out.polygons();
  reindexed(polygons, face_im, point_im, out_s);
}

template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3, typename Index, typename RealT, std::size_t Dims>
auto reindexed(const tf::polygons<Policy> &polygons,
               const tf::index_map<Range0, Range1> &face_im,
               const tf::index_map<Range2, Range3> &point_im,
               tf::polygons_buffer<Index, RealT, Dims, tf::dynamic_size> &out) {
  out.faces_buffer().offsets_buffer().allocate(face_im.kept_ids().size() + 1);
  auto it = out.faces_buffer().offsets_buffer().begin();
  *it = 0;
  for (const auto &face :
       tf::make_indirect_range(face_im.kept_ids(), polygons.faces())) {
    it[1] = face.size() + it[0];
    ++it;
  }
  out.faces_buffer().data_buffer().allocate(*it);
  out.points_buffer().allocate(point_im.kept_ids().size());
  auto out_s = out.polygons();
  reindexed(polygons, face_im, point_im, out_s);
}

template <typename Policy, typename Range0, typename Range1, typename Range2,
          typename Range3>
auto reindexed(const tf::polygons<Policy> &polygons,
               const tf::index_map<Range0, Range1> &face_im,
               const tf::index_map<Range2, Range3> &point_im) {
  tf::polygons_buffer<std::decay_t<decltype(face_im.kept_ids()[0])>,
                      tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>,
                      tf::static_size_v<decltype(polygons[0])>>
      out;
  reindexed(polygons, face_im, point_im, out);
  return out;
}
} // namespace tf
