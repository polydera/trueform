/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./edge_id_in_face.hpp"
#include "./face_membership.hpp"
namespace tf {
namespace implementation {
template <typename Index, typename Range, typename F>
auto face_edge_neighbors(std::integral_constant<std::size_t, 3>,
                         const tf::face_membership<Index> &blink, const Range &,
                         Index face_id, const Index &v0, const Index &v1,
                         const F &apply) {

  const auto &range0 = blink[v0];
  auto it0 = range0.begin();
  auto end0 = range0.end();
  const auto &range1 = blink[v1];
  auto it1 = range1.begin();
  auto end1 = range1.end();
  // intersection on sorted ranges
  while ((it0 != end0) & (it1 != end1)) {
    if (*it0 > *it1)
      ++it0;
    else {
      if (char(Index(*it0) != face_id) & char(!(*it1 > *it0))) {
        if (apply(*it0++))
          return;
      }
      ++it1;
    }
  }
}
template <std::size_t N, typename Index, typename Range, typename F>
auto face_edge_neighbors(std::integral_constant<std::size_t, N>,
                         const tf::face_membership<Index> &blink,
                         const Range &faces, Index face_id, const Index &v0,
                         const Index &v1, const F &apply) {

  const auto &range0 = blink[v0];
  auto it0 = range0.begin();
  auto end0 = range0.end();
  const auto &range1 = blink[v1];
  auto it1 = range1.begin();
  auto end1 = range1.end();
  // intersection on sorted ranges
  while ((it0 != end0) & (it1 != end1)) {
    if (*it0 > *it1)
      ++it0;
    else {
      if (char(Index(*it0) != face_id) & char(!(*it1 > *it0))) {
        const auto &face1 = faces[*it1];
        Index size = face1.size();
        Index edge_id = tf::edge_id_in_face(face1, v0, v1);
        if (edge_id != size && apply(*it0++))
          return;
      }
      ++it1;
    }
  }
}
} // namespace implementation

template <typename Index, typename Range, typename F>
auto face_edge_neighbors_apply(const tf::face_membership<Index> &blink,
                               const Range &faces, Index face_id,
                               const Index &v0, const Index &v1,
                               const F &apply) {
  implementation::face_edge_neighbors(
      std::integral_constant<std::size_t,
                             tf::static_size_v<decltype(faces[face_id])>>{},
      blink, faces, face_id, v0, v1, apply);
}

template <typename Index, typename Range, typename Iterator>
auto face_edge_neighbors(const tf::face_membership<Index> &blink,
                         const Range &faces, Index face_id, const Index &v0,
                         const Index &v1, Iterator out) {
  implementation::face_edge_neighbors(
      std::integral_constant<std::size_t,
                             tf::static_size_v<decltype(faces[face_id])>>{},
      blink, faces, face_id, v0, v1, [&](const auto &val) {
        *out++ = val;
        return false;
      });
  return out;
}

template <typename Index, typename Range, typename Iterator>
auto face_edge_neighbors(const tf::face_membership<Index> &blink,
                         const Range &faces, Index face_id, const Index &v0,
                         const Index &v1, Iterator begin, Iterator end) {
  implementation::face_edge_neighbors(
      std::integral_constant<std::size_t,
                             tf::static_size_v<decltype(faces[face_id])>>{},
      blink, faces, face_id, v0, v1, [&](const auto &val) {
        *begin++ = val;
        return begin == end;
      });
  return begin;
}
} // namespace tf
