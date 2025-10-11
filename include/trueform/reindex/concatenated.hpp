/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/segments.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/vectors_buffer.hpp"
#include "../core/views/slice.hpp"
#include "../core/views/slide_range.hpp"
#include "../core/views/zip.hpp"
namespace tf {

namespace reindex {
template <typename Index, typename Policy0, typename Policy1,
          typename... Policies, typename RealT, std::size_t Dims,
          std::size_t Ngon>
auto concatenated_impl(const tf::polygons<Policy0> &polygons0,
                       const tf::polygons<Policy1> &polygons1,
                       const tf::polygons<Policies> &...polygons,
                       tf::polygons_buffer<Index, RealT, Dims, Ngon> &out) {
  Index start_p = 0;
  Index start_f = 0;

  auto make_copy = [&](const auto &polygons) {
    const Index end_p = start_p + static_cast<Index>(polygons.points().size());
    const Index end_f = start_f + static_cast<Index>(polygons.faces().size());

    const Index point_offset = start_p;
    tf::parallel_apply(
        tf::zip(polygons.faces(),
                tf::slice(out.faces_buffer(), start_f, end_f)),
        [point_offset](auto pair) {
          auto &&in_face = std::get<0>(pair);
          auto &&out_face = std::get<1>(pair);
          // write each vertex id with the point offset
          for (auto &&zipped : tf::zip(in_face, out_face)) {
            auto &&v_in = std::get<0>(zipped);
            auto &&v_out = std::get<1>(zipped);
            v_out = static_cast<Index>(v_in) + point_offset;
          }
        },
        tf::checked);

    tf::parallel_copy(polygons.points(),
                      tf::slice(out.points_buffer(), start_p, end_p));

    start_p = end_p;
    start_f = end_f;
  };

  std::apply([&](const auto &...polygons) { (make_copy(polygons), ...); },
             std::forward_as_tuple(polygons0, polygons1, polygons...));
  return out;
}

template <typename Index, typename Policy0, typename Policy1,
          typename... Policies>
auto concatenated_same_gons(const tf::polygons<Policy0> &polygons0,
                            const tf::polygons<Policy1> &polygons1,
                            const tf::polygons<Policies> &...polygons) {
  tf::polygons_buffer<Index, tf::coordinate_type<Policy0, Policy1, Policies...>,
                      tf::coordinate_dims_v<Policy0>,
                      tf::static_size_v<decltype(polygons0.faces()[0])>>
      out;
  Index total_face_size = polygons0.faces().size() + polygons1.faces().size() +
                          (0 + ... + polygons.faces().size());
  Index total_point_size = polygons0.points().size() +
                           polygons1.points().size() +
                           (0 + ... + polygons.points().size());
  out.faces_buffer().allocate(total_face_size);
  out.points_buffer().allocate(total_point_size);

  concatenated_impl(polygons0, polygons1, polygons..., out);
  return out;
}

template <typename Index, typename Policy0, typename Policy1,
          typename... Policies>
auto concatenated_diff_gons(const tf::polygons<Policy0> &polygons0,
                            const tf::polygons<Policy1> &polygons1,
                            const tf::polygons<Policies> &...polygons) {
  tf::polygons_buffer<Index, tf::coordinate_type<Policy0, Policy1, Policies...>,
                      tf::coordinate_dims_v<Policy0>, tf::dynamic_size>
      out;

  Index total_faces = polygons0.faces().size() + polygons1.faces().size() +
                      (0 + ... + polygons.faces().size());

  auto &offsets = out.faces_buffer().offsets_buffer();
  offsets.allocate(total_faces + 1);
  offsets[0] = 0;

  Index start_f = 0;
  auto fill_offsets = [&](const auto &polygons) {
    Index end_f = start_f + polygons.faces().size();
    auto r = tf::slice(tf::make_slide_range<2>(offsets), start_f, end_f);
    for (auto &&[ofs, face] : tf::zip(r, polygons.faces()))
      ofs[1] = face.size() + ofs[0];
    start_f = end_f;
  };

  tf::apply([&](const auto &...polygons) { (fill_offsets(polygons), ...); },
            std::forward_as_tuple(polygons0, polygons1, polygons...));

  Index total_point_size = polygons0.points().size() +
                           polygons1.points().size() +
                           (0 + ... + polygons.points().size());
  out.faces_buffer().allocate(offsets.back());
  out.points_buffer().allocate(total_point_size);
  concatenated_impl(polygons0, polygons1, polygons..., out);
  return out;
}
} // namespace reindex

template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::polygons<Policy0> &polygons0,
                  const tf::polygons<Policy1> &polygons1,
                  const tf::polygons<Policies> &...polygons) {
  using index_t = std::common_type_t<decltype(polygons0.faces()[0][0]),
                                     decltype(polygons1.faces()[0][0]),
                                     decltype(polygons.faces()[0][0])...>;
  constexpr bool all_same_gons =
      (tf::static_size_v<decltype(polygons0.faces()[0])> ==
       tf::static_size_v<decltype(polygons1.faces()[0])>) &&
      (true && ... &&
       (tf::static_size_v<decltype(polygons0.faces()[0])> ==
        tf::static_size_v<decltype(polygons.faces()[0])>));
  if constexpr (all_same_gons)
    return tf::reindex::concatenated_same_gons<index_t>(polygons0, polygons1,
                                                        polygons...);
  else
    return tf::reindex::concatenated_diff_gons<index_t>(polygons0, polygons1,
                                                        polygons...);
}

template <typename Index, typename Policy0, typename Policy1,
          typename... Policies>
auto concatenated(const tf::segments<Policy0> &segments0,
                  const tf::segments<Policy1> &segments1,
                  const tf::segments<Policies> &...segments) {
  tf::segments_buffer<Index, tf::coordinate_type<Policy0, Policy1, Policies...>,
                      tf::coordinate_dims_v<Policy0>>
      out;
  Index total_edge_size = segments0.edges().size() + segments1.edges().size() +
                          (0 + ... + segments.edges().size());
  Index total_point_size = segments0.points().size() +
                           segments1.points().size() +
                           (0 + ... + segments.points().size());
  out.edges_buffer().allocate(total_edge_size);
  out.points_buffer().allocate(total_point_size);

  Index start_p = 0;
  Index start_f = 0;

  auto make_copy = [&](const auto &segments) {
    const Index end_p = start_p + static_cast<Index>(segments.points().size());
    const Index end_f = start_f + static_cast<Index>(segments.edges().size());

    const Index point_offset = start_p;
    tf::parallel_apply(
        tf::zip(segments.edges(),
                tf::slice(out.edges_buffer(), start_f, end_f)),
        [point_offset](auto pair) {
          auto &&in_edge = std::get<0>(pair);
          auto &&out_edge = std::get<1>(pair);
          out_edge[0] = in_edge[0] + point_offset;
          out_edge[1] = in_edge[1] + point_offset;
        },
        tf::checked);

    tf::parallel_copy(segments.points(),
                      tf::slice(out.points_buffer(), start_p, end_p));

    start_p = end_p;
    start_f = end_f;
  };

  std::apply([&](const auto &...segments) { (make_copy(segments), ...); },
             std::forward_as_tuple(segments0, segments1, segments...));

  return out;
}

template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::points<Policy0> &points0,
                  const tf::points<Policy1> &points1,
                  const tf::points<Policies> &...points) {
  auto total_point_size = points0.points().size() + points1.points().size() +
                          (0 + ... + points.points().size());
  tf::points_buffer<tf::coordinate_type<Policy0, Policy1, Policies...>,
                    tf::coordinate_dims_v<Policy0>>
      out;
  out.allocate(total_point_size);

  std::size_t start_p = 0;
  std::size_t start_f = 0;

  auto make_copy = [&](const auto &points) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(points.points().size());
    const std::size_t end_f =
        start_f + static_cast<std::size_t>(points.edges().size());

    tf::parallel_copy(points, tf::slice(out, start_p, end_p));

    start_p = end_p;
    start_f = end_f;
  };

  std::apply([&](const auto &...points) { (make_copy(points), ...); },
             std::forward_as_tuple(points0, points1, points...));
  return out;
}

template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::vectors<Policy0> &vectors0,
                  const tf::vectors<Policy1> &vectors1,
                  const tf::vectors<Policies> &...vectors) {
  auto total_vector_size = vectors0.vectors().size() +
                           vectors1.vectors().size() +
                           (0 + ... + vectors.vectors().size());
  tf::vectors_buffer<tf::coordinate_type<Policy0, Policy1, Policies...>,
                     tf::coordinate_dims_v<Policy0>>
      out;
  out.allocate(total_vector_size);

  std::size_t start_p = 0;
  std::size_t start_f = 0;

  auto make_copy = [&](const auto &vectors) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(vectors.vectors().size());
    const std::size_t end_f =
        start_f + static_cast<std::size_t>(vectors.edges().size());

    tf::parallel_copy(vectors, tf::slice(out, start_p, end_p));

    start_p = end_p;
    start_f = end_f;
  };

  std::apply([&](const auto &...vectors) { (make_copy(vectors), ...); },
             std::forward_as_tuple(vectors0, vectors1, vectors...));
  return out;
}

template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::unit_vectors<Policy0> &unit_vectors0,
                  const tf::unit_vectors<Policy1> &unit_vectors1,
                  const tf::unit_vectors<Policies> &...unit_vectors) {
  auto total_unit_vector_size = unit_vectors0.unit_vectors().size() +
                                unit_vectors1.unit_vectors().size() +
                                (0 + ... + unit_vectors.unit_vectors().size());
  tf::unit_vectors_buffer<tf::coordinate_type<Policy0, Policy1, Policies...>,
                          tf::coordinate_dims_v<Policy0>>
      out;
  out.allocate(total_unit_vector_size);

  std::size_t start_p = 0;
  std::size_t start_f = 0;

  auto make_copy = [&](const auto &unit_vectors) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(unit_vectors.unit_vectors().size());
    const std::size_t end_f =
        start_f + static_cast<std::size_t>(unit_vectors.edges().size());

    tf::parallel_copy(unit_vectors, tf::slice(out, start_p, end_p));

    start_p = end_p;
    start_f = end_f;
  };

  std::apply(
      [&](const auto &...unit_vectors) { (make_copy(unit_vectors), ...); },
      std::forward_as_tuple(unit_vectors0, unit_vectors1, unit_vectors...));
  return out;
}

} // namespace tf
