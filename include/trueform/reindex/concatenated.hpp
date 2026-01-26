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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/frame_of.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/segments.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/transformed.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/vectors_buffer.hpp"
#include "../core/views/slice.hpp"
#include "../core/views/slide_range.hpp"
#include "../core/views/zip.hpp"
namespace tf {

/// @cond INTERNAL
namespace reindex {

namespace impl {
template <typename R0, typename R1, typename T>
auto copy_with_transformation(const R0 &source, R1 &&destincation,
                              const T &transformation) {
  if constexpr (tf::linalg::is_identity<T>)
    tf::parallel_copy(source, destincation);
  else {
    tf::parallel_for_each(tf::zip(source, destincation),
                       [&transformation](auto &&pair) {
                         auto &&[src, dst] = pair;
                         dst = tf::transformed(src, transformation);
                       });
  }
}
} // namespace impl

template <typename Index, typename Policy0, typename Policy1,
          typename... Policies, typename RealT, std::size_t Dims,
          std::size_t Ngon>
auto concatenated_impl(tf::polygons_buffer<Index, RealT, Dims, Ngon> &out,
                       const tf::polygons<Policy0> &polygons0,
                       const tf::polygons<Policy1> &polygons1,
                       const tf::polygons<Policies> &...polygons) {
  Index start_p = 0;
  Index start_f = 0;

  auto make_copy = [&](const auto &polygons) {
    const Index end_p = start_p + static_cast<Index>(polygons.points().size());
    const Index end_f = start_f + static_cast<Index>(polygons.faces().size());

    const Index point_offset = start_p;
    tf::parallel_for_each(
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

    impl::copy_with_transformation(polygons.points(),
                                    tf::slice(out.points_buffer(), start_p, end_p),
                                    tf::frame_of(polygons));

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

  concatenated_impl(out, polygons0, polygons1, polygons...);
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
  out.faces_buffer().data_buffer().allocate(offsets.back());
  out.points_buffer().allocate(total_point_size);
  concatenated_impl(out, polygons0, polygons1, polygons...);
  return out;
}
} // namespace reindex
/// @endcond

/// @ingroup reindex
/// @brief Concatenate multiple polygon collections.
///
/// Merges polygons into a single @ref tf::polygons_buffer with
/// adjusted face indices. Applies frame transformations if present.
/// Preserves static face size if all inputs match, otherwise uses dynamic.
///
/// @tparam Policy0 Policy of first polygons.
/// @tparam Policy1 Policy of second polygons.
/// @tparam Policies Policies of additional polygons.
/// @param polygons0 First @ref tf::polygons.
/// @param polygons1 Second @ref tf::polygons.
/// @param polygons Additional @ref tf::polygons (variadic).
/// @return A @ref tf::polygons_buffer containing all geometry.
template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::polygons<Policy0> &polygons0,
                  const tf::polygons<Policy1> &polygons1,
                  const tf::polygons<Policies> &...polygons) {
  using index_t =
      std::common_type_t<std::decay_t<decltype(polygons0.faces()[0][0])>,
                         std::decay_t<decltype(polygons1.faces()[0][0])>,
                         std::decay_t<decltype(polygons.faces()[0][0])>...>;
  constexpr bool all_same_gons =
      (tf::static_size_v<decltype(polygons0.faces()[0])> != tf::dynamic_size) &&
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

/// @ingroup reindex
/// @brief Concatenate multiple segment collections.
///
/// Merges segments into a single @ref tf::segments_buffer with
/// adjusted edge indices. Applies frame transformations if present.
///
/// @tparam Policy0 Policy of first segments.
/// @tparam Policy1 Policy of second segments.
/// @tparam Policies Policies of additional segments.
/// @param segments0 First @ref tf::segments.
/// @param segments1 Second @ref tf::segments.
/// @param segments Additional @ref tf::segments (variadic).
/// @return A @ref tf::segments_buffer containing all geometry.
template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::segments<Policy0> &segments0,
                  const tf::segments<Policy1> &segments1,
                  const tf::segments<Policies> &...segments) {
  using Index =
      std::common_type_t<std::decay_t<decltype(segments0.edges()[0][0])>,
                         std::decay_t<decltype(segments1.edges()[0][0])>,
                         std::decay_t<decltype(segments.edges()[0][0])>...>;
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
    tf::parallel_for_each(
        tf::zip(segments.edges(),
                tf::slice(out.edges_buffer(), start_f, end_f)),
        [point_offset](auto pair) {
          auto &&in_edge = std::get<0>(pair);
          auto &&out_edge = std::get<1>(pair);
          out_edge[0] = in_edge[0] + point_offset;
          out_edge[1] = in_edge[1] + point_offset;
        },
        tf::checked);

    reindex::impl::copy_with_transformation(segments.points(),
                                    tf::slice(out.points_buffer(), start_p, end_p),
                                    tf::frame_of(segments));

    start_p = end_p;
    start_f = end_f;
  };

  std::apply([&](const auto &...segments) { (make_copy(segments), ...); },
             std::forward_as_tuple(segments0, segments1, segments...));

  return out;
}

/// @ingroup reindex
/// @brief Concatenate multiple point collections.
///
/// Merges points into a single @ref tf::points_buffer.
/// Applies frame transformations if present.
///
/// @tparam Policy0 Policy of first points.
/// @tparam Policy1 Policy of second points.
/// @tparam Policies Policies of additional points.
/// @param points0 First @ref tf::points.
/// @param points1 Second @ref tf::points.
/// @param points Additional @ref tf::points (variadic).
/// @return A @ref tf::points_buffer containing all points.
template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::points<Policy0> &points0,
                  const tf::points<Policy1> &points1,
                  const tf::points<Policies> &...points) {
  auto total_point_size =
      points0.size() + points1.size() + (0 + ... + points.size());
  tf::points_buffer<tf::coordinate_type<Policy0, Policy1, Policies...>,
                    tf::coordinate_dims_v<Policy0>>
      out;
  out.allocate(total_point_size);

  std::size_t start_p = 0;

  auto make_copy = [&](const auto &points) {
    const std::size_t end_p = start_p + static_cast<std::size_t>(points.size());

    reindex::impl::copy_with_transformation(points, tf::slice(out, start_p, end_p),
                                            tf::frame_of(points));

    start_p = end_p;
  };

  std::apply([&](const auto &...points) { (make_copy(points), ...); },
             std::forward_as_tuple(points0, points1, points...));
  return out;
}

/// @ingroup reindex
/// @brief Concatenate multiple vector collections.
///
/// Merges vectors into a single @ref tf::vectors_buffer.
/// Applies frame transformations if present.
///
/// @tparam Policy0 Policy of first vectors.
/// @tparam Policy1 Policy of second vectors.
/// @tparam Policies Policies of additional vectors.
/// @param vectors0 First @ref tf::vectors.
/// @param vectors1 Second @ref tf::vectors.
/// @param vectors Additional @ref tf::vectors (variadic).
/// @return A @ref tf::vectors_buffer containing all vectors.
template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::vectors<Policy0> &vectors0,
                  const tf::vectors<Policy1> &vectors1,
                  const tf::vectors<Policies> &...vectors) {
  auto total_vector_size =
      vectors0.size() + vectors1.size() + (0 + ... + vectors.size());
  tf::vectors_buffer<tf::coordinate_type<Policy0, Policy1, Policies...>,
                     tf::coordinate_dims_v<Policy0>>
      out;
  out.allocate(total_vector_size);

  std::size_t start_p = 0;

  auto make_copy = [&](const auto &vectors) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(vectors.size());

    reindex::impl::copy_with_transformation(vectors, tf::slice(out, start_p, end_p),
                                            tf::frame_of(vectors));

    start_p = end_p;
  };

  std::apply([&](const auto &...vectors) { (make_copy(vectors), ...); },
             std::forward_as_tuple(vectors0, vectors1, vectors...));
  return out;
}

/// @ingroup reindex
/// @brief Concatenate multiple unit vector collections.
///
/// Merges unit vectors into a single @ref tf::unit_vectors_buffer.
/// Applies frame transformations if present.
///
/// @tparam Policy0 Policy of first unit vectors.
/// @tparam Policy1 Policy of second unit vectors.
/// @tparam Policies Policies of additional unit vectors.
/// @param unit_vectors0 First @ref tf::unit_vectors.
/// @param unit_vectors1 Second @ref tf::unit_vectors.
/// @param unit_vectors Additional @ref tf::unit_vectors (variadic).
/// @return A @ref tf::unit_vectors_buffer containing all unit vectors.
template <typename Policy0, typename Policy1, typename... Policies>
auto concatenated(const tf::unit_vectors<Policy0> &unit_vectors0,
                  const tf::unit_vectors<Policy1> &unit_vectors1,
                  const tf::unit_vectors<Policies> &...unit_vectors) {
  auto total_unit_vector_size = unit_vectors0.size() + unit_vectors1.size() +
                                (0 + ... + unit_vectors.size());
  tf::unit_vectors_buffer<tf::coordinate_type<Policy0, Policy1, Policies...>,
                          tf::coordinate_dims_v<Policy0>>
      out;
  out.allocate(total_unit_vector_size);

  std::size_t start_p = 0;

  auto make_copy = [&](const auto &unit_vectors) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(unit_vectors.size());

    reindex::impl::copy_with_transformation(unit_vectors, tf::slice(out, start_p, end_p),
                                            tf::frame_of(unit_vectors));

    start_p = end_p;
  };

  std::apply(
      [&](const auto &...unit_vectors) { (make_copy(unit_vectors), ...); },
      std::forward_as_tuple(unit_vectors0, unit_vectors1, unit_vectors...));
  return out;
}

/// @cond INTERNAL
namespace reindex {
template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon,
          typename Range>
auto concatenated_impl(tf::polygons_buffer<Index, RealT, Dims, Ngon> &out,
                       const Range &r) {
  Index start_p = 0;
  Index start_f = 0;

  auto make_copy = [&](const auto &polygons) {
    const Index end_p = start_p + static_cast<Index>(polygons.points().size());
    const Index end_f = start_f + static_cast<Index>(polygons.faces().size());

    const Index point_offset = start_p;
    tf::parallel_for_each(
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

    impl::copy_with_transformation(polygons.points(),
                                    tf::slice(out.points_buffer(), start_p, end_p),
                                    tf::frame_of(polygons));

    start_p = end_p;
    start_f = end_f;
  };

  for (const auto &polygons : r)
    make_copy(polygons);
  return out;
}

template <typename Index, typename Range>
auto concatenated_same_gons(const Range &r) {
  tf::polygons_buffer<Index, tf::coordinate_type<typename Range::value_type>,
                      tf::coordinate_dims_v<typename Range::value_type>,
                      tf::static_size_v<decltype(r[0].faces()[0])>>
      out;
  Index total_face_size = 0;
  Index total_point_size = 0;
  for (const auto &polygons : r) {
    total_face_size += polygons.size();
    total_point_size += polygons.points().size();
  }
  out.faces_buffer().allocate(total_face_size);
  out.points_buffer().allocate(total_point_size);

  concatenated_impl(out, r);
  return out;
}

template <typename Index, typename Range>
auto concatenated_diff_gons(const Range &r) {
  tf::polygons_buffer<Index, tf::coordinate_type<typename Range::value_type>,
                      tf::coordinate_dims_v<typename Range::value_type>,
                      tf::static_size_v<decltype(r[0].faces()[0])>>
      out;
  Index total_faces = 0;
  Index total_point_size = 0;
  for (const auto &polygons : r) {
    total_faces += polygons.size();
    total_point_size += polygons.points().size();
  }

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

  for (const auto &polygons : r)
    fill_offsets(polygons);

  out.faces_buffer().allocate(offsets.back());
  out.points_buffer().allocate(total_point_size);
  concatenated_impl(out, r);
  return out;
}

template <typename Policy, typename Range>
auto concatenated(const tf::polygons<Policy> &, const Range r) {
  using index_t = std::decay_t<decltype(r[0].faces()[0][0])>;
  constexpr bool all_same_gons =
      tf::static_size_v<decltype(r[0].faces()[0])> != tf::dynamic_size;
  if constexpr (all_same_gons)
    return tf::reindex::concatenated_same_gons<index_t>(r);
  else
    return tf::reindex::concatenated_diff_gons<index_t>(r);
}

template <typename Policy, typename Range>
auto concatenated(const tf::segments<Policy> &, const Range &r) {
  using Index = std::decay_t<decltype(r[0].edges()[0][0])>;
  tf::segments_buffer<Index, tf::coordinate_type<typename Range::value_type>,
                      tf::coordinate_dims_v<typename Range::value_type>>
      out;
  Index total_edge_size = 0;
  Index total_point_size = 0;
  for (const auto &segments : r) {
    total_edge_size += segments.size();
    total_point_size += segments.points().size();
  }
  out.edges_buffer().allocate(total_edge_size);
  out.points_buffer().allocate(total_point_size);

  Index start_p = 0;
  Index start_f = 0;

  auto make_copy = [&](const auto &segments) {
    const Index end_p = start_p + static_cast<Index>(segments.points().size());
    const Index end_f = start_f + static_cast<Index>(segments.edges().size());

    const Index point_offset = start_p;
    tf::parallel_for_each(
        tf::zip(segments.edges(),
                tf::slice(out.edges_buffer(), start_f, end_f)),
        [point_offset](auto pair) {
          auto &&in_edge = std::get<0>(pair);
          auto &&out_edge = std::get<1>(pair);
          out_edge[0] = in_edge[0] + point_offset;
          out_edge[1] = in_edge[1] + point_offset;
        },
        tf::checked);

    impl::copy_with_transformation(segments.points(),
                                    tf::slice(out.points_buffer(), start_p, end_p),
                                    tf::frame_of(segments));

    start_p = end_p;
    start_f = end_f;
  };

  for (const auto &segments : r)
    make_copy(segments);

  return out;
}

template <typename Policy, typename Range>
auto concatenated(const tf::points<Policy> &, const Range &r) {
  auto total_point_size = 0;
  for (const auto &points : r)
    total_point_size += points.size();
  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      out;
  out.allocate(total_point_size);

  std::size_t start_p = 0;

  auto make_copy = [&](const auto &points) {
    const std::size_t end_p = start_p + static_cast<std::size_t>(points.size());

    impl::copy_with_transformation(points, tf::slice(out, start_p, end_p),
                                   tf::frame_of(points));

    start_p = end_p;
  };

  for (const auto &points : r)
    make_copy(points);

  return out;
}

template <typename Policy, typename Range>
auto concatenated(const tf::vectors<Policy> &, const Range &r) {
  auto total_vector_size = 0;
  for (const auto &vectors : r)
    total_vector_size += vectors.size();
  tf::vectors_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      out;
  out.allocate(total_vector_size);

  std::size_t start_p = 0;

  auto make_copy = [&](const auto &vectors) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(vectors.size());

    impl::copy_with_transformation(vectors, tf::slice(out, start_p, end_p),
                                   tf::frame_of(vectors));

    start_p = end_p;
  };

  for (const auto &vectors : r)
    make_copy(vectors);
  return out;
}
template <typename Policy, typename Range>
auto concatenated(const tf::unit_vectors<Policy> &, const Range &r) {
  auto total_unit_vector_size = 0;
  for (const auto &unit_vectors : r)
    total_unit_vector_size += unit_vectors.size();
  tf::unit_vectors_buffer<tf::coordinate_type<Policy>,
                          tf::coordinate_dims_v<Policy>>
      out;
  out.allocate(total_unit_vector_size);

  std::size_t start_p = 0;

  auto make_copy = [&](const auto &unit_vectors) {
    const std::size_t end_p =
        start_p + static_cast<std::size_t>(unit_vectors.size());

    impl::copy_with_transformation(unit_vectors, tf::slice(out, start_p, end_p),
                                   tf::frame_of(unit_vectors));

    start_p = end_p;
  };

  for (const auto &unit_vectors : r)
    make_copy(unit_vectors);

  return out;
}
} // namespace reindex
/// @endcond

/// @ingroup reindex
/// @brief Concatenate a range of geometry collections.
///
/// Merges all geometry in the range into a single buffer.
/// Dispatches to the appropriate type-specific concatenation.
///
/// @tparam Iterator Range iterator type.
/// @tparam N Static size hint.
/// @param r Range of geometry collections (polygons, segments, points, vectors, or unit_vectors).
/// @return A buffer containing all concatenated geometry.
template <typename Iterator, std::size_t N>
auto concatenated(const tf::range<Iterator, N> &r) {
  return reindex::concatenated(r[0], r);
}

} // namespace tf
