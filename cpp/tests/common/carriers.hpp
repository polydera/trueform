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

#include "trueform/core/buffer.hpp"
#include "trueform/core/points_buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/core/unit_vectors_buffer.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/edge_mesh_cache.hpp"
#include "trueform/cpp/core/identity_transformation.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/core/point_cloud_cache.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace tf::cpp::test {

/// The one filler every builder below states its storage through.
template <typename T, typename Values>
auto fill_storage(tf::buffer<T> &storage, const Values &values) -> void {
  storage.allocate(static_cast<std::size_t>(values.size()));
  std::copy(values.begin(), values.end(), storage.begin());
}

/// @brief Core's own storage, filled: what a caller holds when it holds a mesh.
///
/// Each builder takes a braced list or any container a test already holds, so
/// a fixture that computes its coordinates hands them over as they stand.
template <typename Index, typename Real, std::size_t Dims = 3, typename Indices,
          typename Coordinates>
auto polygons_of(const Indices &indices, const Coordinates &coordinates)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  tf::polygons_buffer<Index, Real, Dims, 3> out;
  fill_storage(out.faces_buffer().data_buffer(), indices);
  fill_storage(out.points_buffer().data_buffer(), coordinates);
  return out;
}

template <typename Index, typename Real, std::size_t Dims = 3>
auto polygons_of(std::initializer_list<Index> indices,
                 std::initializer_list<Real> coordinates)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  return polygons_of<Index, Real, Dims, std::initializer_list<Index>,
                     std::initializer_list<Real>>(indices, coordinates);
}

/// @overload The mixed arity, whose faces state their own offsets.
template <typename Index, typename Real, std::size_t Dims = 3, typename Offsets,
          typename Indices, typename Coordinates>
auto polygons_of(const Offsets &offsets, const Indices &indices,
                 const Coordinates &coordinates)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> out;
  fill_storage(out.faces_buffer().offsets_buffer(), offsets);
  fill_storage(out.faces_buffer().data_buffer(), indices);
  fill_storage(out.points_buffer().data_buffer(), coordinates);
  return out;
}

template <typename Index, typename Real, std::size_t Dims = 3>
auto polygons_of(std::initializer_list<Index> offsets,
                 std::initializer_list<Index> indices,
                 std::initializer_list<Real> coordinates)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> {
  return polygons_of<Index, Real, Dims, std::initializer_list<Index>,
                     std::initializer_list<Index>, std::initializer_list<Real>>(
      offsets, indices, coordinates);
}

template <typename Index, typename Real, std::size_t Dims = 3, typename Indices,
          typename Coordinates>
auto segments_of(const Indices &indices, const Coordinates &coordinates)
    -> tf::segments_buffer<Index, Real, Dims> {
  tf::segments_buffer<Index, Real, Dims> out;
  fill_storage(out.edges_buffer().data_buffer(), indices);
  fill_storage(out.points_buffer().data_buffer(), coordinates);
  return out;
}

template <typename Index, typename Real, std::size_t Dims = 3>
auto segments_of(std::initializer_list<Index> indices,
                 std::initializer_list<Real> coordinates)
    -> tf::segments_buffer<Index, Real, Dims> {
  return segments_of<Index, Real, Dims, std::initializer_list<Index>,
                     std::initializer_list<Real>>(indices, coordinates);
}

template <typename Real, std::size_t Dims = 3, typename Coordinates>
auto points_of(const Coordinates &coordinates)
    -> tf::points_buffer<Real, Dims> {
  tf::points_buffer<Real, Dims> out;
  fill_storage(out.data_buffer(), coordinates);
  return out;
}

template <typename Real, std::size_t Dims = 3>
auto points_of(std::initializer_list<Real> coordinates)
    -> tf::points_buffer<Real, Dims> {
  return points_of<Real, Dims, std::initializer_list<Real>>(coordinates);
}

/// @brief Where a test puts its carrier: the sixteen (or nine) numbers a
/// placement is, and the frame a reading takes them as.
///
/// One producer for all three carriers — a placement is the same fact whatever
/// it places, and `place()` COPIES into this slot, so the caller's own array is
/// its own again the moment it returns.
template <typename Real, std::size_t Dims> struct placement_slot {
  std::array<Real, (Dims + 1) * (Dims + 1)> values{};
  bool placed = false;

  auto frame() const -> tf::transformation_view<const Real, Dims> {
    return placed ? tf::make_transformation_view<Dims>(values.data())
                  : identity_transformation_view<Real, Dims>();
  }

  auto place(const std::array<Real, (Dims + 1) * (Dims + 1)> &stated) -> void {
    values = stated;
    placed = true;
  }
};

/// @brief What a TEST owns when it holds a mesh.
///
/// The library owns nothing: a caller holds its geometry, its cache and its
/// placement, and assembles. So a test holds the same three and `mesh()` is
/// that assembly — the one entrance every entry takes. A test that MOVES a
/// point writes through `polygons` and then says `cache.points_changed()`,
/// which is the protocol itself.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
struct owned_mesh {
  using index_type = Index;
  using real_type = Real;
  using mesh_type = cpp::mesh<Index, Real, Dims, Ngon>;
  static constexpr std::size_t dims = Dims;
  static constexpr std::size_t face_arity = Ngon;

  tf::polygons_buffer<Index, Real, Dims, Ngon> polygons{};
  mutable cpp::cache<Index, Real, Dims, Ngon> cache{};
  placement_slot<Real, Dims> placement{};

  auto mesh() const -> mesh_type {
    return {polygons.faces(), polygons.points(), cache, frame()};
  }

  auto frame() const -> tf::transformation_view<const Real, Dims> {
    return placement.frame();
  }

  auto place(const std::array<Real, (Dims + 1) * (Dims + 1)> &values) -> void {
    placement.place(values);
  }
};

/// @brief The same, for the edge carrier.
template <typename Index, typename Real, std::size_t Dims = 3>
struct owned_edge_mesh {
  using index_type = Index;
  using real_type = Real;
  using edge_mesh_type = cpp::edge_mesh<Index, Real, Dims>;
  static constexpr std::size_t dims = Dims;

  tf::segments_buffer<Index, Real, Dims> segments{};
  mutable cpp::edge_mesh_cache<Index, Real, Dims> cache{};
  placement_slot<Real, Dims> placement{};

  auto edge_mesh() const -> edge_mesh_type {
    return {segments.edges(), segments.points(), cache, frame()};
  }

  auto frame() const -> tf::transformation_view<const Real, Dims> {
    return placement.frame();
  }

  auto place(const std::array<Real, (Dims + 1) * (Dims + 1)> &values) -> void {
    placement.place(values);
  }
};

/// @brief The same, for the point carrier, whose normals are its points'.
template <typename Real, std::size_t Dims = 3> struct owned_point_cloud {
  using real_type = Real;
  using point_cloud_type = cpp::point_cloud<Real, Dims>;
  static constexpr std::size_t dims = Dims;

  tf::points_buffer<Real, Dims> points{};
  tf::unit_vectors_buffer<Real, Dims> normals{};
  mutable cpp::point_cloud_cache<Real, Dims> cache{};
  placement_slot<Real, Dims> placement{};

  auto point_cloud() const -> point_cloud_type {
    return {points.points(), normals.unit_vectors(), cache, frame()};
  }

  auto frame() const -> tf::transformation_view<const Real, Dims> {
    return placement.frame();
  }

  auto place(const std::array<Real, (Dims + 1) * (Dims + 1)> &values) -> void {
    placement.place(values);
  }
};

/// @brief A minted result, read as an operand of its own.
///
/// Every entry hands back core's own storage, so a check that wants a
/// measurement of one assembles over it exactly as a caller would. This one
/// TAKES the storage, and the cache and the identity placement come with it.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
auto operand_of(tf::polygons_buffer<Index, Real, Dims, Ngon> polygons)
    -> owned_mesh<Index, Real, Dims, Ngon> {
  owned_mesh<Index, Real, Dims, Ngon> result;
  result.polygons = std::move(polygons);
  return result;
}

/// @brief The same, over storage a check still holds.
///
/// Both the buffer and the cache are the caller's, so a check binds them and
/// then takes the reading — the borrow law the assembly states. One producer
/// of that assembly for every suite that measures a result in place.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reading_over(const tf::polygons_buffer<Index, Real, Dims, Ngon> &polygons,
                  cpp::cache<Index, Real, Dims, Ngon> &cache)
    -> cpp::mesh<Index, Real, Dims, Ngon> {
  return {polygons.faces(), polygons.points(), cache};
}

} // namespace tf::cpp::test
