#pragma once

#include <trueform/core/buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/segments_buffer.hpp>
#include <trueform/core/static_size.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/edge_mesh.hpp>
#include <trueform/cpp/core/edge_mesh_cache.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/core/offset_blocked_buffer.hpp>
#include <trueform/cpp/core/point_cloud.hpp>
#include <trueform/cpp/core/point_cloud_cache.hpp>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace consumer {

/// @brief What a CALLER holds when it holds a mesh.
///
/// The library owns no geometry: a caller holds core's own storage and a cache
/// it keeps, and `mesh()` is the assembly every entry takes. A consumer that
/// rewrites its storage says so on the cache, exactly as any caller does.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
struct owned_mesh {
  using index_type = Index;
  using real_type = Real;
  using mesh_type = tf::cpp::mesh<Index, Real, Dims, Ngon>;
  static constexpr std::size_t dims = Dims;
  static constexpr std::size_t face_arity = Ngon;

  tf::polygons_buffer<Index, Real, Dims, Ngon> polygons{};
  mutable tf::cpp::cache<Index, Real, Dims, Ngon> cache{};

  auto mesh() const -> mesh_type {
    return {polygons.faces(), polygons.points(), cache};
  }
};

/// @brief The same, for the edge carrier.
template <typename Index, typename Real, std::size_t Dims = 3>
struct owned_edge_mesh {
  using index_type = Index;
  using real_type = Real;
  using edge_mesh_type = tf::cpp::edge_mesh<Index, Real, Dims>;
  static constexpr std::size_t dims = Dims;

  tf::segments_buffer<Index, Real, Dims> segments{};
  mutable tf::cpp::edge_mesh_cache<Index, Real, Dims> cache{};

  auto edge_mesh() const -> edge_mesh_type {
    return {segments.edges(), segments.points(), cache};
  }
};

/// @brief The same, for the point carrier.
template <typename Real, std::size_t Dims = 3> struct owned_point_cloud {
  using real_type = Real;
  using point_cloud_type = tf::cpp::point_cloud<Real, Dims>;
  static constexpr std::size_t dims = Dims;

  tf::points_buffer<Real, Dims> points{};
  mutable tf::cpp::point_cloud_cache<Real, Dims> cache{};

  auto point_cloud() const -> point_cloud_type {
    return {points.points(), cache};
  }
};

/// @brief The mixed-arity carrier beside a triangle one of the same axes.
template <typename Owned>
using mixed_mesh_of =
    owned_mesh<typename Owned::index_type, typename Owned::real_type,
               Owned::dims, tf::dynamic_size>;

/// @brief The packed face indices of a mesh, whatever its layout states them
/// with: the sides of a face are consecutive in both.
template <typename Polygons> auto face_indices_of(const Polygons &value) {
  return tf::make_range(value.faces_buffer().data_buffer());
}

} // namespace consumer

namespace trueform_installed_test {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  auto output = storage.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

/// The one filler every fixture below states its storage through.
template <typename T, typename Values>
auto fill_storage(tf::buffer<T> &storage, const Values &values) -> void {
  storage.allocate(static_cast<std::size_t>(values.size()));
  std::copy(values.begin(), values.end(), storage.begin());
}

/// @brief Core's own storage, filled: what a caller holds when it holds a mesh.
template <typename Index, typename Real, std::size_t Dims = 3>
auto polygons_of(std::initializer_list<Index> indices,
                 std::initializer_list<Real> coordinates)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  tf::polygons_buffer<Index, Real, Dims, 3> out;
  fill_storage(out.faces_buffer().data_buffer(), indices);
  fill_storage(out.points_buffer().data_buffer(), coordinates);
  return out;
}

/// @overload The mixed arity, whose faces state their own offsets.
template <typename Index, typename Real, std::size_t Dims = 3>
auto polygons_of(std::initializer_list<Index> offsets,
                 std::initializer_list<Index> indices,
                 std::initializer_list<Real> coordinates)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> out;
  fill_storage(out.faces_buffer().offsets_buffer(), offsets);
  fill_storage(out.faces_buffer().data_buffer(), indices);
  fill_storage(out.points_buffer().data_buffer(), coordinates);
  return out;
}

template <typename Index, typename Real, std::size_t Dims = 3>
auto segments_of(std::initializer_list<Index> indices,
                 std::initializer_list<Real> coordinates)
    -> tf::segments_buffer<Index, Real, Dims> {
  tf::segments_buffer<Index, Real, Dims> out;
  fill_storage(out.edges_buffer().data_buffer(), indices);
  fill_storage(out.points_buffer().data_buffer(), coordinates);
  return out;
}

template <typename Real, std::size_t Dims = 3>
auto points_of(std::initializer_list<Real> coordinates)
    -> tf::points_buffer<Real, Dims> {
  tf::points_buffer<Real, Dims> out;
  fill_storage(out.data_buffer(), coordinates);
  return out;
}

template <typename Index, typename Real, std::size_t Dims>
auto open_triangle() -> consumer::owned_mesh<Index, Real, Dims> {
  if constexpr (Dims == 2)
    return {polygons_of<Index, Real, Dims>({0, 1, 2}, {0, 0, 1, 0, 0, 1})};
  else
    return {polygons_of<Index, Real, Dims>({0, 1, 2},
                                           {0, 0, 0, 1, 0, 0, 0, 1, 0})};
}

template <typename Index, typename Real>
auto negative_tetrahedron() -> consumer::owned_mesh<Index, Real, 3> {
  return {polygons_of<Index, Real, 3>(
      {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2},
      {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1})};
}

/// A mesh of no faces, at the arity the caller asks for.
template <typename Index, typename Real, std::size_t Ngon = 3>
auto empty_mesh_3d() -> consumer::owned_mesh<Index, Real, 3, Ngon> {
  if constexpr (Ngon == 3)
    return {polygons_of<Index, Real, 3>({}, {})};
  else
    return {polygons_of<Index, Real, 3>({Index{0}}, {}, {})};
}

/// The quad's points as an ARRAY, which is what the array-taking entries take.
template <typename Real, std::size_t Dims>
auto quad_points() -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return make_array<Real>({0, 0, 1, 0, 1, 1, 0, 1}, {4, 2});
  else
    return make_array<Real>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, {4, 3});
}

/// The same quad a caller HOLDS: one mixed face over core's own storage.
template <typename Index, typename Real, std::size_t Dims>
auto quad_mesh() -> consumer::owned_mesh<Index, Real, Dims, tf::dynamic_size> {
  if constexpr (Dims == 2)
    return {polygons_of<Index, Real, Dims>({0, 4}, {0, 1, 2, 3},
                                           {0, 0, 1, 0, 1, 1, 0, 1})};
  else
    return {polygons_of<Index, Real, Dims>(
        {0, 4}, {0, 1, 2, 3}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0})};
}

template <typename Index>
auto dynamic_quad_faces() -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>({0, 4}, {2}), make_array<Index>({0, 1, 2, 3}, {4}));
}

} // namespace trueform_installed_test
