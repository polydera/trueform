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
#include <trueform/core/faces.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/identity_transformation.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/vtk/cpp/source/require_mesh_source.hpp>

#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkType.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace tf::vtk {

/// @brief A vtkPolyData read as a trueform mesh.
///
/// It owns a cache and nothing else. The geometry stays where VTK put it, the
/// mesh borrows the geometry and the cache both, and this source is what states
/// the cache's changes: a polydata's own modification times are read back and
/// turned into `points_changed()` / `faces_changed()`, so a caller that edits
/// the polydata gets structures built for what it edited, and one that does not
/// gets the ones it already paid for.
///
/// Faces are read at dynamic arity because that is what a vtkPolyData states:
/// a cell array is offsets and connectivity, and every polygon in it may have
/// a different number of points.
///
/// ASKING IS A WRITE. `mesh()` reads VTK's modification times and states what
/// they say to the cache, so it mutates this source and may fill the cache. It
/// is the same contract the cache carries: one thread asks a source for its
/// mesh, and what it hands back is that thread's reading.
template <typename Real> class mesh_source {
public:
  /// The matrix spells its indices `std::int32_t` / `std::int64_t`, and
  /// `vtkIdType` is whatever VTK was built with — the same width and
  /// signedness, but on an LP64 target a DISTINCT type the archive carries no
  /// carrier for. So a source reads at the matrix's own index of that width,
  /// and VTK's ids convert at the seam.
  using index_type =
      std::conditional_t<sizeof(vtkIdType) == 8, std::int64_t, std::int32_t>;
  static_assert(std::is_integral_v<vtkIdType> && std::is_signed_v<vtkIdType> &&
                    sizeof(index_type) == sizeof(vtkIdType),
                "vtkIdType must be a signed integer of a width the C++ facade "
                "matrix carries");

  using cache_type = tf::cpp::cache<index_type, Real, 3, tf::dynamic_size>;
  using mesh_type = tf::cpp::mesh<index_type, Real, 3, tf::dynamic_size>;

  mesh_source() = delete;

  /// @brief Read a polydata whose memory this layer does not own.
  ///
  /// The four facts a borrow depends on are stated here, so a source that
  /// cannot be read where it lies says so once rather than at every use.
  explicit mesh_source(vtkPolyData *polydata)
      : _source(readable(polydata)), _cache(std::make_shared<cache_type>()),
        _keepalive(
            std::make_shared<const kept_alive>(kept_alive{_source, _cache})),
        _readable_cells_stamp(cell_mtime(polydata)),
        _readable_points_stamp(point_mtime(polydata)),
        _readable_strips_stamp(strip_mtime(polydata)) {}

  auto polydata() const -> vtkPolyData * { return _source.Get(); }

  /// @brief The mesh this polydata is, at the moment it is asked.
  ///
  /// A source is live, so the four facts a borrow depends on are asked again
  /// whenever it says something changed, and the fifth — that every id names a
  /// point the source has — is the cache's, which remembers it per reading.
  auto mesh() const -> mesh_type {
    state_changes();
    const auto value = mesh_type(
        borrowed_faces(), borrowed_points(), *_cache,
        tf::cpp::identity_transformation_view<Real, 3>(), keepalive());
    value.require_indices();
    return value;
  }

  auto cache() const -> cache_type & { return *_cache; }

private:
  /// What a mesh borrows is the polydata's arrays and this source's cache, so
  /// the keepalive is both. A vtkPolyData is refcounted and the cache is
  /// held behind a handle, so retaining them is retaining what the view reads —
  /// and a mesh read off a temporary source keeps answering.
  struct kept_alive {
    vtkSmartPointer<vtkPolyData> polydata;
    std::shared_ptr<cache_type> cache;
  };

  /// The four facts a borrow depends on, asked before anything is built over
  /// the memory they are about.
  static auto readable(vtkPolyData *polydata) -> vtkPolyData * {
    if (!polydata)
      throw std::invalid_argument("mesh source: polydata must not be null");
    source::require_readable<Real>(polydata);
    return polydata;
  }

  static auto cell_mtime(vtkPolyData *polydata) -> std::uint64_t {
    auto *cells = polydata ? polydata->GetPolys() : nullptr;
    return static_cast<std::uint64_t>(cells ? cells->GetMTime() : 0);
  }

  static auto point_mtime(vtkPolyData *polydata) -> std::uint64_t {
    auto *points = polydata ? polydata->GetPoints() : nullptr;
    return static_cast<std::uint64_t>(points ? points->GetMTime() : 0);
  }

  /// Strips are one of the four facts — they are cells this source does not
  /// read — so what states them is watched like the other three.
  static auto strip_mtime(vtkPolyData *polydata) -> std::uint64_t {
    auto *strips = polydata ? polydata->GetStrips() : nullptr;
    return static_cast<std::uint64_t>(strips ? strips->GetMTime() : 0);
  }

  /// A source is the caller's own statement of what moved: VTK's modification
  /// times are the only thing here that knows, so they are read back and stated
  /// to the cache, which is the one channel the generations have.
  auto state_changes() const -> void {
    const auto cells = cell_mtime(_source);
    const auto points = point_mtime(_source);
    const auto strips = strip_mtime(_source);
    if (_readable_cells_stamp == cells && _readable_points_stamp == points &&
        _readable_strips_stamp == strips)
      return;
    source::require_readable<Real>(_source);
    if (_readable_cells_stamp != cells)
      _cache->faces_changed();
    if (_readable_points_stamp != points)
      _cache->points_changed();
    _readable_cells_stamp = cells;
    _readable_points_stamp = points;
    _readable_strips_stamp = strips;
  }

  /// Both of what it holds are fixed for this source's lifetime, so the
  /// keepalive a reading takes is built once and handed out, not minted per
  /// reading.
  auto keepalive() const -> std::shared_ptr<const void> { return _keepalive; }

  /// The views are BORROWED over VTK's own arrays: nothing is copied, and the
  /// mesh's own handle is what keeps the polydata alive. A polydata with no
  /// polys states no offsets at all, which is the empty reading a mesh already
  /// reads as no faces.
  auto borrowed_faces() const -> typename mesh_type::faces_type {
    auto *cells = _source->GetPolys();
    return tf::make_faces(
        tf::make_range(static_cast<const index_type *>(
                           source::cell_offsets<index_type>(cells)),
                       source::cell_offset_count(cells)),
        tf::make_range(static_cast<const index_type *>(
                           source::cell_connectivity<index_type>(cells)),
                       source::cell_connectivity_count(cells)));
  }

  auto borrowed_points() const -> typename mesh_type::points_type {
    auto *points = _source->GetPoints();
    const auto point_count = points ? points->GetNumberOfPoints() : 0;
    return tf::make_points<3>(tf::make_range(
        static_cast<const Real *>(source::interleaved_points<Real>(_source)),
        static_cast<std::size_t>(point_count) * 3));
  }

  vtkSmartPointer<vtkPolyData> _source;
  std::shared_ptr<cache_type> _cache;
  std::shared_ptr<const void> _keepalive;
  mutable std::uint64_t _readable_cells_stamp = 0;
  mutable std::uint64_t _readable_points_stamp = 0;
  mutable std::uint64_t _readable_strips_stamp = 0;
};

/// @brief Read a polydata as a mesh source.
template <typename Real>
auto make_mesh_source(vtkPolyData *polydata) -> mesh_source<Real> {
  return mesh_source<Real>(polydata);
}

} // namespace tf::vtk
