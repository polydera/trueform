/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/face_link.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>
#include <trueform/topology/vertex_link.hpp>
#include <trueform/vtk/core/make_curves.hpp>
#include <trueform/vtk/core/make_normals.hpp>
#include <trueform/vtk/core/make_paths.hpp>
#include <trueform/vtk/core/make_points.hpp>
#include <trueform/vtk/core/make_polygons.hpp>
#include <trueform/vtk/core/make_polys.hpp>
#include <memory>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

class vtkInformationVector;

namespace tf::vtk {

/// @brief vtkPolyData subclass with cached trueform acceleration structures.
///
/// Inherits from vtkPolyData and adds lazy-built acceleration structures
/// (AABB trees, face membership, manifold edge link) that are automatically
/// invalidated when the underlying data changes.
///
/// Always uses dynamic-size polygons internally. For triangle meshes, use
/// set_as_triangles(true) to enable optimized triangle accessors.
///
/// Use SafeDownCast to detect trueform-enhanced polydata in VTK pipelines:
/// @code
/// if (auto* tf_poly = tf::vtk::polydata::SafeDownCast(input)) {
///   const auto& tree = tf_poly->poly_tree();  // access cached tree
/// }
/// @endcode
class polydata : public vtkPolyData {
public:
  vtkTypeMacro(polydata, vtkPolyData);
  static auto New() -> polydata *;

  /// @brief Retrieve polydata from information vector.
  /// Creates tf::vtk::polydata if needed.
  static auto GetData(vtkInformationVector *v, int i = 0) -> polydata *;

  /// @brief Shallow copy from another data object.
  /// If source is polydata, also shares cached structures.
  void ShallowCopy(vtkDataObject *src) override;

  /// @brief Mark data as pure triangles for optimized access.
  /// @param value True if all polygons are triangles.
  auto set_as_triangles(bool value) -> void;

  /// @brief Check if data is marked as triangles.
  auto is_triangles() const -> bool;

  /// @brief Get points view.
  auto points() -> points_t;

  /// @brief Get dynamic polygons view.
  auto polys() -> dynamic_polys_t;

  /// @brief Get triangle polygons view (only valid if is_triangles() is true).
  auto triangles() -> polys_t<3>;

  /// @brief Get paths/lines view.
  auto paths() -> paths_t;

  /// @brief Get polygons (faces + points).
  auto polygons() -> dynamic_polygons_t;

  /// @brief Get triangle polygons (faces + points). Only valid if
  /// is_triangles().
  auto triangle_polygons() -> polygons_t<3>;

  /// @brief Get curves (paths + points).
  auto curves() -> curves_t;

  /// @brief Get point normals view.
  /// @return A tf::unit_vectors view over point normals, or empty if none.
  auto point_normals() -> normals_t;

  /// @brief Get cell normals view.
  /// @return A tf::unit_vectors view over cell normals, or empty if none.
  auto cell_normals() -> normals_t;

  /// @brief Get AABB tree for polygons. Built lazily on first access.
  auto poly_tree() -> const tf::aabb_tree<vtkIdType, float, 3> &;

  /// @brief Get face membership structure. Built lazily on first access.
  auto face_membership() -> const tf::face_membership<vtkIdType> &;

  /// @brief Get manifold edge link (dynamic). Built lazily on first access.
  auto manifold_edge_link()
      -> const tf::manifold_edge_link<vtkIdType, tf::dynamic_size> &;

  /// @brief Get face link structure. Built lazily on first access.
  auto face_link() -> const tf::face_link<vtkIdType> &;

  /// @brief Get vertex link structure. Built lazily on first access.
  auto vertex_link() -> const tf::vertex_link<vtkIdType> &;

  /// @brief Get edges buffer from lines. Built lazily on first access.
  auto edges_buffer() -> const tf::blocked_buffer<vtkIdType, 2> &;

  /// @brief Get AABB tree for line segments. Built lazily on first access.
  auto segment_tree() -> const tf::aabb_tree<vtkIdType, float, 3> &;

  /// @brief Get AABB tree for points. Built lazily on first access.
  auto point_tree() -> const tf::aabb_tree<vtkIdType, float, 3> &;

protected:
  polydata();
  ~polydata() override = default;

private:
  auto build_poly_tree() -> void;
  auto build_face_membership() -> void;
  auto build_manifold_edge_link() -> void;
  auto build_face_link() -> void;
  auto build_vertex_link() -> void;
  auto build_edges_buffer() -> void;
  auto build_segment_tree() -> void;
  auto build_point_tree() -> void;

  bool _is_triangles = false;

  vtkMTimeType _poly_tree_mtime = 0;
  vtkMTimeType _fm_mtime = 0;
  vtkMTimeType _mel_mtime = 0;
  vtkMTimeType _fl_mtime = 0;
  vtkMTimeType _vl_mtime = 0;
  vtkMTimeType _edges_buffer_mtime = 0;
  vtkMTimeType _segment_tree_mtime = 0;
  vtkMTimeType _point_tree_mtime = 0;

  std::shared_ptr<tf::aabb_tree<vtkIdType, float, 3>> _poly_tree;
  std::shared_ptr<tf::face_membership<vtkIdType>> _fm;
  std::shared_ptr<tf::manifold_edge_link<vtkIdType, tf::dynamic_size>> _mel;
  std::shared_ptr<tf::face_link<vtkIdType>> _fl;
  std::shared_ptr<tf::vertex_link<vtkIdType>> _vl;
  std::shared_ptr<tf::blocked_buffer<vtkIdType, 2>> _edges_buffer;
  std::shared_ptr<tf::aabb_tree<vtkIdType, float, 3>> _segment_tree;
  std::shared_ptr<tf::aabb_tree<vtkIdType, float, 3>> _point_tree;

  polydata(const polydata &) = delete;
  void operator=(const polydata &) = delete;
};

} // namespace tf::vtk
