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
#include <trueform/core.hpp>
#include <trueform/spatial/aabb_mod_tree.hpp>
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
#include <trueform/vtk/core/make_segments.hpp>
#include <trueform/vtk/core/tree_index_map.hpp>
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
/// Always uses dynamic-size polygons internally.
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

  /// @brief Get points view.
  auto points() -> points_t;

  /// @brief Get polys view.
  auto polys() -> polys_t;

  /// @brief Get paths/lines view.
  auto paths() -> paths_t;

  /// @brief Get polygons (faces + points).
  auto polygons() -> polygons_t;

  /// @brief Get curves (paths + points).
  auto curves() -> curves_t;

  /// @brief Get edges view (from lines). Built lazily on first access.
  auto edges() -> edges_t;

  /// @brief Get segments view (edges + points). Built lazily on first access.
  auto segments() -> segments_t;

  /// @brief Get point normals view.
  /// @return A tf::unit_vectors view over point normals, or empty if none.
  auto point_normals() -> normals_t;

  /// @brief Get cell normals view.
  /// @return A tf::unit_vectors view over cell normals, or empty if none.
  auto cell_normals() -> normals_t;

  /// @brief Get AABB tree for polygons. Built lazily on first access.
  auto poly_tree() -> const tf::aabb_mod_tree<vtkIdType, float, 3> &;

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
  auto segment_tree() -> const tf::aabb_mod_tree<vtkIdType, float, 3> &;

  /// @brief Reset segment_tree (forces rebuild on next access).
  auto reset_segment_tree() -> void;

  /// @brief Update segment_tree incrementally with dirty segment IDs.
  /// @param dirty_ids Range of segment IDs whose geometry changed.
  auto update_segment_tree(tf::range<vtkIdType *, tf::dynamic_size> dirty_ids)
      -> void;

  /// @brief Update segment_tree incrementally with dirty segment IDs.
  /// @param dirty_ids Vector of segment IDs whose geometry changed.
  auto update_segment_tree(const std::vector<vtkIdType> &dirty_ids) -> void {
    update_segment_tree(tf::make_range(
        const_cast<vtkIdType *>(dirty_ids.data()), dirty_ids.size()));
  }

  /// @brief Update segment_tree incrementally with tree_index_map.
  /// @param tree_map Index map for remapping scenarios.
  auto update_segment_tree(const tree_index_map_t &tree_map) -> void;

  /// @brief Get AABB tree for points. Built lazily on first access.
  auto point_tree() -> const tf::aabb_mod_tree<vtkIdType, float, 3> &;

  /// @brief Reset poly_tree (forces rebuild on next access).
  auto reset_poly_tree() -> void;

  /// @brief Reset point_tree (forces rebuild on next access).
  auto reset_point_tree() -> void;

  /// @brief Update point_tree incrementally with dirty point IDs.
  /// @param dirty_ids Range of point IDs whose geometry changed.
  auto update_point_tree(tf::range<vtkIdType *, tf::dynamic_size> dirty_ids)
      -> void;

  /// @brief Update point_tree incrementally with dirty point IDs.
  /// @param dirty_ids Vector of point IDs whose geometry changed.
  auto update_point_tree(const std::vector<vtkIdType> &dirty_ids) -> void {
    update_point_tree(tf::make_range(const_cast<vtkIdType *>(dirty_ids.data()),
                                     dirty_ids.size()));
  }

  /// @brief Update point_tree incrementally with tree_index_map.
  /// @param tree_map Index map for remapping scenarios.
  auto update_point_tree(const tree_index_map_t &tree_map) -> void;

  /// @brief Update poly_tree incrementally with dirty polygon IDs.
  /// @param dirty_ids Range of polygon IDs whose geometry changed.
  auto update_poly_tree(tf::range<vtkIdType *, tf::dynamic_size> dirty_ids)
      -> void;

  /// @brief Update poly_tree incrementally with dirty polygon IDs.
  /// @param dirty_ids Vector of polygon IDs whose geometry changed.
  auto update_poly_tree(const std::vector<vtkIdType> &dirty_ids) -> void {
    update_poly_tree(tf::make_range(const_cast<vtkIdType *>(dirty_ids.data()),
                                    dirty_ids.size()));
  }

  /// @brief Update poly_tree incrementally with tree_index_map.
  /// @param tree_map Index map for remapping scenarios.
  auto update_poly_tree(const tree_index_map_t &tree_map) -> void;

  /// @brief Mark face_membership as modified (prevents rebuild on next access).
  auto modified_face_membership() -> void;

  /// @brief Mark manifold_edge_link as modified (prevents rebuild on next access).
  auto modified_manifold_edge_link() -> void;

  /// @brief Mark face_link as modified (prevents rebuild on next access).
  auto modified_face_link() -> void;

  /// @brief Mark vertex_link as modified (prevents rebuild on next access).
  auto modified_vertex_link() -> void;

  /// @brief Mark edges_buffer as modified (prevents rebuild on next access).
  auto modified_edges_buffer() -> void;

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

  vtkMTimeType _poly_tree_mtime = 0;
  vtkMTimeType _fm_mtime = 0;
  vtkMTimeType _mel_mtime = 0;
  vtkMTimeType _fl_mtime = 0;
  vtkMTimeType _vl_mtime = 0;
  vtkMTimeType _edges_buffer_mtime = 0;
  vtkMTimeType _segment_tree_mtime = 0;
  vtkMTimeType _point_tree_mtime = 0;

  std::shared_ptr<tf::aabb_mod_tree<vtkIdType, float, 3>> _poly_tree;
  std::shared_ptr<tf::face_membership<vtkIdType>> _fm;
  std::shared_ptr<tf::manifold_edge_link<vtkIdType, tf::dynamic_size>> _mel;
  std::shared_ptr<tf::face_link<vtkIdType>> _fl;
  std::shared_ptr<tf::vertex_link<vtkIdType>> _vl;
  std::shared_ptr<tf::blocked_buffer<vtkIdType, 2>> _edges_buffer;
  std::shared_ptr<tf::aabb_mod_tree<vtkIdType, float, 3>> _segment_tree;
  std::shared_ptr<tf::aabb_mod_tree<vtkIdType, float, 3>> _point_tree;

  polydata(const polydata &) = delete;
  void operator=(const polydata &) = delete;
};

} // namespace tf::vtk
