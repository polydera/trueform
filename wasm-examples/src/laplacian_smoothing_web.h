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
 * Author: Ziga Sajovic
 */
#pragma once
#include "main.h"
#include "trueform/core/hash_set.hpp"
#include "trueform/trueform.hpp"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"
#include <algorithm>
#include <array>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <limits>
#include <optional>
#include <vector>

namespace laplacian_smoothing {

// Colors (RGB bytes) - teal color scheme matching website
constexpr std::array<unsigned char, 3> white = {255, 255, 255};
constexpr std::array<unsigned char, 3> highlight = {100, 225, 210};     // Light teal (active)
constexpr std::array<unsigned char, 3> preview_color = {180, 238, 228}; // Lighter teal (preview)

} // namespace laplacian_smoothing

class tf_bridge_laplacian_smoothing : public tf_bridge_interface {
public:
  std::vector<unsigned char> vertex_colors; // RGB per vertex
  tf::vertex_link<int> vlink;
  tf::face_membership<int> fm;
  tf::aabb_mod_tree<int, float, 3> poly_tree;
  float aabb_diagonal = 1.0f;
  bool colors_updated = false;
  bool points_updated = false;

  auto initialize() -> void {
    auto &store = get_mesh_data_store();
    if (store.empty())
      return;

    auto &data = store[0];
    auto points = data.polygons.points();
    const auto n_vertices = points.size();

    // Allocate vertex colors (RGB per vertex)
    vertex_colors.resize(n_vertices * 3);
    for (std::size_t i = 0; i < n_vertices; ++i) {
      vertex_colors[i * 3 + 0] = laplacian_smoothing::white[0];
      vertex_colors[i * 3 + 1] = laplacian_smoothing::white[1];
      vertex_colors[i * 3 + 2] = laplacian_smoothing::white[2];
    }

    // Build face membership and vertex link for neighborhood queries
    fm.build(data.polygons.polygons());
    vlink.build(data.polygons.polygons(), fm);

    // Build mod_tree for spatial queries on polygons
    poly_tree.build(data.polygons.polygons(), tf::config_tree(4, 4));

    // Compute AABB diagonal once
    auto aabb = tf::aabb_from(points);
    aabb_diagonal = aabb.diagonal().length();

    colors_updated = true;
  }

  auto get_vertex_colors_view() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        vertex_colors.size(), vertex_colors.data()));
  }

  auto get_points_view() -> emscripten::val {
    auto &store = get_mesh_data_store();
    if (store.empty())
      return emscripten::val::undefined();
    auto &data = store[0];
    return emscripten::val(emscripten::typed_memory_view(
        data.polygons.points_buffer().data_buffer().size(),
        data.polygons.points_buffer().data_buffer().begin()));
  }
};

class cursor_interactor_laplacian_smoothing final
    : public cursor_interactor_interface {
public:
  cursor_interactor_laplacian_smoothing()
      : cursor_interactor_interface(
            std::make_unique<tf_bridge_laplacian_smoothing>()) {}

  auto get_smoothing_bridge() -> tf_bridge_laplacian_smoothing * {
    return dynamic_cast<tf_bridge_laplacian_smoothing *>(bridge.get());
  }

  auto colors_updated() -> bool {
    auto *b = get_smoothing_bridge();
    if (!b)
      return false;
    bool updated = b->colors_updated;
    b->colors_updated = false;
    return updated;
  }

  auto points_updated() -> bool {
    auto *b = get_smoothing_bridge();
    if (!b)
      return false;
    bool updated = b->points_updated;
    b->points_updated = false;
    return updated;
  }

  auto get_vertex_colors() -> emscripten::val {
    auto *b = get_smoothing_bridge();
    if (!b)
      return emscripten::val::undefined();
    return b->get_vertex_colors_view();
  }

  auto get_points() -> emscripten::val {
    auto *b = get_smoothing_bridge();
    if (!b)
      return emscripten::val::undefined();
    return b->get_points_view();
  }

  auto set_radius(float r) -> void {
    // Clamp radius between 1% and 10% of AABB diagonal
    float diag = get_aabb_diagonal();
    float min_r = diag * 0.01f;
    float max_r = diag * 0.10f;
    radius_ = std::clamp(r, min_r, max_r);
  }
  auto set_lambda(float l) -> void { lambda_ = l; }

  auto get_aabb_diagonal() -> float {
    auto *b = get_smoothing_bridge();
    return b ? b->aabb_diagonal : 1.0f;
  }

  auto OnLeftButtonDown() -> bool override {
    // Check if we're over the mesh (tracked from last OnMouseMove)
    if (last_hit_vertex_) {
      // Hit mesh - enter painting mode and apply first brush stroke
      painting_ = true;
      tf::tick();
      update_brush(*last_hit_vertex_);
      add_update_time(tf::tock());
      return true;
    }

    // Miss mesh - let camera handle it
    return false;
  }

  auto OnLeftButtonUp() -> bool override {
    if (painting_) {
      painting_ = false;
      // Recolor active brush to preview color instead of clearing
      recolor_to_preview();
      // Full rebuild on mouse up for optimal tree structure
      rebuild_tree();
      return true;
    }
    return false;
  }

  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    (void)camera_position;
    (void)camera_focal_point;

    auto *b = get_smoothing_bridge();
    if (!b || get_mesh_data_store().empty())
      return false;

    tf::ray<float, 3> ray{origin, direction};

    if (!painting_) {
      // Not painting - show brush preview
      tf::tick();
      show_preview(ray);
      add_update_time(tf::tock());
      // Return false if not over mesh to let camera handle interaction
      return last_hit_vertex_.has_value();
    }

    // Painting mode - smooth and highlight
    auto hit = try_pick(ray);
    if (hit) {
      tf::tick();
      update_brush(*hit);
      add_update_time(tf::tock());
      return true;
    } else {
      // Moved off mesh while painting - clear highlight
      clear_highlight();
      return true;
    }
  }

private:
  float radius_ = 1.0f;
  float lambda_ = 0.3f;
  bool painting_ = false;
  std::optional<int> last_hit_vertex_;
  tf::topology::neighborhood_applier<int> applier_;
  std::vector<int> current_indices_;
  std::vector<int> preview_indices_;
  std::vector<int> polygon_ids_;
  tf::hash_set<int> polygon_set_;
  std::vector<float> update_times_;

  auto add_update_time(float t) -> void { m_time = add_time(update_times_, t); }

  // Try to pick a vertex on the mesh, returns closest vertex id if hit
  auto try_pick(const tf::ray<float, 3> &ray) -> std::optional<int> {
    auto *b = get_smoothing_bridge();
    if (!b || get_instances().empty())
      return std::nullopt;

    auto &data = get_mesh_data_store()[0];
    auto &inst = get_instances()[0];

    // Use our poly_tree (which is kept updated) for ray casting
    auto form =
        data.polygons.polygons() | tf::tag(b->poly_tree) | tf::tag(inst.frame);
    auto ray_result = tf::ray_cast(ray, form, tf::ray_config<float>{});

    if (!ray_result)
      return std::nullopt;

    auto face = data.polygons.faces()[ray_result.element];
    auto points = data.polygons.points();
    auto hit_point = ray.origin + ray_result.info.t * ray.direction;

    // Find closest vertex in the face
    int closest_vertex = face[0];
    float min_dist2 = std::numeric_limits<float>::max();

    for (auto vid : face) {
      float d2 = tf::distance2(points[vid], hit_point);
      if (d2 < min_dist2) {
        min_dist2 = d2;
        closest_vertex = vid;
      }
    }

    return closest_vertex;
  }

  auto clear_highlight() -> void {
    auto *b = get_smoothing_bridge();
    if (!b || current_indices_.empty())
      return;

    auto &data = get_mesh_data_store()[0];
    const auto n_points = data.polygons.points().size();

    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    auto neigh_colors = tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, laplacian_smoothing::white);

    current_indices_.clear();
    b->colors_updated = true;
  }

  auto clear_preview() -> void {
    auto *b = get_smoothing_bridge();
    if (!b || preview_indices_.empty())
      return;

    auto &data = get_mesh_data_store()[0];
    const auto n_points = data.polygons.points().size();

    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    auto preview_colors =
        tf::make_indirect_range(preview_indices_, colors_range);
    tf::parallel_fill(preview_colors, laplacian_smoothing::white);

    preview_indices_.clear();
    b->colors_updated = true;
  }

  auto show_preview(const tf::ray<float, 3> &ray) -> void {
    auto *b = get_smoothing_bridge();
    if (!b)
      return;

    auto hit = try_pick(ray);
    if (!hit) {
      last_hit_vertex_ = std::nullopt;
      clear_preview();
      return;
    }

    // Track hit for OnLeftButtonDown
    last_hit_vertex_ = hit;

    auto &data = get_mesh_data_store()[0];
    auto points = data.polygons.points();
    const auto n_points = points.size();

    // Clear previous preview
    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    if (!preview_indices_.empty()) {
      auto prev_colors =
          tf::make_indirect_range(preview_indices_, colors_range);
      tf::parallel_fill(prev_colors, laplacian_smoothing::white);
    }

    // Collect new preview neighborhood
    preview_indices_.clear();
    applier_(
        b->vlink, *hit,
        [&points](int seed, int neighbor) {
          return tf::distance2(points[seed], points[neighbor]);
        },
        radius_,
        [this](int idx) { preview_indices_.push_back(idx); }, true);

    // Highlight preview
    auto preview_colors =
        tf::make_indirect_range(preview_indices_, colors_range);
    tf::parallel_fill(preview_colors, laplacian_smoothing::preview_color);

    b->colors_updated = true;
  }

  auto recolor_to_preview() -> void {
    auto *b = get_smoothing_bridge();
    if (!b || current_indices_.empty())
      return;

    auto &data = get_mesh_data_store()[0];
    const auto n_points = data.polygons.points().size();

    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    auto neigh_colors = tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, laplacian_smoothing::preview_color);

    // Transfer to preview indices
    preview_indices_ = std::move(current_indices_);
    current_indices_.clear();
    b->colors_updated = true;
  }

  auto update_brush(int closest_vertex) -> void {
    auto *b = get_smoothing_bridge();
    if (!b)
      return;

    auto &data = get_mesh_data_store()[0];
    auto points = data.polygons.points();
    const auto n_points = points.size();

    // Restore previous highlight
    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    if (!current_indices_.empty()) {
      auto prev_colors =
          tf::make_indirect_range(current_indices_, colors_range);
      tf::parallel_fill(prev_colors, laplacian_smoothing::white);
    }

    // Collect neighborhood and dirty polygon IDs
    current_indices_.clear();
    polygon_set_.clear();
    polygon_ids_.clear();

    applier_(
        b->vlink, closest_vertex,
        [&points](int seed, int neighbor) {
          return tf::distance2(points[seed], points[neighbor]);
        },
        radius_,
        [this, b](int idx) {
          current_indices_.push_back(idx);
          for (auto poly_id : b->fm[idx]) {
            if (polygon_set_.insert(poly_id).second)
              polygon_ids_.push_back(poly_id);
          }
        },
        true);

    // Highlight neighborhood with active brush color
    auto neigh_colors = tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, laplacian_smoothing::highlight);

    // Apply Laplacian smoothing
    auto neigh_points = tf::make_indirect_range(current_indices_, points);
    auto neigh_neighbors = tf::make_indirect_range(
        current_indices_, tf::make_block_indirect_range(b->vlink, points));

    tf::parallel_for_each(
        tf::zip(neigh_points, neigh_neighbors),
        [this](auto tup) {
          auto [pt, neighbors] = tup;
          pt = tf::laplacian_smoothed(pt, tf::make_points(neighbors), lambda_);
        },
        tf::checked);

    // Update mod_tree incrementally
    tf::hash_set<int> dirty_set(polygon_ids_.begin(), polygon_ids_.end());
    auto keep_if = [&dirty_set](int id) {
      return dirty_set.find(id) == dirty_set.end();
    };
    b->poly_tree.update(data.polygons.polygons(), polygon_ids_, keep_if,
                        tf::config_tree(4, 4));

    b->colors_updated = true;
    b->points_updated = true;
  }

  auto rebuild_tree() -> void {
    auto *b = get_smoothing_bridge();
    if (!b)
      return;

    auto &data = get_mesh_data_store()[0];
    b->poly_tree.build(data.polygons.polygons(), tf::config_tree(4, 4));
  }
};

inline int run_main_laplacian_smoothing(std::string path) {
  interactor = std::make_unique<cursor_interactor_laplacian_smoothing>();

  // Load mesh
  auto poly = tf::read_stl<int>(path);
  if (!poly.size()) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  utils::center_and_scale_p(poly);

  // Add mesh data (with topology for vertex_link)
  auto mesh_id = interactor->add_mesh_data(std::move(poly), true);
  interactor->add_instance(mesh_id);

  // Initialize smoothing structures
  if (auto *smoothing_interactor =
          dynamic_cast<cursor_interactor_laplacian_smoothing *>(
              interactor.get())) {
    smoothing_interactor->get_smoothing_bridge()->initialize();

    // Set radius as percentage of AABB diagonal (5%, range 1-10%)
    float diag = smoothing_interactor->get_aabb_diagonal();
    smoothing_interactor->set_radius(diag * 0.05f);
  }

  return 0;
}
