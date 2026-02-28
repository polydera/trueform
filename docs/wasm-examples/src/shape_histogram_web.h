#pragma once
#include "main.h"
#include "trueform/trueform.hpp"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"
#include <algorithm>
#include <array>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>

// Histogram parameters
constexpr int num_bins = 10;
constexpr float bin_width = 2.0f / num_bins; // Range [-1, 1]

// Colors (RGB bytes)
constexpr std::array<unsigned char, 3> white = {255, 255, 255};
constexpr std::array<unsigned char, 3> highlight = {63, 255, 233}; // Cyan #3fffe9

class tf_bridge_shape_histogram : public tf_bridge_interface {
public:
  tf::buffer<float> shape_index;
  std::vector<unsigned char> vertex_colors; // RGB per vertex
  std::array<int, num_bins> histogram_bins{};
  tf::vertex_link<int> vlink;
  float aabb_diagonal = 1.0f;
  bool colors_updated = false;

  auto build_shape_index() -> void {
    auto &store = get_mesh_data_store();
    if (store.empty())
      return;

    auto &data = store[0];
    const auto n_vertices = data.polygons.points().size();

    // Allocate shape index
    shape_index.allocate(n_vertices);
    tf::compute_shape_index(data.polygons.polygons(), shape_index);

    // Allocate vertex colors (RGB per vertex)
    vertex_colors.resize(n_vertices * 3);
    for (std::size_t i = 0; i < n_vertices; ++i) {
      vertex_colors[i * 3 + 0] = white[0];
      vertex_colors[i * 3 + 1] = white[1];
      vertex_colors[i * 3 + 2] = white[2];
    }

    // Build vertex link for neighborhood queries
    if (!data.face_membership) {
      data.face_membership.emplace();
      data.face_membership->build(data.polygons.polygons());
    }
    vlink.build(data.polygons.polygons(), *data.face_membership);

    // Compute AABB diagonal once
    auto aabb = tf::aabb_from(data.polygons.points());
    aabb_diagonal = aabb.diagonal().length();

    colors_updated = true;
  }

  auto get_vertex_colors_view() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        vertex_colors.size(), vertex_colors.data()));
  }

  auto get_histogram_bins_view() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        histogram_bins.size(), histogram_bins.data()));
  }
};

class cursor_interactor_shape_histogram final
    : public cursor_interactor_interface {
public:
  cursor_interactor_shape_histogram()
      : cursor_interactor_interface(
            std::make_unique<tf_bridge_shape_histogram>()) {}

  auto get_shape_bridge() -> tf_bridge_shape_histogram * {
    return dynamic_cast<tf_bridge_shape_histogram *>(bridge.get());
  }

  auto colors_updated() -> bool {
    auto *b = get_shape_bridge();
    if (!b)
      return false;
    bool updated = b->colors_updated;
    b->colors_updated = false;
    return updated;
  }

  auto get_vertex_colors() -> emscripten::val {
    auto *b = get_shape_bridge();
    if (!b)
      return emscripten::val::undefined();
    return b->get_vertex_colors_view();
  }

  auto get_histogram_bins() -> emscripten::val {
    auto *b = get_shape_bridge();
    if (!b)
      return emscripten::val::undefined();
    return b->get_histogram_bins_view();
  }

  auto set_radius(float r) -> void { radius_ = r; }

  auto get_aabb_diagonal() -> float {
    auto *b = get_shape_bridge();
    return b ? b->aabb_diagonal : 1.0f;
  }

  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    (void)camera_position;
    (void)camera_focal_point;

    auto *b = get_shape_bridge();
    if (!b || get_mesh_data_store().empty())
      return false;

    tf::ray<float, 3> ray{origin, direction};

    // Pick against mesh
    auto [instance_id, hit_point] = bridge->ray_hit(ray);

    if (instance_id) {
      // Find closest vertex to hit point
      auto &data = get_mesh_data_store()[0];
      auto &inst = get_instances()[*instance_id];

      // Get the cell/face that was hit
      auto form = data.polygons.polygons() | tf::tag(data.tree) | tf::tag(inst.frame);
      auto ray_result = tf::ray_cast(ray, form, tf::ray_config<float>{});

      if (ray_result) {
        auto face = data.polygons.faces()[ray_result.element];
        auto points = data.polygons.points();

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

        tf::tick();
        update_neighborhood(closest_vertex);
        add_update_time(tf::tock());
        return true;
      }
    } else {
      // Nothing picked - clear selection
      clear_selection();
    }

    return false;
  }

  // Disable dragging - we only want hover
  auto OnLeftButtonDown() -> bool override { return false; }
  auto OnLeftButtonUp() -> bool override { return false; }

private:
  float radius_ = 1.0f;
  int last_vertex_ = -1;
  tf::topology::neighborhood_applier<int> applier_;
  std::vector<int> current_indices_;
  std::vector<int> previous_indices_;
  std::vector<float> update_times_;

  auto add_update_time(float t) -> void { m_time = add_time(update_times_, t); }

  auto clear_selection() -> void {
    auto *b = get_shape_bridge();
    if (!b || previous_indices_.empty())
      return;

    auto &data = get_mesh_data_store()[0];
    const auto n_vertices = data.polygons.points().size();

    // Get colors range for indirect access
    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_vertices));

    // Restore previous colors to white
    auto prev_colors = tf::make_indirect_range(previous_indices_, colors_range);
    tf::parallel_fill(prev_colors, white);

    previous_indices_.clear();
    last_vertex_ = -1;

    // Clear histogram
    b->histogram_bins.fill(0);
    b->colors_updated = true;
  }

  auto update_neighborhood(int vertex_id) -> void {
    auto *b = get_shape_bridge();
    if (!b)
      return;

    last_vertex_ = vertex_id;

    auto &data = get_mesh_data_store()[0];
    auto points = data.polygons.points();
    const auto n_vertices = points.size();

    // Get colors range for indirect access
    auto colors_ptr = b->vertex_colors.data();
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_vertices));

    // Restore previous colors to white
    if (!previous_indices_.empty()) {
      auto prev_colors =
          tf::make_indirect_range(previous_indices_, colors_range);
      tf::parallel_fill(prev_colors, white);
    }

    // Collect new neighborhood indices
    current_indices_.clear();
    applier_(
        b->vlink, vertex_id,
        [&points](int seed, int neighbor) {
          return tf::distance2(points[seed], points[neighbor]);
        },
        radius_, [this](int idx) { current_indices_.push_back(idx); }, true);

    // Color current neighborhood
    auto neigh_colors = tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, highlight);

    // Compute histogram
    b->histogram_bins.fill(0);
    auto neigh_si = tf::make_indirect_range(current_indices_, b->shape_index);
    for (float si : neigh_si) {
      int bin = std::clamp(static_cast<int>((si + 1.0f) / bin_width), 0,
                           num_bins - 1);
      b->histogram_bins[bin]++;
    }

    // Store current as previous for next update
    previous_indices_ = current_indices_;
    b->colors_updated = true;
  }
};

inline int run_main_shape_histogram(std::string path) {
  interactor = std::make_unique<cursor_interactor_shape_histogram>();

  // Load mesh
  auto poly = tf::read_stl<int>(path);
  if (!poly.size()) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  utils::center_and_scale_p(poly);

  // Add mesh data (with topology for vertex_link)
  auto mesh_id = interactor->add_mesh_data(std::move(poly), true);
  interactor->add_instance(mesh_id);

  // Build shape index and initialize colors
  if (auto *shape_interactor =
          dynamic_cast<cursor_interactor_shape_histogram *>(interactor.get())) {
    shape_interactor->get_shape_bridge()->build_shape_index();

    // Set radius as percentage of AABB diagonal
    auto &data = shape_interactor->get_mesh_data_store()[0];
    auto aabb = tf::aabb_from(data.polygons.points());
    float diag = aabb.diagonal().length();
    shape_interactor->set_radius(diag * 0.075f);
  }

  return 0;
}
