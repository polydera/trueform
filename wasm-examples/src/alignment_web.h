#pragma once
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "main.h"
#include "trueform/trueform.hpp"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"

// Colors
namespace alignment_colors {
constexpr std::array<double, 3> TARGET = {0.0, 0.835, 0.745};  // Bright teal (with alpha)
} // namespace alignment_colors

class alignment_bridge : public tf_bridge_interface {
public:
  // Target is loaded first (instance 0), source is loaded second (instance 1)
  static constexpr std::size_t TARGET_ID = 0;
  static constexpr std::size_t SOURCE_ID = 1;

  // Point cloud and normals for target (for ICP)
  tf::points_buffer<float, 3> target_points;
  tf::unit_vectors_buffer<float, 3> target_normals;
  tf::aabb_tree<int, float, 3> target_point_tree;

  auto prepare_target_for_alignment() -> void {
    if (instances.size() < 2) {
      return;
    }

    auto &target_data = mesh_data_store[instances[TARGET_ID].mesh_data_id];
    auto points = target_data.polygons.points();

    // Copy points for alignment
    target_points.allocate(points.size());
    tf::parallel_copy(points, target_points.points());

    // Compute point normals
    auto polygons = target_data.polygons.polygons();
    auto normals = tf::compute_point_normals(polygons);
    target_normals.allocate(normals.size());
    tf::parallel_copy(normals, target_normals.unit_vectors());

    // Build spatial tree on target points
    target_point_tree.build(target_points.points(), tf::config_tree(4, 4));
  }
};

class cursor_interactor_alignment : public cursor_interactor_interface {
public:
  cursor_interactor_alignment()
      : cursor_interactor_interface(std::make_unique<alignment_bridge>()) {}

  // Get the bridge as alignment_bridge
  auto get_alignment_bridge() -> alignment_bridge * {
    return static_cast<alignment_bridge *>(bridge.get());
  }

  // Run OBB + ICP alignment
  auto run_alignment() -> float {
    auto *ab = get_alignment_bridge();
    if (!ab || ab->get_instances().size() < 2) {
      return -1.0f;
    }

    tf::tick();

    auto &source_inst = ab->get_instance(alignment_bridge::SOURCE_ID);
    auto &source_data = ab->get_mesh_data(source_inst.mesh_data_id);

    // Get source points
    auto source_points = source_data.polygons.points();

    // Get current source frame
    auto T_source = source_inst.frame.transformation();

    // Create source cloud with current transformation
    auto source_cloud = source_points | tf::tag(T_source);

    // Create target cloud with tree, normals, and frame
    auto &target_inst = ab->get_instance(alignment_bridge::TARGET_ID);
    auto target_cloud =
        ab->target_points.points() | tf::tag(ab->target_point_tree) |
        tf::tag_normals(ab->target_normals.unit_vectors()) |
        tf::tag(target_inst.frame);

    // Stage 1: OBB coarse alignment
    auto T_obb_delta = tf::fit_obb_alignment(source_cloud, target_cloud);
    auto T_after_obb = tf::transformed(T_source, T_obb_delta);

    // Stage 2: ICP refinement (point-to-plane with normals)
    auto source_after_obb = source_points | tf::tag(T_after_obb);

    tf::icp_config icp_cfg;
    icp_cfg.max_iterations = 50;
    icp_cfg.n_samples = 1000;
    icp_cfg.k = 1;

    auto T_icp_delta =
        tf::fit_icp_alignment(source_after_obb, target_cloud, icp_cfg);
    auto T_final = tf::transformed(T_after_obb, T_icp_delta);

    // Update source instance matrix
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        source_inst.matrix[i * 4 + j] = T_final(i, j);
      }
    }
    source_inst.matrix_updated = true;
    ab->update_frame(alignment_bridge::SOURCE_ID);

    m_aligned = true;
    m_alignment_time = tf::tock();

    return m_alignment_time;
  }

  // Set source instance matrix from JavaScript (when gizmo changes)
  auto set_source_matrix(const std::array<double, 16> &matrix) -> void {
    auto *ab = get_alignment_bridge();
    if (!ab || ab->get_instances().empty()) {
      return;
    }

    auto &source_inst = ab->get_instance(alignment_bridge::SOURCE_ID);
    source_inst.matrix = matrix;
    source_inst.matrix_updated = true;
    ab->update_frame(alignment_bridge::SOURCE_ID);

    m_aligned = false;
  }

  // Get source instance matrix (for syncing gizmo)
  auto get_source_matrix() -> std::array<double, 16> {
    auto *ab = get_alignment_bridge();
    if (!ab || ab->get_instances().empty()) {
      return {};
    }
    return ab->get_instance(alignment_bridge::SOURCE_ID).matrix;
  }

  auto is_aligned() const -> bool { return m_aligned; }

  auto get_alignment_time() const -> float { return m_alignment_time; }

  auto get_aabb_diagonal() const -> float { return m_aabb_diagonal; }

  auto set_aabb_diagonal(float diag) -> void { m_aabb_diagonal = diag; }

private:
  bool m_aligned = false;
  float m_alignment_time = 0.0f;
  float m_aabb_diagonal = 1.0f;
};

int run_main_alignment(std::vector<std::string> &paths) {
  if (paths.empty()) {
    throw std::runtime_error("At least one STL path is required.");
  }

  interactor = std::make_unique<cursor_interactor_alignment>();
  auto *align_interactor =
      dynamic_cast<cursor_interactor_alignment *>(interactor.get());

  std::size_t total_polygons = 0;

  // Load target (always 50k dragon - second path or first if only one)
  // Apply Taubin smoothing to target
  {
    const auto &target_path = paths.size() > 1 ? paths[1] : paths[0];
    auto target_poly = tf::read_stl<int>(target_path);
    if (!target_poly.size()) {
      throw std::runtime_error("Failed to read target file: " + target_path);
    }
    utils::center_and_scale_p(target_poly);

    // Build topology for Taubin smoothing
    tf::face_membership<int> fm;
    fm.build(target_poly.polygons());
    tf::vertex_link<int> vlink;
    vlink.build(target_poly.polygons(), fm);

    // Apply Taubin smoothing (50 iterations, lambda=0.9)
    auto tagged_points = target_poly.points() | tf::tag(vlink);
    auto smoothed_points = tf::taubin_smoothed(tagged_points, 50, 0.9f);
    tf::parallel_copy(smoothed_points.points(), target_poly.points());

    total_polygons += target_poly.size();

    auto target_mesh_id =
        interactor->add_mesh_data(std::move(target_poly), false);
    auto target_instance_id = interactor->add_instance(
        target_mesh_id, alignment_colors::TARGET[0],
        alignment_colors::TARGET[1], alignment_colors::TARGET[2]);

    // Keep target at identity, make it semi-transparent and non-selectable
    auto &target_inst = interactor->get_instances()[target_instance_id];
    // matrix is already identity from constructor
    target_inst.set_opacity(0.5);
    target_inst.selectable = false;
    target_inst.update_frame();
  }

  // Load source (user-selected mesh)
  {
    const auto &source_path = paths[0];
    auto source_poly = tf::read_stl<int>(source_path);
    if (!source_poly.size()) {
      throw std::runtime_error("Failed to read source file: " + source_path);
    }
    utils::center_and_scale_p(source_poly);
    total_polygons += source_poly.size();

    // Compute AABB diagonal for UI
    auto aabb = tf::aabb_from(source_poly.points());
    float diag = tf::distance(aabb.min, aabb.max);
    align_interactor->set_aabb_diagonal(diag);

    auto source_mesh_id =
        interactor->add_mesh_data(std::move(source_poly), false);
    // Source uses default white color
    auto source_instance_id = interactor->add_instance(source_mesh_id);

    // Source starts at origin, JS will position based on screen dimensions
    auto &source_inst = interactor->get_instances()[source_instance_id];
    source_inst.update_frame();
  }

  // Prepare target point cloud and normals for ICP
  if (auto *ab = align_interactor->get_alignment_bridge()) {
    ab->prepare_target_for_alignment();
  }

  interactor->total_polygons = total_polygons;
  return 0;
}
