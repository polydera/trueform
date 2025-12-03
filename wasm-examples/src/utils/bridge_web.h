#pragma once
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"
#include <emscripten/bind.h>
#include <emscripten/val.h>

// Shared mesh data - one per unique mesh
struct mesh_data {
  tf::polygons_buffer<int, float, 3, 3> polygons;
  tf::aabb_tree<int, float, 3> tree;
  std::optional<tf::face_membership<int>> face_membership;
  std::optional<tf::manifold_edge_link<int, 3>> manifold_edge_link;

  auto get_points() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        polygons.points_buffer().data_buffer().size(),
        polygons.points_buffer().data_buffer().begin()));
  }

  auto get_faces() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        polygons.faces_buffer().data_buffer().size(),
        polygons.faces_buffer().data_buffer().begin()));
  }
};

// Per-instance data - one per actor/entity
struct instance {
  std::size_t mesh_data_id = 0;
  tf::frame<double, 3> frame;
  std::array<double, 16> matrix{};
  std::array<double, 3> color{1.0, 1.0, 1.0};
  bool matrix_updated = true;

  instance() {
    matrix.fill(0.0);
    for (int i = 0; i < static_cast<int>(matrix.size()); i += 5) {
      matrix[i] = 1.0;
    }
  }

  auto set_color(double r, double g, double b) -> void {
    color = {r, g, b};
  }

  auto get_matrix() -> emscripten::val {
    matrix_updated = false;
    return emscripten::val(
        emscripten::typed_memory_view(matrix.size(), matrix.data()));
  }

  auto update_frame() -> void {
    frame.fill(matrix.data());
    matrix_updated = true;
  }
};

// Result mesh for operations like boolean, isobands
struct result_mesh {
  tf::polygons_buffer<int, float, 3, 3> polygons;
  tf::curves_buffer<int, float, 3> curves;
  bool updated = false;

  auto set_polygons(tf::polygons_buffer<int, float, 3, 3> polys) -> void {
    polygons = std::move(polys);
    updated = true;
  }

  auto set_curves(tf::curves_buffer<int, float, 3> c) -> void {
    curves = std::move(c);
    updated = true;
  }

  auto get_points() -> emscripten::val {
    updated = false;
    return emscripten::val(emscripten::typed_memory_view(
        polygons.points_buffer().data_buffer().size(),
        polygons.points_buffer().data_buffer().begin()));
  }

  auto get_faces() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        polygons.faces_buffer().data_buffer().size(),
        polygons.faces_buffer().data_buffer().begin()));
  }

  auto get_curve_points() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        curves.points_buffer().data_buffer().size(),
        curves.points_buffer().data_buffer().begin()));
  }

  auto get_curve_ids() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        curves.paths_buffer().data_buffer().size(),
        curves.paths_buffer().data_buffer().begin()));
  }

  auto get_curve_offsets() -> emscripten::val {
    return emscripten::val(emscripten::typed_memory_view(
        curves.paths_buffer().offsets_buffer().size(),
        curves.paths_buffer().offsets_buffer().begin()));
  }
};
