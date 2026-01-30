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
#include <array>
#include <iostream>
#include <set>
#include <trueform/trueform.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/filters.hpp>
#include <trueform/vtk/functions.hpp>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

namespace {

// Colors
constexpr std::array<unsigned char, 3> base_color = {220, 220, 225};
constexpr std::array<unsigned char, 3> brush_active = {255, 140, 100};
constexpr std::array<unsigned char, 3> brush_preview = {180, 200, 255};

/// Custom interactor for interactive Laplacian smoothing
class laplacian_smoothing_interactor
    : public vtkInteractorStyleTrackballCamera {
public:
  static laplacian_smoothing_interactor *New();
  vtkTypeMacro(laplacian_smoothing_interactor,
               vtkInteractorStyleTrackballCamera);

  void set_data(tf::vtk::polydata *poly, vtkOpenGLRenderer *renderer,
                vtkUnsignedCharArray *colors, vtkOpenGLActor *mesh_actor,
                float radius, float lambda) {
    poly_ = poly;
    renderer_ = renderer;
    colors_ = colors;
    mesh_actor_ = mesh_actor;
    radius_ = radius;
    lambda_ = lambda;
  }

  void OnLeftButtonDown() override {
    int *pos = this->Interactor->GetEventPosition();
    auto ray = tf::vtk::make_world_ray(renderer_, pos[0], pos[1]);

    std::vector<vtkActor *> actors = {mesh_actor_};
    auto result = tf::vtk::pick(ray, actors);

    if (result) {
      // Hit mesh - enter painting mode
      painting_ = true;
      update_brush(result);
      this->Interactor->Render();
    } else {
      // Miss mesh - let camera handle it
      vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
    }
  }

  void OnLeftButtonUp() override {
    if (painting_) {
      painting_ = false;
      // Recolor active brush to preview color instead of clearing
      if (!current_indices_.empty()) {
        const auto n_points = poly_->points().size();
        auto colors_ptr = colors_->GetPointer(0);
        auto colors_range =
            tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));
        auto neigh_colors =
            tf::make_indirect_range(current_indices_, colors_range);
        tf::parallel_fill(neigh_colors, brush_preview);
        // Transfer to preview indices
        preview_indices_ = std::move(current_indices_);
        current_indices_.clear();
        colors_->Modified();
        poly_->Modified();
        this->Interactor->Render();
      }
      poly_->reset_poly_tree();
      tf::tick();
      poly_->poly_tree();
      tf::tock("build");
    }
    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
  }

  void OnMouseMove() override {
    if (!painting_) {
      // Not painting - show brush preview
      show_preview();
      // Let camera handle rotation/pan if button is held
      vtkInteractorStyleTrackballCamera::OnMouseMove();
      return;
    }

    // Painting mode - smooth and highlight
    int *pos = this->Interactor->GetEventPosition();
    auto ray = tf::vtk::make_world_ray(renderer_, pos[0], pos[1]);

    std::vector<vtkActor *> actors = {mesh_actor_};
    auto result = tf::vtk::pick(ray, actors);

    if (result) {
      update_brush(result);
      this->Interactor->Render();
    } else {
      // Moved off mesh while painting - clear highlight
      clear_highlight();
    }
  }

  void OnMouseWheelForward() override {
    if (this->Interactor->GetControlKey()) {
      // Ctrl + scroll up - increase radius
      radius_ *= 1.1f;
      show_preview();
      std::cout << "Brush radius: " << radius_ << std::endl;
    } else {
      vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
    }
  }

  void OnMouseWheelBackward() override {
    if (this->Interactor->GetControlKey()) {
      // Ctrl + scroll down - decrease radius
      radius_ *= 0.9f;
      show_preview();
      std::cout << "Brush radius: " << radius_ << std::endl;
    } else {
      vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
    }
  }

private:
  void clear_highlight() {
    if (!colors_ || !poly_ || current_indices_.empty())
      return;

    const auto n_points = poly_->points().size();
    auto colors_ptr = colors_->GetPointer(0);
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    auto neigh_colors = tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, base_color);

    current_indices_.clear();
    colors_->Modified();
    poly_->Modified();
    this->Interactor->Render();
  }

  void clear_preview() {
    if (!colors_ || !poly_ || preview_indices_.empty())
      return;

    const auto n_points = poly_->points().size();
    auto colors_ptr = colors_->GetPointer(0);
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    auto preview_colors =
        tf::make_indirect_range(preview_indices_, colors_range);
    tf::parallel_fill(preview_colors, base_color);

    preview_indices_.clear();
    colors_->Modified();
    poly_->Modified();
    this->Interactor->Render();
  }

  void show_preview() {
    if (!poly_)
      return;

    int *pos = this->Interactor->GetEventPosition();
    auto ray = tf::vtk::make_world_ray(renderer_, pos[0], pos[1]);

    std::vector<vtkActor *> actors = {mesh_actor_};
    auto result = tf::vtk::pick(ray, actors);

    if (!result) {
      clear_preview();
      return;
    }

    auto faces = poly_->polys();
    auto points = poly_->points();
    auto face = faces[result.cell_id];
    const auto n_points = points.size();

    vtkIdType closest_vertex = face[0];
    float min_dist2 = std::numeric_limits<float>::max();
    for (auto vid : face) {
      float d2 = tf::distance2(points[vid], result.position);
      if (d2 < min_dist2) {
        min_dist2 = d2;
        closest_vertex = vid;
      }
    }

    // Clear previous preview
    auto colors_ptr = colors_->GetPointer(0);
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    if (!preview_indices_.empty()) {
      auto prev_colors =
          tf::make_indirect_range(preview_indices_, colors_range);
      tf::parallel_fill(prev_colors, base_color);
    }

    // Collect new preview neighborhood
    preview_indices_.clear();
    const auto &vlink = poly_->vertex_link();
    applier_(
        vlink, closest_vertex,
        [&points](vtkIdType seed, vtkIdType neighbor) {
          return tf::distance2(points[seed], points[neighbor]);
        },
        radius_,
        [this](vtkIdType idx) { preview_indices_.push_back(idx); }, true);

    // Highlight preview
    auto preview_colors =
        tf::make_indirect_range(preview_indices_, colors_range);
    tf::parallel_fill(preview_colors, brush_preview);

    colors_->Modified();
    poly_->Modified();
    this->Interactor->Render();
  }

  void update_brush(const tf::vtk::pick_result &result) {
    if (!poly_)
      return;

    auto faces = poly_->polys();
    auto points = poly_->points();
    auto face = faces[result.cell_id];
    const auto n_points = points.size();

    vtkIdType closest_vertex = face[0];
    float min_dist2 = std::numeric_limits<float>::max();
    for (auto vid : face) {
      float d2 = tf::distance2(points[vid], result.position);
      if (d2 < min_dist2) {
        min_dist2 = d2;
        closest_vertex = vid;
      }
    }

    const auto &vlink = poly_->vertex_link();

    // Restore previous highlight
    auto colors_ptr = colors_->GetPointer(0);
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points * 3));

    if (!current_indices_.empty()) {
      auto prev_colors =
          tf::make_indirect_range(current_indices_, colors_range);
      tf::parallel_fill(prev_colors, base_color);
    }

    // Collect neighborhood
    current_indices_.clear();
    polygon_set_.clear();
    polygon_ids_.clear();
    const auto &fm = poly_->face_membership();
    applier_(
        vlink, closest_vertex,
        [&points](vtkIdType seed, vtkIdType neighbor) {
          return tf::distance2(points[seed], points[neighbor]);
        },
        radius_,
        [this, &fm](vtkIdType idx) {
          current_indices_.push_back(idx);
          for (auto poly_id : fm[idx])
            if (polygon_set_.insert(poly_id).second)
              polygon_ids_.push_back(poly_id);
        },
        true);

    // Highlight neighborhood with active brush color
    auto neigh_colors =
        tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, brush_active);

    // Apply Laplacian smoothing
    auto neigh_points = tf::make_indirect_range(current_indices_, points);
    auto neigh_neighbors = tf::make_indirect_range(
        current_indices_, tf::make_block_indirect_range(vlink, points));

    tf::parallel_for_each(
        tf::zip(neigh_points, neigh_neighbors),
        [&](auto tup) {
          auto [pt, neighbors] = tup;
          pt = tf::laplacian_smoothed(pt, tf::make_points(neighbors), lambda_);
        },
        tf::checked);

    poly_->GetPoints()->Modified();
    colors_->Modified();
    poly_->Modified();

    tf::tick();
    poly_->update_poly_tree(polygon_ids_);
    tf::tock("update");
  }

  tf::vtk::polydata *poly_ = nullptr;
  vtkOpenGLRenderer *renderer_ = nullptr;
  vtkUnsignedCharArray *colors_ = nullptr;
  vtkOpenGLActor *mesh_actor_ = nullptr;

  float radius_ = 1.0f;
  float lambda_ = 0.5f;
  bool painting_ = false;

  tf::topology::neighborhood_applier<vtkIdType> applier_;
  std::vector<vtkIdType> current_indices_;
  std::vector<vtkIdType> preview_indices_;
  std::vector<vtkIdType> polygon_ids_;
  tf::hash_set<vtkIdType> polygon_set_;
};

vtkStandardNewMacro(laplacian_smoothing_interactor);

} // namespace

int main(int argc, char *argv[]) {
  auto poly =
      tf::vtk::read_stl(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");

  auto points = poly->points();
  const auto n_vertices = points.size();
  std::cout << "Vertices: " << n_vertices << std::endl;

  auto aabb = tf::vtk::aabb_from(poly);
  float diag = aabb.diagonal().length();
  float initial_radius = diag * 0.05f;
  std::cout << "Brush radius: " << initial_radius << std::endl;

  vtkNew<vtkUnsignedCharArray> colors;
  colors->SetNumberOfComponents(3);
  colors->SetNumberOfTuples(n_vertices);
  colors->SetName("Colors");

  auto colors_ptr =
      reinterpret_cast<std::array<unsigned char, 3> *>(colors->GetPointer(0));
  auto colors_range = tf::make_range(colors_ptr, n_vertices);
  tf::parallel_fill(colors_range, base_color);

  poly->GetPointData()->SetScalars(colors);

  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputData(poly);
  mapper->SetScalarModeToUsePointData();
  mapper->SetColorModeToDirectScalars();

  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mapper);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(mesh_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);
  window->SetWindowName("Interactive Laplacian Smoothing");

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  constexpr float lambda = 0.3f;
  vtkNew<laplacian_smoothing_interactor> style;
  style->set_data(poly, renderer, colors, mesh_actor, initial_radius, lambda);
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();

  std::cout << "Controls:" << std::endl;
  std::cout << "  Left drag on mesh: Smooth" << std::endl;
  std::cout << "  Left drag off mesh: Rotate camera" << std::endl;
  std::cout << "  Right drag: Zoom" << std::endl;
  std::cout << "  Middle drag: Pan" << std::endl;
  std::cout << "  Ctrl + scroll: Adjust brush radius" << std::endl;

  interactor->Start();

  return 0;
}
