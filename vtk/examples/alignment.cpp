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
#include <iostream>
#include <trueform/trueform.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/functions.hpp>
#include <util/drag_interactor.hpp>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

// Colors (teal-based scheme)
constexpr double SOURCE_COLOR[3] = {0.0, 0.659, 0.604};
constexpr double TARGET_COLOR[3] = {0.2, 0.35, 0.33}; // Dim teal
constexpr double ALIGNED_COLOR[3] = {0.0, 0.835, 0.745};

class alignment_interactor : public tf::vtk::examples::drag_interactor {
public:
  static auto New() -> alignment_interactor *;
  vtkTypeMacro(alignment_interactor, tf::vtk::examples::drag_interactor);

  auto initialize(tf::vtk::polydata *source_poly,
                  tf::vtk::polydata *target_poly,
                  tf::transformation<float, 3> *T_source,
                  tf::transformation<float, 3> *T_target,
                  vtkOpenGLActor *source_actor) -> void {
    _source_poly = source_poly;
    _target_poly = target_poly;
    _T_source = T_source;
    _T_target = T_target;
    _source_actor = source_actor;
  }

  auto OnKeyPress() -> void override {
    auto key = this->Interactor->GetKeySym();

    if (std::string(key) == "a" || std::string(key) == "A") {
      run_alignment();
      this->Interactor->Render();
    } else {
      tf::vtk::examples::drag_interactor::OnKeyPress();
    }
  }

  auto OnRightButtonDown() -> void override {
    _rotating = true;
    int x, y;
    this->Interactor->GetEventPosition(x, y);
    _last_x = x;
    _last_y = y;

    // Sync T_source from actor (in case user just dragged it)
    sync_transform_from_actor();

    // Rotation center = mesh centroid in world space
    auto centroid = tf::centroid(_source_poly->points());
    _rotation_center = tf::transformed(centroid, *_T_source);

    this->Interactor->GetRenderWindow()->HideCursor();
  }

  auto OnRightButtonUp() -> void override {
    if (_rotating) {
      _rotating = false;
      this->Interactor->GetRenderWindow()->ShowCursor();
    }
  }

  auto OnMouseMove() -> void override {
    if (_rotating) {
      int x, y;
      this->Interactor->GetEventPosition(x, y);
      int dx = x - _last_x;
      int dy = y - _last_y;
      _last_x = x;
      _last_y = y;

      // Convert mouse movement to rotation angles (degrees)
      float angle_x = dy * 0.5f;
      float angle_y = dx * 0.5f;

      // Rotate around world X and Y axes, centered at mesh centroid
      auto Rx =
          tf::make_rotation(tf::deg(angle_x), tf::axis<0>, _rotation_center);
      auto Ry =
          tf::make_rotation(tf::deg(angle_y), tf::axis<1>, _rotation_center);

      // Apply rotations to current transform
      *_T_source = tf::transformed(tf::transformed(*_T_source, Rx), Ry);
      _source_actor->SetUserMatrix(tf::vtk::make_vtk_matrix(*_T_source));

      this->Interactor->Render();
    } else {
      tf::vtk::examples::drag_interactor::OnMouseMove();
    }
  }

protected:
  alignment_interactor() = default;
  ~alignment_interactor() override = default;

private:
  auto sync_transform_from_actor() -> void {
    // Sync T_source from the actor's user matrix (after dragging)
    auto *matrix = _source_actor->GetUserMatrix();
    if (matrix) {
      *_T_source = tf::vtk::make_frame(matrix).transformation();
    }
  }

  auto run_alignment() -> void {
    // Sync transform from actor (in case user dragged it)
    sync_transform_from_actor();

    auto source_points = _source_poly->points();
    auto target_points = _target_poly->points();
    auto target_normals = _target_poly->point_normals();
    const auto &target_tree = _target_poly->point_tree();

    // Create point clouds with transformations
    // Target has normals for point-to-plane ICP
    auto source_cloud = source_points | tf::tag(*_T_source);
    auto target_cloud = target_points | tf::tag(target_tree) |
                        tf::tag_normals(target_normals) | tf::tag(*_T_target);

    // OBB alignment
    auto T_obb_delta = tf::fit_obb_alignment(source_cloud, target_cloud);
    auto T_after_obb = tf::transformed(*_T_source, T_obb_delta);

    // ICP alignment (point-to-plane with target normals)
    auto source_after_obb = source_points | tf::tag(T_after_obb);

    tf::icp_config icp_cfg;
    icp_cfg.max_iterations = 50;
    icp_cfg.n_samples = 1000;
    icp_cfg.k = 1;

    auto T_icp_delta =
        tf::fit_icp_alignment(source_after_obb, target_cloud, icp_cfg);
    auto T_final = tf::transformed(T_after_obb, T_icp_delta);

    // Update source transform and visualization
    *_T_source = T_final;
    _source_actor->SetUserMatrix(tf::vtk::make_vtk_matrix(*_T_source));
    _source_actor->GetProperty()->SetColor(ALIGNED_COLOR[0], ALIGNED_COLOR[1],
                                           ALIGNED_COLOR[2]);
  }

  tf::vtk::polydata *_source_poly = nullptr;
  tf::vtk::polydata *_target_poly = nullptr;
  tf::transformation<float, 3> *_T_source = nullptr;
  tf::transformation<float, 3> *_T_target = nullptr;
  vtkOpenGLActor *_source_actor = nullptr;

  // Rotation state
  bool _rotating = false;
  int _last_x = 0;
  int _last_y = 0;
  tf::point<float, 3> _rotation_center;
};

vtkStandardNewMacro(alignment_interactor);

int main() {
  std::cout << "=== Alignment Example ===" << std::endl;
  std::cout << "Drag source mesh (teal) to move it" << std::endl;
  std::cout << "Press 'A' to align source to target" << std::endl;
  std::cout << std::endl;

  // Load mesh
  auto source_poly =
      tf::vtk::read_stl(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  auto target_poly =
      tf::vtk::read_stl(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-50k.stl");
  std::cout << "Loaded mesh with " << source_poly->points().size()
            << " vertices" << std::endl;

  // Create smoothed version for target using Taubin smoothing
  std::cout << "Smoothing mesh (50 Taubin iterations)..." << std::endl;
  auto smoothed_points = tf::vtk::taubin_smoothed(target_poly, 50, 0.9f);
  target_poly->SetPoints(smoothed_points);

  // Compute normals for point-to-plane ICP
  tf::vtk::compute_point_normals(target_poly);
  std::cout << "Done." << std::endl;

  // Create transformations using factory functions
  auto centroid = tf::centroid(source_poly->points());
  auto aabb = tf::aabb_from(source_poly->points());
  float diag = tf::distance(aabb.min, aabb.max);

  // Source: rotation around Z + translation
  auto T_source =
      tf::transformed(tf::make_rotation(tf::deg(45.f), tf::axis<2>, centroid),
                      tf::make_transformation_from_translation(
                          tf::make_vector(diag * 0.3f, 0.0f, 0.0f)));

  auto T_target = tf::make_identity_transformation<float, 3>();

  std::cout << "Mesh diagonal: " << diag << std::endl;

  // VTK visualization - Source mesh (draggable)
  vtkNew<vtkOpenGLPolyDataMapper> source_mapper;
  source_mapper->SetInputData(source_poly);
  vtkNew<vtkOpenGLActor> source_actor;
  source_actor->SetMapper(source_mapper);
  source_actor->GetProperty()->SetColor(SOURCE_COLOR[0], SOURCE_COLOR[1],
                                        SOURCE_COLOR[2]);
  source_actor->SetUserMatrix(tf::vtk::make_vtk_matrix(T_source));

  // VTK visualization - Target mesh (static)
  vtkNew<vtkOpenGLPolyDataMapper> target_mapper;
  target_mapper->SetInputData(target_poly);
  vtkNew<vtkOpenGLActor> target_actor;
  target_actor->SetMapper(target_mapper);
  target_actor->GetProperty()->SetColor(TARGET_COLOR[0], TARGET_COLOR[1],
                                        TARGET_COLOR[2]);
  target_actor->GetProperty()->SetOpacity(0.5);
  target_actor->SetUserMatrix(tf::vtk::make_vtk_matrix(T_target));

  // Renderer
  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(source_actor);
  renderer->AddActor(target_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<alignment_interactor> style;
  style->initialize(source_poly, target_poly, &T_source, &T_target,
                    source_actor);
  style->add_actor(source_actor, renderer); // Only source is draggable
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
