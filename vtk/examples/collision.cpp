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
#include <trueform/core.hpp>
#include <trueform/random.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <trueform/vtk/functions.hpp>
#include <util/drag_interactor.hpp>
#include <vtkCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

int main() {
  // Load mesh
  vtkNew<tf::vtk::stl_reader> reader;
  reader->set_file_name(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  reader->Update();

  // SafeDownCast to get polydata with cached structures
  auto *poly = tf::vtk::polydata::SafeDownCast(reader->GetOutput());

  // Compute AABB and centroid to determine spacing and rotation pivot
  auto aabb = tf::vtk::aabb_from(reader->GetOutput());
  auto extent = aabb.max - aabb.min;
  float spacing = std::max({extent[0], extent[1], extent[2]}) * 1.2f;

  auto points = tf::vtk::make_points(reader->GetOutput());
  auto centroid = tf::centroid(points);

  // Create actors for 5x5 grid
  constexpr int grid_size = 5;
  std::vector<vtkSmartPointer<vtkOpenGLActor>> actors;
  std::vector<vtkSmartPointer<vtkMatrix4x4>> matrices;

  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      // Create transform with random rotation at centroid, placed at grid position
      tf::point<float, 3> position{i * spacing, j * spacing, 0.f};
      auto transform = tf::random_transformation_at(centroid, position);
      auto matrix = tf::vtk::make_vtk_matrix(transform);

      // Create mapper and actor
      vtkNew<vtkOpenGLPolyDataMapper> mapper;
      mapper->SetInputConnection(reader->GetOutputPort());

      auto actor = vtkSmartPointer<vtkOpenGLActor>::New();
      actor->SetMapper(mapper);
      actor->SetUserMatrix(matrix);
      actor->GetProperty()->SetColor(0.8, 0.8, 0.8);

      actors.push_back(actor);
      matrices.push_back(matrix);
    }
  }

  // Set up renderer
  vtkNew<vtkOpenGLRenderer> renderer;
  for (auto &actor : actors) {
    renderer->AddActor(actor);
  }
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  // Set up drag interactor with collision callback
  vtkNew<tf::vtk::examples::drag_interactor> style;

  // Add all actors to drag interactor
  for (auto &actor : actors) {
    style->add_actor(actor, renderer);
  }

  // Collision callback
  style->set_callback([poly, &matrices](vtkActor *selected,
                                        std::vector<vtkActor *> &all_actors) {
    // Reset all actors to default color
    for (auto *actor : all_actors) {
      actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
    }

    // Find index of selected actor
    std::size_t selected_idx = 0;
    for (auto &&[idx, actor] : tf::enumerate(all_actors)) {
      if (actor == selected) {
        selected_idx = idx;
        break;
      }
    }

    auto *selected_matrix = matrices[selected_idx].Get();

    // Check collisions using tf::zip
    for (auto &&[matrix, actor] : tf::zip(matrices, all_actors)) {
      // Skip self
      if (actor == selected) {
        continue;
      }

      // Check intersection
      bool collision =
          tf::vtk::intersects(std::make_pair(poly, selected_matrix),
                              std::make_pair(poly, matrix.Get()));

      if (collision) {
        actor->GetProperty()->SetColor(0.9, 0.7, 0.7);
      }
    }
  });

  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
