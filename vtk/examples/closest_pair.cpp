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
#include <vtkLineSource.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTubeFilter.h>

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

  // Create two actors with random rotations
  std::vector<vtkSmartPointer<vtkOpenGLActor>> actors;
  std::vector<vtkSmartPointer<vtkMatrix4x4>> matrices;

  for (int i = 0; i < 2; ++i) {
    // Create transform with random rotation at centroid, placed side by side
    tf::point<float, 3> position{i * spacing, 0.f, 0.f};
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

  // Create line source and tube filter for visualizing closest pair
  vtkNew<vtkLineSource> line_source;
  line_source->SetPoint1(0, 0, 0);
  line_source->SetPoint2(0, 0, 0);

  vtkNew<vtkTubeFilter> tube_filter;
  tube_filter->SetInputConnection(line_source->GetOutputPort());
  tube_filter->SetRadius(spacing * 0.002f);
  tube_filter->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> line_mapper;
  line_mapper->SetInputConnection(tube_filter->GetOutputPort());

  vtkNew<vtkOpenGLActor> line_actor;
  line_actor->SetMapper(line_mapper);
  line_actor->GetProperty()->SetColor(0.3, 1.0,
                                      0.3); // Will be updated dynamically

  // Set up renderer
  vtkNew<vtkOpenGLRenderer> renderer;
  for (auto &actor : actors) {
    renderer->AddActor(actor);
  }
  renderer->AddActor(line_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  // Distance-based color interpolation (green = close, red = far)
  float max_distance = spacing * 0.75;

  // Closest pair update - computes and visualizes closest points between meshes
  auto update_closest_pair =
      [poly, &matrices, &line_source, &line_actor, &window,
       max_distance](vtkActor *, std::vector<vtkActor *> &) {
        auto result =
            tf::vtk::neighbor_search(std::make_pair(poly, matrices[0].Get()),
                                     std::make_pair(poly, matrices[1].Get()));

        line_source->SetPoint1(result.info.first[0], result.info.first[1],
                               result.info.first[2]);
        line_source->SetPoint2(result.info.second[0], result.info.second[1],
                               result.info.second[2]);
        line_source->Modified();

        // Interpolate color: green (close) -> red (far)
        float t = std::min(std::sqrt(result.info.metric) / max_distance, 1.0f);
        line_actor->GetProperty()->SetColor(t, 1.0 - t, 0.3);

        window->Render();
      };

  // Set up drag interactor
  vtkNew<tf::vtk::examples::drag_interactor> style;
  for (auto &actor : actors) {
    style->add_actor(actor, renderer);
  }
  style->set_callback(update_closest_pair);
  interactor->SetInteractorStyle(style);

  // Initial computation
  std::vector<vtkActor *> dummy;
  update_closest_pair(nullptr, dummy);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
