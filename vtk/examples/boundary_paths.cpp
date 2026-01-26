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
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/filters/boundary_paths.hpp>
#include <trueform/vtk/filters/isobands.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTubeFilter.h>

int main() {
  // Load dragon mesh
  vtkNew<tf::vtk::stl_reader> reader;
  reader->set_file_name(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  reader->Update();

  auto *poly = reader->GetOutput();

  // Create scalar field based on Z coordinate (height)
  auto points = tf::vtk::make_points(poly);
  auto center = tf::centroid(points);

  vtkNew<vtkFloatArray> scalars;
  scalars->SetName("height");
  scalars->SetNumberOfTuples(poly->GetNumberOfPoints());

  auto scalars_range = tf::vtk::make_range(scalars.Get());
  // Use Z coordinate as scalar
  tf::parallel_transform(points, scalars_range,
                         [](const auto &p) { return p[2]; });

  float min_z = *std::min_element(scalars_range.begin(), scalars_range.end());
  float max_z = *std::max_element(scalars_range.begin(), scalars_range.end());
  float mid_z = (min_z + max_z) / 2.0f;

  poly->GetPointData()->SetScalars(scalars);

  // Use isobands to extract upper half of the mesh
  vtkNew<tf::vtk::isobands> bands;
  bands->SetInputConnection(reader->GetOutputPort());
  bands->set_cut_values({mid_z, max_z + 1.0f}); // from middle to above max
  bands->set_selected_bands({0});               // select the upper band

  // Extract boundary paths from the cut mesh
  vtkNew<tf::vtk::boundary_paths> boundary;
  boundary->SetInputConnection(bands->GetOutputPort(0));

  // Visualization - cut mesh
  vtkNew<vtkOpenGLPolyDataMapper> mesh_mapper;
  mesh_mapper->SetInputConnection(bands->GetOutputPort(0));
  mesh_mapper->ScalarVisibilityOff();
  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mesh_mapper);
  mesh_actor->GetProperty()->SetColor(0.8, 0.8, 0.85);

  // Visualization - boundary paths as tubes
  vtkNew<vtkTubeFilter> tube;
  tube->SetInputConnection(boundary->GetOutputPort());
  tube->SetRadius(0.0005);
  tube->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> boundary_mapper;
  boundary_mapper->SetInputConnection(tube->GetOutputPort());
  vtkNew<vtkOpenGLActor> boundary_actor;
  boundary_actor->SetMapper(boundary_mapper);
  boundary_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(mesh_actor);
  renderer->AddActor(boundary_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
