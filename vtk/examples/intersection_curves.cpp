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
#include <trueform/vtk/filters/adapter.hpp>
#include <trueform/vtk/filters/intersection_curves.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <util/drag_interactor.hpp>
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

  // Create matrices for both meshes
  auto points = tf::vtk::make_points(reader->GetOutput());
  auto center = tf::centroid(points);
  auto rotation = tf::make_rotation(tf::deg<double>{90.0}, tf::axis<2>, center);

  vtkNew<vtkMatrix4x4> matrix0;
  auto matrix1 = tf::vtk::make_vtk_matrix(rotation);

  // Create adapters
  vtkNew<tf::vtk::adapter> adapter0;
  adapter0->SetInputConnection(reader->GetOutputPort());

  vtkNew<tf::vtk::adapter> adapter1;
  adapter1->SetInputConnection(reader->GetOutputPort());

  // Compute intersection curves
  vtkNew<tf::vtk::intersection_curves> curves;
  curves->SetInputConnection(0, adapter0->GetOutputPort());
  curves->SetInputConnection(1, adapter1->GetOutputPort());
  curves->set_matrix0(matrix0);
  curves->set_matrix1(matrix1);

  // Visualization
  vtkNew<vtkOpenGLPolyDataMapper> mapper0;
  mapper0->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> actor0;
  actor0->SetMapper(mapper0);
  actor0->SetUserMatrix(matrix0);
  actor0->GetProperty()->SetColor(0.8, 0.8, 0.9);

  vtkNew<vtkOpenGLPolyDataMapper> mapper1;
  mapper1->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> actor1;
  actor1->SetMapper(mapper1);
  actor1->SetUserMatrix(matrix1);
  actor1->GetProperty()->SetColor(0.9, 0.8, 0.8);

  vtkNew<vtkTubeFilter> tube;
  tube->SetInputConnection(curves->GetOutputPort());
  tube->SetRadius(0.0005);
  tube->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> curves_mapper;
  curves_mapper->SetInputConnection(tube->GetOutputPort());
  vtkNew<vtkOpenGLActor> curves_actor;
  curves_actor->SetMapper(curves_mapper);
  curves_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(actor0);
  renderer->AddActor(actor1);
  renderer->AddActor(curves_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<tf::vtk::examples::drag_interactor> style;
  style->add_actor(actor0, renderer);
  style->add_actor(actor1, renderer);
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
