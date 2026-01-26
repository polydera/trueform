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
#include <trueform/vtk/filters/boolean.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <util/drag_interactor.hpp>
#include <vtkCamera.h>
#include <vtkLookupTable.h>
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

  // Compute boolean
  vtkNew<tf::vtk::boolean> boolean_filter;
  boolean_filter->SetInputConnection(0, adapter0->GetOutputPort());
  boolean_filter->SetInputConnection(1, adapter1->GetOutputPort());
  boolean_filter->set_matrix0(matrix0);
  boolean_filter->set_matrix1(matrix1);
  boolean_filter->set_operation(tf::boolean_op::left_difference);
  boolean_filter->set_return_curves(true);

  // Input mesh actors (left viewport)
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

  // Intersection curves actor (left viewport)
  vtkNew<vtkTubeFilter> tube;
  tube->SetInputConnection(boolean_filter->GetOutputPort(1));
  tube->SetRadius(0.0005);
  tube->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> curves_mapper;
  curves_mapper->SetInputConnection(tube->GetOutputPort());
  vtkNew<vtkOpenGLActor> curves_actor;
  curves_actor->SetMapper(curves_mapper);
  curves_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  // Lookup table mapping labels to mesh colors
  vtkNew<vtkLookupTable> lut;
  lut->SetNumberOfTableValues(2);
  lut->SetTableValue(0, 0.8, 0.8, 0.9, 1.0);  // mesh 0 color
  lut->SetTableValue(1, 0.9, 0.8, 0.8, 1.0);  // mesh 1 color
  lut->SetTableRange(0, 1);
  lut->Build();

  // Result mesh actor (right viewport)
  vtkNew<vtkOpenGLPolyDataMapper> result_mapper;
  result_mapper->SetInputConnection(boolean_filter->GetOutputPort(0));
  result_mapper->SetScalarModeToUseCellData();
  result_mapper->SetLookupTable(lut);
  result_mapper->SetScalarRange(0, 1);
  vtkNew<vtkOpenGLActor> result_actor;
  result_actor->SetMapper(result_mapper);

  // Left renderer: input meshes and curves
  vtkNew<vtkOpenGLRenderer> renderer_left;
  renderer_left->AddActor(actor0);
  renderer_left->AddActor(actor1);
  renderer_left->AddActor(curves_actor);
  renderer_left->SetBackground(0.1, 0.1, 0.15);
  renderer_left->SetViewport(0.0, 0.0, 0.5, 1.0);

  // Right renderer: result mesh
  vtkNew<vtkOpenGLRenderer> renderer_right;
  renderer_right->AddActor(result_actor);
  renderer_right->SetBackground(0.1, 0.1, 0.15);
  renderer_right->SetViewport(0.5, 0.0, 1.0, 1.0);

  // Share camera between renderers
  renderer_right->SetActiveCamera(renderer_left->GetActiveCamera());

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer_left);
  window->AddRenderer(renderer_right);
  window->SetSize(1600, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<tf::vtk::examples::drag_interactor> style;
  style->add_actor(actor0, renderer_left);
  style->add_actor(actor1, renderer_left);
  interactor->SetInteractorStyle(style);

  renderer_left->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
