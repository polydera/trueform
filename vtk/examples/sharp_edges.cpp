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
#include <iostream>
#include <trueform/trueform.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

int main() {
  auto box = tf::make_box_mesh(2.f, 3.f, 4.f, 10, 10, 10);
  auto polygons = box.polygons();

  auto edges = tf::make_sharp_edges(polygons, tf::deg(45.f));
  auto segments = tf::make_segments(edges, polygons.points());

  std::cout << "Box: " << polygons.faces().size() << " faces, "
            << polygons.points().size() << " points\n";
  std::cout << "Sharp edges (>45°): " << edges.size() << "\n";

  // Tube mesh from sharp edges for visualization
  auto edge_curves = tf::make_curves(
      tf::make_paths(edges), polygons.points());
  auto tubes = tf::make_tube_mesh(edge_curves, 0.02f, 6);

  std::cout << "Tube mesh: " << tubes.faces_buffer().size() << " faces\n";

  // Box mesh
  auto box_pd = tf::vtk::make_vtk_polydata(box);
  vtkNew<vtkOpenGLPolyDataMapper> box_mapper;
  box_mapper->SetInputData(box_pd);
  vtkNew<vtkOpenGLActor> box_actor;
  box_actor->SetMapper(box_mapper);
  box_actor->GetProperty()->SetColor(0.8, 0.8, 0.9);
  box_actor->GetProperty()->SetOpacity(0.5);

  // Edge tubes
  auto tubes_pd = tf::vtk::make_vtk_polydata(tubes);
  vtkNew<vtkOpenGLPolyDataMapper> tubes_mapper;
  tubes_mapper->SetInputData(tubes_pd);
  vtkNew<vtkOpenGLActor> tubes_actor;
  tubes_actor->SetMapper(tubes_mapper);
  tubes_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(box_actor);
  renderer->AddActor(tubes_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);
  window->SetWindowName("Sharp Edges (>45°)");

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
