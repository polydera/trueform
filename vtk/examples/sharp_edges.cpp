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
#include <trueform/vtk/filters/stl_reader.hpp>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

int main() {
  vtkNew<tf::vtk::stl_reader> reader;
  reader->set_file_name(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  reader->Update();

  auto *poly = tf::vtk::polydata::SafeDownCast(reader->GetOutput());
  auto polygons = poly->polygons();

  std::cout << "Mesh: " << polygons.faces().size() << " faces, "
            << polygons.points().size() << " points\n";

  // Sharp edges
  auto edges = tf::make_sharp_edges(polygons, tf::deg(30.f));
  std::cout << "Sharp edges (>30°): " << edges.size() << "\n";

  // Connect edges to paths
  auto paths = tf::connect_edges_to_paths(tf::make_edges(edges));
  auto curves = tf::make_curves(paths, polygons.points());
  std::cout << "Paths: " << curves.size() << "\n";

  // Tube mesh from curves
  auto mel = tf::mean_edge_length(polygons);
  auto tubes = tf::make_tube_mesh(curves, 0.0005340481836018277f);
  std::cout << "Tubes: " << tubes.faces_buffer().size() << " faces, "
            << tubes.points_buffer().size() << " points\n";

  // Mesh actor
  vtkNew<vtkOpenGLPolyDataMapper> mesh_mapper;
  mesh_mapper->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mesh_mapper);
  mesh_actor->GetProperty()->SetColor(0.8, 0.8, 0.9);

  // Tube actor
  auto tubes_pd = tf::vtk::make_vtk_polydata(tubes);
  vtkNew<vtkOpenGLPolyDataMapper> tubes_mapper;
  tubes_mapper->SetInputData(tubes_pd);
  vtkNew<vtkOpenGLActor> tubes_actor;
  tubes_actor->SetMapper(tubes_mapper);
  tubes_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(mesh_actor);
  renderer->AddActor(tubes_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);
  window->SetWindowName("Sharp Edges (>30°) — dragon 500k");

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
