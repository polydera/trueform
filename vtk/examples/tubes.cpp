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
#include <chrono>
#include <iostream>
#include <trueform/core.hpp>
#include <trueform/geometry/make_tube_mesh.hpp>
#include <trueform/intersect.hpp>
#include <trueform/topology.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <trueform/vtk/functions.hpp>
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

  auto *poly = tf::vtk::polydata::SafeDownCast(reader->GetOutput());

  // Initial transformation
  auto points = tf::vtk::make_points(reader->GetOutput());
  auto center = tf::centroid(points);
  auto rotation =
      tf::make_rotation(tf::deg<double>{90.0}, tf::axis<2>, center);

  vtkNew<vtkMatrix4x4> matrix0;
  auto matrix1 = tf::vtk::make_vtk_matrix(rotation);

  // Left viewport: mesh actors + VTK tube filter (red)
  vtkNew<vtkOpenGLPolyDataMapper> mapper0_left;
  mapper0_left->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> actor0_left;
  actor0_left->SetMapper(mapper0_left);
  actor0_left->SetUserMatrix(matrix0);
  actor0_left->GetProperty()->SetColor(0.8, 0.8, 0.9);

  vtkNew<vtkOpenGLPolyDataMapper> mapper1_left;
  mapper1_left->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> actor1_left;
  actor1_left->SetMapper(mapper1_left);
  actor1_left->SetUserMatrix(matrix1);
  actor1_left->GetProperty()->SetColor(0.9, 0.8, 0.8);

  vtkNew<vtkOpenGLPolyDataMapper> vtk_tubes_mapper;
  vtkNew<vtkOpenGLActor> vtk_tubes_actor;
  vtk_tubes_actor->SetMapper(vtk_tubes_mapper);
  vtk_tubes_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  // Right viewport: mesh actors + trueform tubes (green)
  vtkNew<vtkOpenGLPolyDataMapper> mapper0_right;
  mapper0_right->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> actor0_right;
  actor0_right->SetMapper(mapper0_right);
  actor0_right->SetUserMatrix(matrix0);
  actor0_right->GetProperty()->SetColor(0.8, 0.8, 0.9);

  vtkNew<vtkOpenGLPolyDataMapper> mapper1_right;
  mapper1_right->SetInputConnection(reader->GetOutputPort());
  vtkNew<vtkOpenGLActor> actor1_right;
  actor1_right->SetMapper(mapper1_right);
  actor1_right->SetUserMatrix(matrix1);
  actor1_right->GetProperty()->SetColor(0.9, 0.8, 0.8);

  vtkNew<vtkOpenGLPolyDataMapper> tf_tubes_mapper;
  vtkNew<vtkOpenGLActor> tf_tubes_actor;
  tf_tubes_actor->SetMapper(tf_tubes_mapper);
  tf_tubes_actor->GetProperty()->SetColor(0.2, 1.0, 0.2);

  // Recompute callback
  auto compute_and_update = [&]() {
    auto frame0 = tf::vtk::make_frame(matrix0);
    auto frame1 = tf::vtk::make_frame(matrix1);

    auto base = poly->polygons() | tf::tag(poly->face_membership()) |
                tf::tag(poly->manifold_edge_link()) | tf::tag(poly->poly_tree());

    // Compute intersection curves (shared by both)
    auto tc0 = std::chrono::high_resolution_clock::now();
    auto curves = tf::make_intersection_curves(
        base | tf::tag(frame0), base | tf::tag(frame1));
    auto tc1 = std::chrono::high_resolution_clock::now();

    // --- trueform path: make_tubes directly (before move!) ---
    auto num_curves = curves.size();
    auto tt0 = std::chrono::high_resolution_clock::now();
    auto tf_tube_mesh = tf::make_tube_mesh(curves.curves(), 0.0005f, 12);
    auto tt1 = std::chrono::high_resolution_clock::now();

    // Verify topology
    auto tube_polys = tf_tube_mesh.polygons();
    std::cout << "  closed: " << tf::is_closed(tube_polys)
              << ", manifold: " << tf::is_manifold(tube_polys) << "\n";

    tf_tubes_mapper->SetInputData(tf::vtk::make_vtk_polydata(tf_tube_mesh));

    // --- VTK path: convert to VTK polydata, then vtkTubeFilter ---
    auto tv0 = std::chrono::high_resolution_clock::now();
    auto vtk_curves_pd = tf::vtk::make_vtk_polydata(std::move(curves));
    auto tv1 = std::chrono::high_resolution_clock::now();

    vtkNew<vtkTubeFilter> vtk_tube;
    vtk_tube->SetInputData(vtk_curves_pd);
    vtk_tube->SetRadius(0.0005);
    vtk_tube->SetNumberOfSides(12);

    auto tv2 = std::chrono::high_resolution_clock::now();
    vtk_tube->Update();
    auto tv3 = std::chrono::high_resolution_clock::now();

    vtk_tubes_mapper->SetInputConnection(vtk_tube->GetOutputPort());

    auto curves_ms =
        std::chrono::duration<double, std::milli>(tc1 - tc0).count();
    auto vtk_copy_ms =
        std::chrono::duration<double, std::milli>(tv1 - tv0).count();
    auto vtk_tube_ms =
        std::chrono::duration<double, std::milli>(tv3 - tv2).count();
    auto tf_tube_ms =
        std::chrono::duration<double, std::milli>(tt1 - tt0).count();

    std::cout << "curves: " << curves_ms << " ms | "
              << "VTK copy: " << vtk_copy_ms << " ms, "
              << "VTK tube: " << vtk_tube_ms << " ms | "
              << "TF tubes: " << tf_tube_ms << " ms ("
              << num_curves << " curves, "
              << tf_tube_mesh.faces_buffer().size() << " tris)\n";
  };

  compute_and_update();

  // Left renderer: VTK tubes
  vtkNew<vtkOpenGLRenderer> renderer_left;
  renderer_left->AddActor(actor0_left);
  renderer_left->AddActor(actor1_left);
  renderer_left->AddActor(vtk_tubes_actor);
  renderer_left->SetBackground(0.1, 0.1, 0.15);
  renderer_left->SetViewport(0.0, 0.0, 0.5, 1.0);

  // Right renderer: trueform tubes
  vtkNew<vtkOpenGLRenderer> renderer_right;
  renderer_right->AddActor(actor0_right);
  renderer_right->AddActor(actor1_right);
  renderer_right->AddActor(tf_tubes_actor);
  renderer_right->SetBackground(0.1, 0.1, 0.15);
  renderer_right->SetViewport(0.5, 0.0, 1.0, 1.0);
  renderer_right->SetActiveCamera(renderer_left->GetActiveCamera());

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer_left);
  window->AddRenderer(renderer_right);
  window->SetSize(1600, 900);
  window->SetWindowName("VTK vtkTubeFilter vs tf::make_tubes");

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  // Drag interactor — both actors in left renderer
  vtkNew<tf::vtk::examples::drag_interactor> style;
  style->add_actor(actor0_left, renderer_left);
  style->add_actor(actor1_left, renderer_left);
  style->set_callback(
      [&](vtkActor *, std::vector<vtkActor *> &) { compute_and_update(); });

  interactor->SetInteractorStyle(style);

  renderer_left->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
