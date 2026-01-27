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
#include <trueform/geometry/triangulated.hpp>
#include <trueform/trueform.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <trueform/vtk/functions/make_isocontours.hpp>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTubeFilter.h>

class cross_section_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> cross_section_interactor *;
  vtkTypeMacro(cross_section_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(vtkPolyData *mesh, vtkOpenGLPolyDataMapper *slice_mapper,
                  vtkTubeFilter *tube, float min_z, float max_z) -> void {
    _mesh = mesh;
    _slice_mapper = slice_mapper;
    _tube = tube;
    _min_z = min_z;
    _max_z = max_z;
    _range = max_z - min_z;
    _cut_value = (_min_z + _max_z) * 0.5f;
    _step = _range * 0.0075f;
    _margin = _range * 0.01f;
    update_cut();
  }

  auto OnMouseWheelBackward() -> void override {
    _cut_value = std::max(_min_z + _margin, _cut_value - _step);
    update_cut();
    this->Interactor->Render();
  }

  auto OnMouseWheelForward() -> void override {
    _cut_value = std::min(_max_z - _margin, _cut_value + _step);
    update_cut();
    this->Interactor->Render();
  }

protected:
  cross_section_interactor() = default;
  ~cross_section_interactor() override = default;

private:
  auto update_cut() -> void {
    // Get curves at cut value
    auto curves = tf::vtk::make_isocontours(_mesh, nullptr, {_cut_value});
    _tube->SetInputData(curves);

    // Triangulate curves into filled cross-section polygons
    auto curve_data = tf::vtk::make_curves(curves);
    auto slices = tf::triangulated(
        tf::make_polygons(curve_data.paths(), curve_data.points()));

    _slice_mapper->SetInputData(tf::vtk::make_vtk_polydata(std::move(slices)));
  }

  vtkPolyData *_mesh = nullptr;
  vtkOpenGLPolyDataMapper *_slice_mapper = nullptr;
  vtkTubeFilter *_tube = nullptr;
  float _min_z = 0.0f;
  float _max_z = 1.0f;
  float _range = 1.0f;
  float _cut_value = 0.5f;
  float _step = 0.01f;
  float _margin = 0.05f;
};

vtkStandardNewMacro(cross_section_interactor);

int main() {
  // Load mesh
  vtkNew<tf::vtk::stl_reader> reader;
  reader->set_file_name(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  reader->Update();

  auto *poly = reader->GetOutput();

  // Create scalar field as signed distance to a plane through centroid
  auto points = tf::vtk::make_points(poly);
  auto center = tf::centroid(points);
  auto normal = tf::make_unit_vector(1.f, 2.f, 1.f);
  auto plane = tf::make_plane(normal, center);

  vtkNew<vtkFloatArray> scalars;
  scalars->SetName("plane_distance");
  scalars->SetNumberOfTuples(poly->GetNumberOfPoints());

  auto scalars_range = tf::vtk::make_range(scalars.Get());
  tf::parallel_transform(points, scalars_range, tf::distance_f(plane));

  float min_z = *std::min_element(scalars_range.begin(), scalars_range.end());
  float max_z = *std::max_element(scalars_range.begin(), scalars_range.end());

  poly->GetPointData()->SetScalars(scalars);

  // Visualization - original mesh (faded)
  vtkNew<vtkOpenGLPolyDataMapper> mesh_mapper;
  mesh_mapper->SetInputConnection(reader->GetOutputPort());
  mesh_mapper->ScalarVisibilityOff();
  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mesh_mapper);
  mesh_actor->GetProperty()->SetColor(0.5, 0.5, 0.55);
  mesh_actor->GetProperty()->SetOpacity(0.15);

  // Visualization - cross-section fill (subtle)
  vtkNew<vtkOpenGLPolyDataMapper> slice_mapper;
  vtkNew<vtkOpenGLActor> slice_actor;
  slice_actor->SetMapper(slice_mapper);
  slice_actor->GetProperty()->SetColor(0.0, 0.6, 0.54);

  // Visualization - contour curves as tubes (stronger)
  vtkNew<vtkTubeFilter> tube;
  tube->SetRadius(0.0003);
  tube->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> curve_mapper;
  curve_mapper->SetInputConnection(tube->GetOutputPort());
  vtkNew<vtkOpenGLActor> curve_actor;
  curve_actor->SetMapper(curve_mapper);
  curve_actor->GetProperty()->SetColor(0.0, 0.95, 0.85);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(mesh_actor);
  renderer->AddActor(slice_actor);
  renderer->AddActor(curve_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<cross_section_interactor> style;
  style->initialize(poly, slice_mapper, tube, min_z, max_z);
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
