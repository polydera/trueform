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
#include <cmath>
#include <trueform/core.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/filters/isocontours.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
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

class isocontour_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> isocontour_interactor *;
  vtkTypeMacro(isocontour_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(tf::vtk::isocontours *iso, float min_z, float max_z) -> void {
    _iso = iso;
    _min_z = min_z;
    _max_z = max_z;
    _offset = 0.0f;
    _spacing = (_max_z - _min_z) / static_cast<float>(_num_contours);
    update_cut_values();
  }

  auto OnMouseWheelBackward() -> void override {
    _offset -= _spacing * 0.1f;
    update_cut_values();
    this->Interactor->Render();
  }

  auto OnMouseWheelForward() -> void override {
    _offset += _spacing * 0.1f;
    update_cut_values();
    this->Interactor->Render();
  }

protected:
  isocontour_interactor() = default;
  ~isocontour_interactor() override = default;

private:
  auto update_cut_values() -> void {
    std::vector<float> cut_values;

    // Use fmod to wrap offset within spacing
    float wrapped_offset = std::fmod(_offset, _spacing);
    if (wrapped_offset < 0)
      wrapped_offset += _spacing;

    // Generate evenly spaced cut values offset by wrapped_offset
    for (int i = 0; i < _num_contours; ++i) {
      float value = _min_z + wrapped_offset + i * _spacing;
      if (value > _min_z && value < _max_z) {
        cut_values.push_back(value);
      }
    }

    _iso->set_cut_values(std::move(cut_values));
  }

  tf::vtk::isocontours *_iso = nullptr;
  float _min_z = 0.0f;
  float _max_z = 1.0f;
  float _offset = 0.0f;
  float _spacing = 0.1f;
  int _num_contours = 20;
};

vtkStandardNewMacro(isocontour_interactor);

int main() {
  // Load mesh
  vtkNew<tf::vtk::stl_reader> reader;
  reader->set_file_name(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  reader->Update();

  auto *poly = reader->GetOutput();

  // Create scalar field as distance from a plane through centroid
  auto points = tf::vtk::make_points(poly);
  auto center = tf::centroid(points);

  vtkNew<vtkFloatArray> scalars;
  scalars->SetName("distance");
  scalars->SetNumberOfTuples(poly->GetNumberOfPoints());

  auto scalars_range = tf::vtk::make_range(scalars.Get());
  tf::parallel_transform(points, scalars_range, tf::distance_f(center));

  float min_d = *std::min_element(scalars_range.begin(), scalars_range.end());
  float max_d = *std::max_element(scalars_range.begin(), scalars_range.end());

  poly->GetPointData()->SetScalars(scalars);

  // Compute isocontours
  vtkNew<tf::vtk::isocontours> iso;
  iso->SetInputConnection(reader->GetOutputPort());

  // Visualization - mesh
  vtkNew<vtkOpenGLPolyDataMapper> mesh_mapper;
  mesh_mapper->SetInputConnection(reader->GetOutputPort());
  mesh_mapper->ScalarVisibilityOff();
  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mesh_mapper);
  mesh_actor->GetProperty()->SetColor(0.8, 0.8, 0.9);

  // Visualization - isocontours as tubes
  vtkNew<vtkTubeFilter> tube;
  tube->SetInputConnection(iso->GetOutputPort());
  tube->SetRadius(0.0003);
  tube->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> contour_mapper;
  contour_mapper->SetInputConnection(tube->GetOutputPort());
  vtkNew<vtkOpenGLActor> contour_actor;
  contour_actor->SetMapper(contour_mapper);
  contour_actor->GetProperty()->SetColor(1.0, 0.2, 0.2);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(mesh_actor);
  renderer->AddActor(contour_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<isocontour_interactor> style;
  style->initialize(iso, min_d, max_d);
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
