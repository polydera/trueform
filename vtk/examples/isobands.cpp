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
#include <trueform/vtk/filters/isobands.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
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

class isoband_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> isoband_interactor *;
  vtkTypeMacro(isoband_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(tf::vtk::isobands *bands, float min_z, float max_z) -> void {
    _bands = bands;
    _min_z = min_z;
    _max_z = max_z;
    _offset = 0.0f;
    _spacing = (_max_z - _min_z) / static_cast<float>(_num_bands);
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
  isoband_interactor() = default;
  ~isoband_interactor() override = default;

private:
  auto update_cut_values() -> void {
    std::vector<float> cut_values;
    std::vector<int> selected_bands;

    // Use fmod to wrap offset within spacing
    float wrapped_offset = std::fmod(_offset, _spacing);
    if (wrapped_offset < 0)
      wrapped_offset += _spacing;

    // Generate evenly spaced cut values offset by wrapped_offset
    for (int i = 0; i <= _num_bands; ++i) {
      float value = _min_z + wrapped_offset + i * _spacing;
      cut_values.push_back(value);
    }

    // Select every other band (alternating pattern)
    int parity = static_cast<int>(std::floor(_offset / _spacing)) & 1;
    for (int i = 0; i < _num_bands; ++i) {
      if ((i & 1) == parity) {
        selected_bands.push_back(i);
      }
    }

    _bands->set_cut_values(std::move(cut_values));
    _bands->set_selected_bands(std::move(selected_bands));
  }

  tf::vtk::isobands *_bands = nullptr;
  float _min_z = 0.0f;
  float _max_z = 1.0f;
  float _offset = 0.0f;
  float _spacing = 0.1f;
  int _num_bands = 10;
};

vtkStandardNewMacro(isoband_interactor);

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

  float min_d = *std::min_element(scalars_range.begin(), scalars_range.end());
  float max_d = *std::max_element(scalars_range.begin(), scalars_range.end());

  poly->GetPointData()->SetScalars(scalars);

  // Compute isobands
  vtkNew<tf::vtk::isobands> bands;
  bands->SetInputConnection(reader->GetOutputPort());
  bands->set_return_curves(true);

  // Visualization - original mesh (faded)
  vtkNew<vtkOpenGLPolyDataMapper> mesh_mapper;
  mesh_mapper->SetInputConnection(reader->GetOutputPort());
  mesh_mapper->ScalarVisibilityOff();
  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mesh_mapper);
  mesh_actor->GetProperty()->SetColor(0.5, 0.5, 0.55);
  mesh_actor->GetProperty()->SetOpacity(0.15);

  // Visualization - isobands
  vtkNew<vtkOpenGLPolyDataMapper> band_mapper;
  band_mapper->SetInputConnection(bands->GetOutputPort(0));
  band_mapper->SetScalarModeToUseCellData();
  band_mapper->SetColorModeToMapScalars();

  // Create a lookup table for band colors (shades of teal)
  vtkNew<vtkLookupTable> lut;
  lut->SetNumberOfTableValues(10);
  for (int i = 0; i < 10; ++i) {
    float t = static_cast<float>(i) / 9.0f;
    lut->SetTableValue(i, 0.0 + 0.35 * t, 0.4 + 0.4 * t, 0.36 + 0.38 * t, 1.0);
  }
  lut->Build();
  band_mapper->SetLookupTable(lut);
  band_mapper->SetScalarRange(0, 9);

  vtkNew<vtkOpenGLActor> band_actor;
  band_actor->SetMapper(band_mapper);

  // Visualization - boundary curves as tubes
  vtkNew<vtkTubeFilter> tube;
  tube->SetInputConnection(bands->GetOutputPort(1));
  tube->SetRadius(0.0003);
  tube->SetNumberOfSides(12);

  vtkNew<vtkOpenGLPolyDataMapper> curve_mapper;
  curve_mapper->SetInputConnection(tube->GetOutputPort());
  vtkNew<vtkOpenGLActor> curve_actor;
  curve_actor->SetMapper(curve_mapper);
  curve_actor->GetProperty()->SetColor(0.0, 0.95, 0.85);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(mesh_actor);
  renderer->AddActor(band_actor);
  renderer->AddActor(curve_actor);
  renderer->SetBackground(0.1, 0.1, 0.15);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(renderer);
  window->SetSize(1200, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<isoband_interactor> style;
  style->initialize(bands, min_d, max_d);
  interactor->SetInteractorStyle(style);

  renderer->ResetCamera();
  window->Render();
  interactor->Start();

  return 0;
}
