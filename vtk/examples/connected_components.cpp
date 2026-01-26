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
#include <trueform/core.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/filters/adapter.hpp>
#include <trueform/vtk/filters/connected_components.hpp>
#include <trueform/vtk/filters/isobands.hpp>
#include <trueform/vtk/filters/stl_reader.hpp>
#include <trueform/vtk/functions/split_into_components.hpp>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

int main() {
  // Load dragon mesh
  vtkNew<tf::vtk::stl_reader> reader;
  reader->set_file_name(TRUEFORM_DATA_DIR "/benchmarks/data/dragon-500k.stl");
  reader->Update();

  auto *poly = reader->GetOutput();

  // Create scalar field based on Z coordinate
  auto points = tf::vtk::make_points(poly);

  vtkNew<vtkFloatArray> scalars;
  scalars->SetName("height");
  scalars->SetNumberOfTuples(poly->GetNumberOfPoints());

  auto scalars_range = tf::vtk::make_range(scalars.Get());
  tf::parallel_transform(points, scalars_range,
                         [](const auto &p) { return p[2]; });

  float min_z = *std::min_element(scalars_range.begin(), scalars_range.end());
  float max_z = *std::max_element(scalars_range.begin(), scalars_range.end());

  poly->GetPointData()->SetScalars(scalars);

  // Use isobands to create alternating stripes
  vtkNew<tf::vtk::isobands> bands;
  bands->SetInputConnection(reader->GetOutputPort());

  // Create cut values for 10 bands
  std::vector<float> cut_values;
  for (int i = 0; i <= 10; ++i) {
    cut_values.push_back(min_z + (max_z - min_z) * i / 10.0f);
  }
  bands->set_cut_values(cut_values);

  // Select alternating bands (0, 2, 4, 6, 8) to create disconnected regions
  bands->set_selected_bands({0, 2, 4, 6, 8});
  bands->Update();

  // Adapt isobands output for connected components
  vtkNew<tf::vtk::adapter> adapt;
  adapt->SetInputConnection(bands->GetOutputPort(0));

  // Label connected components using edge connectivity
  vtkNew<tf::vtk::connected_components> cc;
  cc->SetInputConnection(adapt->GetOutputPort());
  cc->set_connectivity(tf::connectivity_type::edge);
  cc->Update();

  std::cout << "Found " << cc->n_components() << " connected components"
            << std::endl;

  // Split into separate polydata objects
  auto [components, component_labels] =
      tf::vtk::split_into_components(cc->GetOutput());
  std::cout << "Split into " << components.size() << " separate meshes"
            << std::endl;

  // Print some stats about each component
  for (auto [label, component] : tf::zip(component_labels, components)) {
    std::cout << "  Component " << label << ": "
              << component->GetNumberOfPoints() << " points, "
              << component->GetNumberOfPolys() << " faces" << std::endl;
  }

  // Create color lookup table for components
  vtkNew<vtkLookupTable> lut;
  lut->SetNumberOfColors(cc->n_components());
  lut->SetHueRange(0.0, 0.8);
  lut->Build();

  // Visualization
  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputConnection(cc->GetOutputPort());
  mapper->SetScalarModeToUseCellData();
  mapper->SelectColorArray("ComponentLabel");
  mapper->SetScalarRange(0, cc->n_components() - 1);
  mapper->SetLookupTable(lut);

  vtkNew<vtkOpenGLActor> actor;
  actor->SetMapper(mapper);

  vtkNew<vtkOpenGLRenderer> renderer;
  renderer->AddActor(actor);
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
