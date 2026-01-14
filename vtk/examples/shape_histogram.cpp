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
#include <array>
#include <iostream>
#include <trueform/trueform.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/functions.hpp>
#include <vtkBarChartActor.h>
#include <vtkDataObject.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLegendBoxActor.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>

namespace {

// Histogram parameters
constexpr int num_bins = 20;
constexpr float bin_width = 2.0f / num_bins; // Range [-1, 1]

// Colors
constexpr std::array<unsigned char, 3> white = {255, 255, 255};
constexpr std::array<unsigned char, 3> highlight = {255, 220, 200}; // Subtle peach

/// Custom interactor for shape histogram visualization
class shape_histogram_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static shape_histogram_interactor *New();
  vtkTypeMacro(shape_histogram_interactor, vtkInteractorStyleTrackballCamera);

  void set_data(tf::vtk::polydata *poly, const tf::buffer<float> &shape_index,
                vtkOpenGLRenderer *mesh_renderer, vtkBarChartActor *bar_chart,
                vtkDataObject *data_object, vtkUnsignedCharArray *colors,
                vtkOpenGLActor *mesh_actor, float radius) {
    poly_ = poly;
    shape_index_ = &shape_index;
    mesh_renderer_ = mesh_renderer;
    bar_chart_ = bar_chart;
    data_object_ = data_object;
    colors_ = colors;
    mesh_actor_ = mesh_actor;
    radius_ = radius;
  }

  void OnMouseMove() override {
    int *pos = this->Interactor->GetEventPosition();

    // Create ray from screen position
    auto ray = tf::vtk::make_world_ray(mesh_renderer_, pos[0], pos[1]);

    // Pick against mesh actor
    std::vector<vtkActor *> actors = {mesh_actor_};
    auto result = tf::vtk::pick(ray, actors);

    if (result) {
      // Get face vertices and find closest to hit position
      auto faces = poly_->polys();
      auto points = poly_->points();
      auto face = faces[result.cell_id];

      vtkIdType closest_vertex = face[0];
      float min_dist2 = std::numeric_limits<float>::max();

      for (auto vid : face) {
        float d2 = tf::distance2(points[vid], result.position);
        if (d2 < min_dist2) {
          min_dist2 = d2;
          closest_vertex = vid;
        }
      }

      update_neighborhood(closest_vertex);
      this->Interactor->Render();
    } else {
      // Nothing picked - clear selection
      clear_selection();
    }

    vtkInteractorStyleTrackballCamera::OnMouseMove();
  }

private:
  void clear_selection() {
    if (!colors_ || !poly_ || previous_indices_.empty())
      return;

    auto points = poly_->points();
    const auto n_points = points.size();

    // Get colors range for indirect access
    auto colors_ptr = colors_->GetPointer(0);
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points));

    // Restore previous colors to white
    auto prev_colors = tf::make_indirect_range(previous_indices_, colors_range);
    tf::parallel_fill(prev_colors, white);

    previous_indices_.clear();
    last_vertex_ = -1;

    // Clear histogram
    auto histogram_data = vtkIntArray::SafeDownCast(
        data_object_->GetFieldData()->GetArray("Histogram"));
    for (int i = 0; i < num_bins; ++i) {
      histogram_data->SetValue(i, 0);
    }
    data_object_->Modified();
    bar_chart_->Modified();

    colors_->Modified();
    poly_->Modified();
    this->Interactor->Render();
  }

  void update_neighborhood(vtkIdType vertex_id) {
    if (!shape_index_ || !colors_ || !poly_)
      return;

    last_vertex_ = vertex_id;

    auto points = poly_->points();
    const auto &vlink = poly_->vertex_link();
    const auto n_points = points.size();

    // Get colors range for indirect access
    auto colors_ptr = colors_->GetPointer(0);
    auto colors_range =
        tf::make_blocked_range<3>(tf::make_range(colors_ptr, n_points));

    // Restore previous colors to white
    if (!previous_indices_.empty()) {
      auto prev_colors =
          tf::make_indirect_range(previous_indices_, colors_range);
      tf::parallel_fill(prev_colors, white);
    }

    // Collect new neighborhood indices
    current_indices_.clear();
    applier_(
        vlink, vertex_id,
        [&points](vtkIdType seed, vtkIdType neighbor) {
          return tf::distance2(points[seed], points[neighbor]);
        },
        radius_, true,
        [this](vtkIdType idx) { current_indices_.push_back(idx); });

    // Color current neighborhood
    auto neigh_colors = tf::make_indirect_range(current_indices_, colors_range);
    tf::parallel_fill(neigh_colors, highlight);

    // Compute histogram
    std::array<int, num_bins> histogram{};
    auto neigh_si = tf::make_indirect_range(current_indices_, *shape_index_);
    for (float si : neigh_si) {
      int bin = std::clamp(static_cast<int>((si + 1.0f) / bin_width), 0,
                           num_bins - 1);
      histogram[bin]++;
    }

    // Update histogram data
    auto histogram_data = vtkIntArray::SafeDownCast(
        data_object_->GetFieldData()->GetArray("Histogram"));
    for (int i = 0; i < num_bins; ++i) {
      histogram_data->SetValue(i, histogram[i]);
    }
    data_object_->Modified();
    bar_chart_->Modified();

    // Store current as previous for next update
    previous_indices_ = current_indices_;

    colors_->Modified();
    poly_->Modified();
  }

  tf::vtk::polydata *poly_ = nullptr;
  const tf::buffer<float> *shape_index_ = nullptr;
  vtkOpenGLRenderer *mesh_renderer_ = nullptr;
  vtkBarChartActor *bar_chart_ = nullptr;
  vtkDataObject *data_object_ = nullptr;
  vtkUnsignedCharArray *colors_ = nullptr;
  vtkOpenGLActor *mesh_actor_ = nullptr;

  float radius_ = 1.0f;
  vtkIdType last_vertex_ = -1;
  tf::topology::neighborhood_applier<vtkIdType> applier_;
  std::vector<vtkIdType> current_indices_;
  std::vector<vtkIdType> previous_indices_;
};

vtkStandardNewMacro(shape_histogram_interactor);

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mesh.stl>" << std::endl;
    return 1;
  }

  // Load mesh using tf::vtk::read_stl (returns polydata with cached structures)
  std::cout << "Loading mesh: " << argv[1] << std::endl;
  auto poly = tf::vtk::read_stl(argv[1]);

  auto points = poly->points();
  const auto n_vertices = points.size();
  std::cout << "Vertices: " << n_vertices << std::endl;

  // Compute AABB and initial radius (7.5% of diagonal)
  auto aabb = tf::vtk::aabb_from(poly);
  float diag = aabb.diagonal().length();
  float initial_radius = diag * 0.075f;
  std::cout << "AABB diagonal: " << diag
            << ", initial radius: " << initial_radius << std::endl;

  // Compute shape index
  std::cout << "Computing shape index..." << std::endl;
  tf::buffer<float> shape_index;
  shape_index.allocate(n_vertices);
  tf::compute_shape_index(poly->polygons(), shape_index);
  std::cout << "Shape index computed." << std::endl;

  // Initialize vertex colors to white
  vtkNew<vtkUnsignedCharArray> colors;
  colors->SetNumberOfComponents(3);
  colors->SetNumberOfTuples(n_vertices);
  colors->SetName("Colors");

  auto colors_ptr =
      reinterpret_cast<std::array<unsigned char, 3> *>(colors->GetPointer(0));
  auto colors_range = tf::make_range(colors_ptr, n_vertices);
  tf::parallel_fill(colors_range, white);

  poly->GetPointData()->SetScalars(colors);

  // Create mesh mapper and actor
  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputData(poly);
  mapper->SetScalarModeToUsePointData();
  mapper->SetColorModeToDirectScalars();

  vtkNew<vtkOpenGLActor> mesh_actor;
  mesh_actor->SetMapper(mapper);

  // Create mesh renderer (left 70%)
  vtkNew<vtkOpenGLRenderer> mesh_renderer;
  mesh_renderer->SetViewport(0.0, 0.0, 0.7, 1.0);
  mesh_renderer->AddActor(mesh_actor);
  mesh_renderer->SetBackground(0.1, 0.1, 0.15);

  // Create histogram data object
  vtkNew<vtkDataObject> data_object;
  vtkNew<vtkIntArray> histogram_data;
  histogram_data->SetName("Histogram");
  histogram_data->SetNumberOfTuples(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    histogram_data->SetValue(i, 0);
  }
  data_object->GetFieldData()->AddArray(histogram_data);

  // Create bar chart actor
  vtkNew<vtkBarChartActor> bar_chart;
  bar_chart->SetInput(data_object);
  bar_chart->SetTitle("Shape Index Histogram");
  bar_chart->SetYTitle("");
  bar_chart->GetPositionCoordinate()->SetValue(0.05, 0.05, 0.0);
  bar_chart->GetPosition2Coordinate()->SetValue(0.95, 0.85, 0.0);
  bar_chart->GetLegendActor()->SetNumberOfEntries(num_bins);
  bar_chart->LegendVisibilityOff();
  bar_chart->GetTitleTextProperty()->SetColor(1.0, 1.0, 1.0);
  bar_chart->GetTitleTextProperty()->SetFontSize(14);

  // Set bar colors (highlight gradient)
  for (int i = 0; i < num_bins; ++i) {
    float t = static_cast<float>(i) / (num_bins - 1);
    bar_chart->SetBarColor(i, 1.0, 0.5 + 0.3 * t, 0.3 + 0.2 * t);
  }

  // Create histogram renderer (right 30%)
  vtkNew<vtkOpenGLRenderer> histogram_renderer;
  histogram_renderer->SetViewport(0.7, 0.0, 1.0, 1.0);
  histogram_renderer->AddActor(bar_chart);
  histogram_renderer->SetBackground(0.15, 0.15, 0.2);

  // Create render window with both renderers
  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(mesh_renderer);
  window->AddRenderer(histogram_renderer);
  window->SetSize(1400, 800);
  window->SetWindowName("Shape Index Histogram");

  // Create interactor
  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  // Set up custom interactor style
  vtkNew<shape_histogram_interactor> style;
  style->set_data(poly, shape_index, mesh_renderer, bar_chart, data_object,
                  colors, mesh_actor, initial_radius);
  interactor->SetInteractorStyle(style);

  mesh_renderer->ResetCamera();
  window->Render();

  std::cout << "Controls:" << std::endl;
  std::cout << "  Mouse move over mesh: Select neighborhood" << std::endl;
  std::cout << "  Left mouse drag: Rotate camera" << std::endl;
  std::cout << "  Right mouse drag: Zoom" << std::endl;
  std::cout << "  Middle mouse drag: Pan" << std::endl;

  interactor->Start();

  return 0;
}
