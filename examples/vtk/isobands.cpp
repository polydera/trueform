#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkOpenGLActor.h"
#include "vtkOpenGLPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkTextActor.h"
#include "vtkTextProperty.h"
#include "vtkTubeFilter.h"

class cursor_interactor : public vtkInteractorStyleTrackballCamera {
private:
  tf::buffer<float> scalars;
  std::vector<float> times;
  int time_index = 0;
  vtkTextActor *text;
  vtkPolyData *poly;
  vtkPolyData *curve_poly;
  vtkPolyData *isobands_poly;
  float distance = 0;
  float min_d;
  float max_d;

  auto add_time(float t) {
    if (times.size() < 100) {
      times.push_back(t);
    } else {
      times[time_index] = t;
    }
    time_index = (time_index + 1) % 100;
    float sum = 0;
    for (auto time : times)
      sum += time;
    auto time = sum / times.size();
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "Isobands time per scroll: %.1f ms",
                  time);
    text->SetInput(buffer);
  }

  auto compute_curves() {
    auto polygons = get_triangles(poly);
    std::vector<float> cutvalues;
    std::vector<int> selected_values;
    int N = 10;

    const float s = (max_d - min_d) / static_cast<float>(N);
    const float a = (distance - min_d) / s;
    int k = static_cast<int>(std::floor(a));
    if (k < 0)
      k = 0;
    if (k >= N)
      k = N - 1;

    // Ensure cutvalues[k] == distance exactly
    for (int i = 0; i < N; ++i)
      cutvalues.push_back(distance + (i - k) * s);

    // Pick every other value but ensure we include the one at index k
    selected_values.clear();
    const int parity = k & 1; // same parity as k
    for (int i = 0; i < N; ++i)
      if ((i & 1) == parity)
        selected_values.push_back(i);
    tf::tick();
    /*auto curves = tf::make_isocontours(polygons, tf::make_range(scalars),*/
    /*                                   tf::make_range(cutvalues));*/
    auto [polys, _, curves] = tf::make_isobands<int>(
        polygons, scalars, tf::make_range(cutvalues),
        tf::make_range(selected_values), tf::return_curves);
    add_time(tf::tock());
    isobands_poly->ShallowCopy(to_polydata(std::move(polys)).get());
    isobands_poly->Modified();
    auto tmp_poly = curves_to_polydata(curves.curves());
    auto tubes = vtk_make_unique<vtkTubeFilter>();
    tubes->SetRadius(0.05);
    tubes->SetInputData(tmp_poly.get());
    tubes->Update();
    curve_poly->ShallowCopy(tubes->GetOutput());
    curve_poly->Modified();
    this->Interactor->Render();
  }

  auto reset_plane() -> void {
    auto points = get_points(poly);
    auto plane = tf::make_plane(tf::normalized(tf::random_vector<float, 3>()),
                                points[tf::random<int>(0, points.size() - 1)]);
    scalars.allocate(points.size());
    tf::parallel_transform(points, scalars, tf::distance_f(plane));
    distance = 0;
    min_d = *std::min_element(scalars.begin(), scalars.end());
    max_d = *std::max_element(scalars.begin(), scalars.end());
  }

public:
  auto initialize(vtkPolyData *_poly, vtkPolyData *_isobands_poly,
                  vtkPolyData *_cpoly, vtkTextActor *_text) -> void {
    poly = _poly;
    isobands_poly = _isobands_poly;
    curve_poly = _cpoly;
    text = _text;
    reset_plane();
    compute_curves();
  }

  auto OnKeyPress() -> void override {
    std::string key = this->GetInteractor()->GetKeySym();
    if (key == "n") {
      reset_plane();
      compute_curves();
    } else {
      vtkInteractorStyleTrackballCamera::OnKeyPress();
    }
  }

  auto OnMouseWheelBackward() -> void override {
    if (this->Interactor->GetShiftKey()) {
      distance -= 0.05;
      compute_curves();
    } else
      vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
  }

  auto OnMouseWheelForward() -> void override {
    if (this->Interactor->GetShiftKey()) {
      distance += 0.05;
      compute_curves();
    } else
      vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
  }

  static cursor_interactor *New();
  vtkTypeMacro(cursor_interactor, vtkInteractorStyleTrackballCamera)
};

vtkStandardNewMacro(cursor_interactor);

int main(int argc, char *argv[]) {
  if (argc < 2 || (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                                 std::string_view(argv[1]) == "--help"))) {
    std::cerr << "Usage: program <input.stl|obj>\n";
    return 1;
  }

  // Load + normalize input
  auto poly = read_mesh(argv[1]);
  center_and_scale(poly.get());

  // Core VTK objects
  auto inter = vtk_make_unique<cursor_interactor>();
  auto rendererL = vtk_make_unique<vtkRenderer>(); // left 3D view
  auto rendererR = vtk_make_unique<vtkRenderer>(); // right 3D view
  auto rendererText =
      vtk_make_unique<vtkRenderer>(); // bottom text strip (same layer)
  auto renderWindow = vtk_make_unique<vtkRenderWindow>();
  auto interactor = vtk_make_unique<vtkRenderWindowInteractor>();
  interactor->SetInteractorStyle(inter.get());

  // Left viewport: original mesh
  auto mapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto actor = vtk_make_unique<vtkOpenGLActor>();
  actor->SetMapper(mapper.get());
  mapper->SetInputData(poly.get());
  rendererL->AddActor(actor.get());

  // Optional curve actor on left
  auto curve_poly = vtk_make_unique<vtkPolyData>();
  curve_poly->Initialize();
  auto cmapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto cactor = vtk_make_unique<vtkOpenGLActor>();
  cactor->SetMapper(cmapper.get());
  cmapper->SetInputData(curve_poly.get());
  cactor->GetProperty()->SetColor(1, 0.1, 0.1);
  rendererL->AddActor(cactor.get());

  // Right viewport: result mesh
  auto poly1 = vtk_make_unique<vtkPolyData>();
  poly1->Initialize();
  auto mapper1 = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto actor1 = vtk_make_unique<vtkOpenGLActor>();
  actor1->SetMapper(mapper1.get());
  mapper1->SetInputData(poly1.get());
  rendererR->AddActor(actor1.get());

  // Add renderers to the window (all in the default layer 0)
  renderWindow->SetInteractor(interactor.get());
  renderWindow->AddRenderer(rendererL.get());
  renderWindow->AddRenderer(rendererR.get());
  renderWindow->AddRenderer(rendererText.get());

  // Viewports: two side-by-side on top (88% height), text bar bottom (12%)
  rendererL->SetViewport(0.0, 0.12, 0.5, 1.0);
  rendererR->SetViewport(0.5, 0.12, 1.0, 1.0);
  rendererText->SetViewport(0.0, 0.0, 1.0, 0.12);
  rendererText->InteractiveOff();
  rendererText->EraseOn(); // each renderer clears its own viewport

  // Backgrounds
  rendererL->SetBackground(27.0 / 255.0, 43.0 / 255.0,
                           52.0 / 255.0); // dark blue/gray
  rendererR->SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0);
  rendererText->SetBackground(0.090, 0.143, 0.173); // darker tone

  // ---- Text: use normalized viewport coords so positions track resizes ----
  auto text0 = vtk_make_unique<vtkTextActor>();
  text0->SetInput("Isobands time per scroll: 0 ms");
  {
    auto *tp = text0->GetTextProperty();
    tp->SetFontSize(40);
    tp->SetColor(1.0, 1.0, 1.0);
    tp->SetJustificationToLeft();
    tp->SetVerticalJustificationToCentered();
  }
  text0->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  text0->SetPosition(0.03,
                     0.30); // ~3% from left, 30% up within the bottom strip
  rendererText->AddActor2D(text0.get());

  auto text1 = vtk_make_unique<vtkTextActor>();
  text1->SetInput("Press n to randomize the plane.");
  {
    auto *tp = text1->GetTextProperty();
    tp->SetFontSize(40);
    tp->SetColor(1.0, 1.0, 1.0);
    tp->SetJustificationToLeft();
    tp->SetVerticalJustificationToCentered();
  }
  text1->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  text1->SetPosition(0.03, 0.70);
  rendererText->AddActor2D(text1.get());

  auto textR = vtk_make_unique<vtkTextActor>();
  textR->SetInput("Hold shift and scroll.\n"
                  "Intersection curve with plane will move.\n"
                  "Powered by trueform.");
  {
    auto *tp = textR->GetTextProperty();
    tp->SetFontSize(40);
    tp->SetColor(1.0, 1.0, 1.0);
    tp->SetJustificationToRight(); // anchor at the right edge
    tp->SetVerticalJustificationToCentered();
    tp->SetLineSpacing(1.5);
  }
  textR->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  textR->SetPosition(0.97, 0.60); // ~3% from right, 60% up in the strip
  rendererText->AddActor2D(textR.get());

  // Interactor wiring + shared camera between 3D views
  inter->initialize(poly.get(), poly1.get(), curve_poly.get(), text0.get());
  rendererR->SetActiveCamera(rendererL->GetActiveCamera());

  rendererL->ResetCamera();
  rendererR->ResetCamera();
  renderWindow->Render();
  interactor->Start();
  return 0;
}
