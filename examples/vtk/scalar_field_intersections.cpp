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
  tf::scalar_field_intersections<int, float, 3> sfi;
  tf::buffer<float> scalars;
  std::vector<float> times;
  int time_index = 0;
  vtkTextActor *text;
  vtkPolyData *poly;
  vtkPolyData *curve_poly;
  float distance = 0;

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
    std::snprintf(buffer, sizeof(buffer),
                  "Intersection curve time per scroll: %.1f ms", time);
    text->SetInput(buffer);
  }

  auto compute_curves() {
    auto polygons = get_triangles(poly);
    std::vector<float> cutvalues;
    for(int i=-10;i<10;++i)
      cutvalues.push_back(distance + i*0.5);
    tf::tick();
    sfi.build_many(polygons, scalars,
                   cutvalues);
    /*sfi.build(polygons, scalars, distance);*/
    auto edges = tf::make_intersection_edges(sfi);
    add_time(tf::tock());
    auto tmp_poly =
        segments_to_lines(tf::make_segments(edges, sfi.intersection_points()));
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
  }

public:
  auto initialize(vtkPolyData *_poly, vtkPolyData *_cpoly, vtkTextActor *_text)
      -> void {
    poly = _poly;
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
  if (argc < 2) {
    std::cerr << "Usage: program <input.stl|obj>  ...\n";
    return 1;
  } else if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                           std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input.stl|obj>  ...\n";
    return 1;
  }

  auto poly = read_mesh(argv[1]);
  center_and_scale(poly.get());

  auto inter = vtk_make_unique<cursor_interactor>();
  auto renderer = vtk_make_unique<vtkRenderer>();
  auto render_window = vtk_make_unique<vtkRenderWindow>();
  auto interactor = vtk_make_unique<vtkRenderWindowInteractor>();
  interactor->SetInteractorStyle(inter.get());
  auto mapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto actor = vtk_make_unique<vtkOpenGLActor>();
  actor->SetMapper(mapper.get());
  mapper->SetInputData(poly.get());
  renderer->AddActor(actor.get());

  auto curve_poly = vtk_make_unique<vtkPolyData>();
  curve_poly->Initialize();
  auto cmapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto cactor = vtk_make_unique<vtkOpenGLActor>();
  cactor->SetMapper(cmapper.get());
  cmapper->SetInputData(curve_poly.get());
  cactor->GetProperty()->SetColor(1, 0.1, 0.1);
  renderer->AddActor(cactor.get());

  renderer->SetBackground(27. / 255, 43. / 255, 52. / 255);
  render_window->SetInteractor(interactor.get());
  render_window->AddRenderer(renderer.get());
  render_window->Render();

  renderer->SetViewport(0.0, 0.12, 1.0, 1.0); // top 80%
  auto renderer_text = vtk_make_unique<vtkRenderer>();
  renderer_text->SetBackground(0.090, 0.143, 0.173); // darker tone
  renderer_text->SetViewport(0.0, 0.0, 1.0, 0.12);   // bottom 20%
  renderer_text->InteractiveOff();
  render_window->AddRenderer(renderer_text.get());

  auto text0 = vtk_make_unique<vtkTextActor>();
  text0->SetInput("Intersection curve time per scroll: 0 ms");
  auto textprop0 = text0->GetTextProperty();
  textprop0->SetFontSize(40);
  textprop0->SetColor(1.0, 1.0, 1.0);
  textprop0->SetJustificationToLeft();
  textprop0->SetVerticalJustificationToCentered();
  text0->SetDisplayPosition(40, 55);
  renderer_text->AddActor2D(text0.get());

  auto text1 = vtk_make_unique<vtkTextActor>();
  text1->SetInput("Press n to randomize the plane.");
  auto textprop1 = text1->GetTextProperty();
  textprop1->SetFontSize(40);
  textprop1->SetColor(1.0, 1.0, 1.0);
  textprop1->SetJustificationToLeft();
  textprop1->SetVerticalJustificationToCentered();
  text1->SetDisplayPosition(40, 115);
  renderer_text->AddActor2D(text1.get());

  auto text3 = vtk_make_unique<vtkTextActor>();
  text3->SetInput("Hold shift and scroll.\n"
                  "Intersection curve with plane will move.\n"
                  "Powered by trueform.");
  auto textprop3 = text3->GetTextProperty();
  textprop3->SetFontSize(40);
  textprop3->SetColor(1.0, 1.0, 1.0);
  textprop3->SetJustificationToRight();
  textprop3->SetVerticalJustificationToCentered();
  textprop3->SetLineSpacing(1.5);
  text3->SetDisplayPosition(renderer->GetSize()[0] - 40, 120);
  auto aligner = vtk_make_unique<RightAlignTextUpdater>(render_window.get(),
                                                        text3.get(), 40, 120);
  renderer_text->AddActor2D(text3.get());

  inter->initialize(poly.get(), curve_poly.get(), text0.get());

  render_window->Render();
  interactor->Start();
  return 0;
}
