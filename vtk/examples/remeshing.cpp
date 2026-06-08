/**
 * Remesh pipeline example with VTK.
 *
 * Left: original mesh. Right top: decimated. Right bottom: isotropic remeshed.
 * Edge wireframe on right panels.
 */

#include <chrono>
#include <iostream>
#include <string>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/functions.hpp>

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

static const std::string data_dir = TRUEFORM_DATA_DIR "/benchmarks/data/";
static constexpr double edge_color[3] = {0.15, 0.15, 0.18};
static constexpr double bg[3] = {27.0 / 255, 43.0 / 255, 52.0 / 255};

auto make_mesh_actor(vtkPolyData *poly, bool edges = false)
    -> vtkSmartPointer<vtkOpenGLActor> {
  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputData(poly);
  auto actor = vtkSmartPointer<vtkOpenGLActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(0.85, 0.85, 0.88);
  actor->GetProperty()->SetAmbient(0.2);
  actor->GetProperty()->SetDiffuse(0.8);
  if (edges) {
    actor->GetProperty()->EdgeVisibilityOn();
    actor->GetProperty()->SetEdgeColor(edge_color[0], edge_color[1],
                                       edge_color[2]);
    actor->GetProperty()->SetLineWidth(1.0);
  }
  return actor;
}

auto make_label(const std::string &text, int font_size = 18)
    -> vtkSmartPointer<vtkTextActor> {
  auto label = vtkSmartPointer<vtkTextActor>::New();
  label->SetInput(text.c_str());
  label->GetTextProperty()->SetFontSize(font_size);
  label->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
  label->GetTextProperty()->SetBold(true);
  label->SetPosition(10, 10);
  return label;
}

int main() {
  std::string path = data_dir + "dragon-500k.stl";
  auto original = tf::vtk::read_stl(path);
  auto n_original = original->GetNumberOfPolys();
  float target = 0.05f;

  std::cout << "original: " << n_original << " faces\n";

  // Decimate
  auto t0 = std::chrono::high_resolution_clock::now();
  auto vtk_dec = tf::vtk::decimated(original, target);
  auto t1 = std::chrono::high_resolution_clock::now();
  double t_dec = std::chrono::duration<double>(t1 - t0).count();
  auto n_dec = vtk_dec->GetNumberOfPolys();
  std::cout << "decimated: " << n_dec << " faces (" << t_dec * 1000 << " ms)\n";

  // Isotropic remesh to mean edge length of decimated result
  float mel = tf::vtk::mean_edge_length(vtk_dec);
  std::cout << "isotropic remesh (target length " << mel << ") ...\n";
  t0 = std::chrono::high_resolution_clock::now();
  tf::isotropic_remesh_config<float> rem_config{mel};
  rem_config.use_quadric = true;
  auto vtk_rem = tf::vtk::isotropic_remeshed(vtk_dec, rem_config);
  t1 = std::chrono::high_resolution_clock::now();
  double t_rem = std::chrono::duration<double>(t1 - t0).count();
  auto n_rem = vtk_rem->GetNumberOfPolys();
  std::cout << "remeshed: " << n_rem << " faces (" << t_rem * 1000 << " ms)\n";

  // Layout: original left, decimated top-right, remeshed bottom-right,
  //         text strip at bottom
  constexpr double text_h = 0.10;
  constexpr double half = 0.5;
  double mid_y = text_h + (1.0 - text_h) / 2.0;

  vtkNew<vtkOpenGLRenderer> ren_orig;
  ren_orig->SetViewport(0.0, text_h, half, 1.0);
  ren_orig->SetBackground(bg[0], bg[1], bg[2]);

  vtkNew<vtkOpenGLRenderer> ren_dec;
  ren_dec->SetViewport(half, mid_y, 1.0, 1.0);
  ren_dec->SetBackground(bg[0], bg[1], bg[2]);

  vtkNew<vtkOpenGLRenderer> ren_rem;
  ren_rem->SetViewport(half, text_h, 1.0, mid_y);
  ren_rem->SetBackground(bg[0], bg[1], bg[2]);

  vtkNew<vtkOpenGLRenderer> ren_text;
  ren_text->SetViewport(0.0, 0.0, 1.0, text_h);
  ren_text->SetBackground(0.090, 0.143, 0.173);
  ren_text->InteractiveOff();

  // Actors
  ren_orig->AddActor(make_mesh_actor(original));
  ren_dec->AddActor(make_mesh_actor(vtk_dec, true));
  ren_rem->AddActor(make_mesh_actor(vtk_rem, true));

  // Labels
  ren_orig->AddViewProp(
      make_label("Original: " + std::to_string(n_original) + " faces"));
  ren_dec->AddViewProp(make_label(
      "Decimated: " + std::to_string(n_dec) + " faces ("
      + std::to_string(int(target * 100)) + "%)", 14));
  ren_rem->AddViewProp(
      make_label("Remeshed: " + std::to_string(n_rem) + " faces", 14));

  // Text strip
  auto fmt = [](double t) {
    return t < 1.0 ? std::to_string(int(t * 1000)) + " ms"
                   : std::to_string(int(t)) + " s";
  };
  ren_text->AddViewProp(make_label(
      "Decimate " + fmt(t_dec) + "  |  Remesh " + fmt(t_rem) + "  |  "
      + std::to_string(n_original) + " -> " + std::to_string(n_dec) + " -> "
      + std::to_string(n_rem) + " faces", 28));

  // Window
  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(ren_orig);
  window->AddRenderer(ren_dec);
  window->AddRenderer(ren_rem);
  window->AddRenderer(ren_text);
  window->SetSize(1600, 800);
  window->SetWindowName("Remesh Pipeline");

  // Share camera
  ren_dec->SetActiveCamera(ren_orig->GetActiveCamera());
  ren_rem->SetActiveCamera(ren_orig->GetActiveCamera());
  ren_orig->ResetCamera();

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);
  vtkNew<vtkInteractorStyleTrackballCamera> style;
  interactor->SetInteractorStyle(style);
  window->Render();
  interactor->Start();

  return 0;
}
