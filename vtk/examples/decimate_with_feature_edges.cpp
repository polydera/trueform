/**
 * Feature edge preservation: decimation and isotropic remeshing.
 *
 * Top-left: original subdivided box.
 * Top-right: decimated with feature preservation.
 * Bottom-left: isotropic remesh (no quadric) with feature preservation.
 * Bottom-right: isotropic remesh (quadric) with feature preservation.
 */

#include <iostream>
#include <string>
#include <trueform/core/angle.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/remesh/decimate_config.hpp>
#include <trueform/remesh/remesh_config.hpp>
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/core/polydata.hpp>
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

static constexpr double edge_color[3] = {0.15, 0.15, 0.18};
static constexpr double bg[3] = {27.0 / 255, 43.0 / 255, 52.0 / 255};

auto make_mesh_actor(vtkPolyData *poly)
    -> vtkSmartPointer<vtkOpenGLActor> {
  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputData(poly);
  auto actor = vtkSmartPointer<vtkOpenGLActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(0.85, 0.85, 0.88);
  actor->GetProperty()->SetAmbient(0.2);
  actor->GetProperty()->SetDiffuse(0.8);
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetEdgeColor(edge_color[0], edge_color[1],
                                     edge_color[2]);
  actor->GetProperty()->SetLineWidth(1.0);
  return actor;
}

auto make_label(const std::string &text, int font_size = 14)
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
  auto box = tf::make_box_mesh(1.f, 1.f, 1.f, 10, 10, 10);
  auto n_original = box.faces_buffer().size();
  auto vtk_box = tf::vtk::make_vtk_polydata(box);

  vtkNew<tf::vtk::polydata> vtk_original;
  vtk_original->ShallowCopy(vtk_box);

  std::cout << "original: " << n_original << " faces\n";

  // Decimation with feature preservation
  tf::decimate_config<float> dec_cfg;
  dec_cfg.parallel = false;
  dec_cfg.feature_angle = tf::deg(30.f);
  dec_cfg.feature_weight = 100;
  dec_cfg.max_aspect_ratio = -1;

  auto vtk_dec = tf::vtk::decimated(vtk_original.Get(), 0.1f, dec_cfg);
  auto n_dec = vtk_dec->GetNumberOfPolys();
  std::cout << "decimated: " << n_dec << " faces\n";

  // Isotropic remesh without quadric
  tf::remesh_config<float> rem_cfg_no_q;
  rem_cfg_no_q.target_length = 0.8f;
  rem_cfg_no_q.use_quadric = false;
  rem_cfg_no_q.feature_angle = tf::deg(30.f);
  rem_cfg_no_q.feature_weight = 100;

  auto vtk_rem_no_q =
      tf::vtk::isotropic_remeshed(vtk_original.Get(), rem_cfg_no_q);
  auto n_rem_no_q = vtk_rem_no_q->GetNumberOfPolys();
  std::cout << "remesh (no quadric): " << n_rem_no_q << " faces\n";

  // Isotropic remesh with quadric
  tf::remesh_config<float> rem_cfg_q;
  rem_cfg_q.target_length = 0.8f;
  rem_cfg_q.use_quadric = true;
  rem_cfg_q.feature_angle = tf::deg(30.f);
  rem_cfg_q.feature_weight = 100;

  auto vtk_rem_q =
      tf::vtk::isotropic_remeshed(vtk_original.Get(), rem_cfg_q);
  auto n_rem_q = vtk_rem_q->GetNumberOfPolys();
  std::cout << "remesh (quadric): " << n_rem_q << " faces\n";

  // Layout: 2x2 grid
  vtkNew<vtkOpenGLRenderer> ren_orig;
  ren_orig->SetViewport(0.0, 0.5, 0.5, 1.0);
  ren_orig->SetBackground(bg[0], bg[1], bg[2]);

  vtkNew<vtkOpenGLRenderer> ren_dec;
  ren_dec->SetViewport(0.5, 0.5, 1.0, 1.0);
  ren_dec->SetBackground(bg[0], bg[1], bg[2]);

  vtkNew<vtkOpenGLRenderer> ren_rem_no_q;
  ren_rem_no_q->SetViewport(0.0, 0.0, 0.5, 0.5);
  ren_rem_no_q->SetBackground(bg[0], bg[1], bg[2]);

  vtkNew<vtkOpenGLRenderer> ren_rem_q;
  ren_rem_q->SetViewport(0.5, 0.0, 1.0, 0.5);
  ren_rem_q->SetBackground(bg[0], bg[1], bg[2]);

  ren_orig->AddActor(make_mesh_actor(vtk_original));
  ren_dec->AddActor(make_mesh_actor(vtk_dec));
  ren_rem_no_q->AddActor(make_mesh_actor(vtk_rem_no_q));
  ren_rem_q->AddActor(make_mesh_actor(vtk_rem_q));

  ren_orig->AddViewProp(
      make_label("Original: " + std::to_string(n_original) + " faces"));
  ren_dec->AddViewProp(
      make_label("Decimated: " + std::to_string(n_dec) + " faces"));
  ren_rem_no_q->AddViewProp(
      make_label("Remesh (no quadric): " + std::to_string(n_rem_no_q) +
                 " faces"));
  ren_rem_q->AddViewProp(
      make_label("Remesh (quadric): " + std::to_string(n_rem_q) + " faces"));

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(ren_orig);
  window->AddRenderer(ren_dec);
  window->AddRenderer(ren_rem_no_q);
  window->AddRenderer(ren_rem_q);
  window->SetSize(1600, 900);
  window->SetWindowName("Feature Edge Preservation");

  ren_dec->SetActiveCamera(ren_orig->GetActiveCamera());
  ren_rem_no_q->SetActiveCamera(ren_orig->GetActiveCamera());
  ren_rem_q->SetActiveCamera(ren_orig->GetActiveCamera());
  ren_orig->ResetCamera();

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);
  vtkNew<vtkInteractorStyleTrackballCamera> style;
  interactor->SetInteractorStyle(style);
  window->Render();
  interactor->Start();

  return 0;
}
