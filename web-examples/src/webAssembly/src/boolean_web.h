#pragma once
#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"
#include "trueform/spatial/form.hpp"
#include "trueform/spatial/ray_cast.hpp"

#include "main.h"
#include "utils/utils.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"

#include <filesystem>
#include <string>
#include <string_view>

#include <emscripten/bind.h>
#include <emscripten/val.h>


class tf_bridge : public tf_bridge_interface {
public:
    tf_bridge() = default;
    ~tf_bridge() override = default;
public:
  auto compute_boolean() {
    auto form0 = tf::make_form(frames[0], trees[0], polys[0]->polygons()) //
                 | tf::tag(face_memberships[0])                         //
                 | tf::tag(manifold_edge_links[0]);

    auto form1 = tf::make_form(frames[1], trees[1],polys[1]->polygons()) //
                 | tf::tag(face_memberships[1])                         //
                 | tf::tag(manifold_edge_links[1]);
    return tf::make_boolean(form0, form1, tf::boolean_op::left_difference,
                            tf::return_curves);
  }

  auto get_actors() -> std::vector<std::unique_ptr<MeshObject>> & { return actors; }
};

class cursor_interactor : public cursor_interactor_interface {
public:
  cursor_interactor() {
    bridge = std::make_unique<tf_bridge>();
  };
  ~cursor_interactor() override = default;

public:
  std::unique_ptr<MeshObject> result_mesh = std::make_unique<MeshObject>();
  std::unique_ptr<MeshObject> curve_mesh = std::make_unique<MeshObject>();

private:
  auto compute_curves() {
    tf::tick();
    if(auto pB = dynamic_cast<tf_bridge*>(bridge.get())) {
      auto [res_mesh, labels, curves] = pB->compute_boolean();
      add_time(tf::tock());
      (void)labels;
      result_mesh->setPolydata(std::move(res_mesh));
      curve_mesh->setCurvesObject(std::move(curves));
    }
  }

  auto randomize_rotations() {
    for (std::unique_ptr<MeshObject>& actor : bridge->get_actors()) {
      tf::vector<double, 3> at{actor->matrix[3], actor->matrix[7],
                               actor->matrix[11]};
      auto tr = tf::random_transformation(at);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
          actor->matrix[i*4 + j] = tr(i, j);
      bridge->update_frame(actor.get());
    }
  }

public:
  bool OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) override {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [actor, point] = bridge->ray_hit(ray);
      if (actor) {
        make_moving_plane(point, cameraPosition, cameraFocalPoint);
        this->last_point = point;
      }
      selected_actor = actor;
      return true;
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - this->last_point;
      this->last_point = next_point;
      move_selected(selected_actor);
      compute_curves();
      return true;
    } else if (camera_mode) {
        return false;
    }
    return false;
  }

  bool OnKeyPress(std::string key) override {
    if (key == "n") {
      randomize_rotations();
      compute_curves();
      return true;
    } else {
      return false;
    }
  }
};

int run_main(std::string path) {
  std::cout << "Reading file: " << path << std::endl;
  auto poly = tf::read_stl<int>(path);
  std::cout << "run main 0: " << poly.size() << std::endl;
  auto poly2 = tf::read_stl<int>(path);
  if (!poly.size()) {
    std::cout << "Failed to read file" << std::endl;
    throw std::runtime_error("Failed to read file");
  }
  interactor = std::make_unique<cursor_interactor>();

  utils::center_and_scale_p(poly);
  auto actor = std::make_unique<MeshObject>();
  actor->polyObject = std::move(poly);
  utils::set_at(actor->matrix, {0 * 15.f, 0.f, 0.f});
  interactor->push_back(std::move(actor));

  utils::center_and_scale_p(poly2);
  auto actor2 = std::make_unique<MeshObject>();
  actor2->polyObject = std::move(poly2);
  utils::set_at(actor2->matrix, {1 * 15.f, 0.f, 0.f});
  interactor->push_back(std::move(actor2));
/*
  // Optional curve actor on left
  auto curve_poly = vtk_make_unique<vtkPolyData>();
  curve_poly->Initialize();
  auto cmapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto cactor = vtk_make_unique<vtkOpenGLActor>();
  cactor->SetMapper(cmapper.get());
  cmapper->SetInputData(curve_poly.get());
  cactor->GetProperty()->SetColor(1, 0.1, 0.1);
  rendererL->AddActor(cactor.get());
*/
  return 0;
}