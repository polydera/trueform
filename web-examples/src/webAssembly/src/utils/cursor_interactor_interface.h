#pragma once

#include "utils/bridge_web.h"
#include <array>
#include <string>
#include <vector>
#include <memory>

// Forward declarations instead of including boolean_web.h
class tf_bridge_interface{
public:
  auto push_back(std::unique_ptr<mesh_object> mesh) -> void {
    map[mesh.get()] = actors.size();
    polys.push_back(&mesh->poly_object);
    frames.emplace_back();
    frames.back().fill(mesh->matrix.data());
    trees.emplace_back();
    trees.back().build(mesh->poly_object.polygons(), tf::config_tree(4, 4));
    face_memberships.emplace_back();
    face_memberships.back().build(mesh->poly_object.polygons());
    manifold_edge_links.emplace_back();
    manifold_edge_links.back().build(mesh->poly_object.faces(),
                                     face_memberships.back());
    actors.push_back(std::move(mesh));
  }

  auto get_actors() const -> const auto & { return actors; }

  auto ray_hit(tf::ray<float, 3> ray)
      -> std::pair<mesh_object *, tf::point<float, 3>> {
    tf::tree_ray_info<int, tf::ray_cast_info<float>> result;
    tf::ray_config<float> config{};
    mesh_object *picked = nullptr;

    for (const auto &[frame, poly, actor, tree] : tf::zip(frames, polys, actors, trees)) {
      auto form = tf::make_form(frame, tree, poly->polygons());
      auto res = tf::ray_cast(ray, form, config);
      if (res) {
        result = res;
        config.max_t = result.info.t;
        picked = actor.get();
      }
    }
    return std::make_pair(picked, ray.origin + result.info.t * ray.direction);
  }

  auto update_frame(mesh_object *actor) -> void {
    auto id = map[actor];
    frames[id].fill(actors[id]->matrix.data());
    actors[id]->matrix_updated = true;
  }

  auto get_actors() -> std::vector<std::unique_ptr<mesh_object>> & { return actors; }

protected:
  std::map<mesh_object *, int> map;
  std::vector<tf::polygons_buffer<int, float, 3, 3> *> polys;
  std::vector<std::unique_ptr<mesh_object>> actors;
  std::vector<tf::frame<double, 3>> frames;
  std::vector<tf::tree<int, float, 3>> trees;
  std::vector<tf::face_membership<int>> face_memberships;
  std::vector<tf::manifold_edge_link<int, 3>> manifold_edge_links;
};

class cursor_interactor_interface {
public:
	cursor_interactor_interface(std::unique_ptr<tf_bridge_interface> bridge) : bridge(std::move(bridge)) {};
    virtual ~cursor_interactor_interface() = default;

    std::unique_ptr<mesh_object> result_mesh = std::make_unique<mesh_object>();
    std::unique_ptr<mesh_object> curve_mesh = std::make_unique<mesh_object>();

protected:
    std::unique_ptr<tf_bridge_interface> bridge;
    int time_index = 0;

    tf::plane<float, 3> moving_plane;
    tf::point<float, 3> last_point;
    tf::vector<float, 3> dx;
    mesh_object *selected_actor = nullptr;
    bool selected_mode = false;
    bool camera_mode = false;

    auto add_time(std::vector<float> &times, float t) {
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
      m_time = time;
      return time;
    }

    auto make_moving_plane(tf::point<float, 3> origin, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) {
        auto normal = tf::make_unit_vector(tf::make_vector_view<3>(cameraFocalPoint.data()) - tf::make_vector_view<3>(cameraPosition.data()));
        moving_plane = tf::make_plane(normal, origin);
    }

    auto move_selected(mesh_object *selected_actor) {
        for (int i = 0; i < 3; ++i)
            selected_actor->matrix[i*4 + 3] += dx[i];
        bridge->update_frame(selected_actor);
    }
public:
    float m_time = 0.f;
    float m_pick_time = 0.f;
    std::size_t total_polygons = 0;

    auto get_actors() -> std::vector<std::unique_ptr<mesh_object>> & {
        return bridge->get_actors();
    }

    auto push_back(std::unique_ptr<mesh_object> mesh) -> void {
        bridge->push_back(std::move(mesh));
    }

    auto OnLeftButtonDown() -> bool {
        if (selected_actor) {
            selected_mode = true;
            return true;
        } else {
            camera_mode = true;
        }
        return false;
    }

    virtual auto OnLeftButtonUp() -> bool {
        if (selected_mode) {
            selected_mode = false;
            return true;
        } else if (camera_mode) {
            camera_mode = false;
        }
        return false;
    }

    virtual auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) -> bool {
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
            return true;
        } else if (camera_mode) {
            return false;
        }
        return false;
    }

    virtual auto OnKeyPress(std::string) -> bool {
        return false;
    }

    virtual auto OnMouseWheel(int, bool) -> bool {
        return false;
    }
};
