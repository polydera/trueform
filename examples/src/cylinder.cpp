#include "cylinder.hpp"

namespace examples {

auto make_tube(float radius, float height, int segments)
    -> tf::polygons_buffer<int, float, 3, 4> {
  tf::polygons_buffer<int, float, 3, 4> mesh;

  const float angle_step = 360.0f / static_cast<float>(segments);

  for (int i = 0; i < segments; ++i) {
    float angle0 = static_cast<float>(i) * angle_step;
    float angle1 = static_cast<float>(i + 1) * angle_step;

    float x0 = radius * tf::cos(tf::deg(angle0));
    float y0 = radius * tf::sin(tf::deg(angle0));
    float x1 = radius * tf::cos(tf::deg(angle1));
    float y1 = radius * tf::sin(tf::deg(angle1));

    int base = i * 4;
    // Four vertices per quad: bottom-current, bottom-next, top-next, top-current
    mesh.points_buffer().emplace_back(x0, y0, 0.0f);
    mesh.points_buffer().emplace_back(x1, y1, 0.0f);
    mesh.points_buffer().emplace_back(x1, y1, height);
    mesh.points_buffer().emplace_back(x0, y0, height);

    // Quad face (counter-clockwise for outward normal)
    mesh.faces_buffer().emplace_back(base, base + 1, base + 2, base + 3);
  }

  return mesh;
}

auto make_disk(float radius, float z, int segments)
    -> tf::polygons_buffer<int, float, 3, 3> {
  tf::polygons_buffer<int, float, 3, 3> mesh;

  const float angle_step = 360.0f / static_cast<float>(segments);

  for (int i = 0; i < segments; ++i) {
    float angle0 = static_cast<float>(i) * angle_step;
    float angle1 = static_cast<float>(i + 1) * angle_step;

    float x0 = radius * tf::cos(tf::deg(angle0));
    float y0 = radius * tf::sin(tf::deg(angle0));
    float x1 = radius * tf::cos(tf::deg(angle1));
    float y1 = radius * tf::sin(tf::deg(angle1));

    int base = i * 3;
    // Three vertices per triangle: center, current, next
    mesh.points_buffer().emplace_back(0.0f, 0.0f, z);
    mesh.points_buffer().emplace_back(x0, y0, z);
    mesh.points_buffer().emplace_back(x1, y1, z);

    // Triangle face (counter-clockwise for +Z normal)
    mesh.faces_buffer().emplace_back(base, base + 1, base + 2);
  }

  return mesh;
}

} // namespace examples
