#pragma once
namespace utils {
auto set_at(std::array<double, 16> &mat, tf::vector<float, 3> at) -> void {
  auto tr = tf::random_transformation(at);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      mat[i * 4 + j] = tr(i, j);
    }
  }
}

auto center_and_scale_p(tf::polygons_buffer<int, float, 3, 3> &poly) -> void {
  auto pts = poly.points();
  auto aabb = tf::aabb_from(tf::make_polygon(pts));
  auto center = aabb.center().as_vector();
  auto r = aabb.diagonal().length() / 2;
  tf::parallel_for_each(pts.as_vector_view(), [&](auto pt) {
    pt -= center;
    pt *= 10 / r;
  });
}
} // namespace utils
