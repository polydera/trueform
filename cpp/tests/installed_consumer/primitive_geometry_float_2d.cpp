#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/geometry/area.hpp>
#include <trueform/cpp/geometry/mean_edge_length.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

using storage_type = tf::buffer<float>;

auto triangle() -> tf::cpp::primitive<float, 2> {
  storage_type storage;
  storage.allocate(6);
  const float values[]{0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 3.0F};
  for (std::size_t index = 0; index < 6; ++index)
    storage[index] = values[index];
  auto points =
      tf::cpp::nd_array<float>::from_buffer(std::move(storage), {3, 2});
  return tf::cpp::primitive<float, 2>(tf::cpp::primitive_kind::triangle,
                                      std::move(points));
}

} // namespace

int main() {
  const auto value = triangle();
  const auto area = tf::cpp::area(value);
  return area.is_scalar() && area.scalar() == 6.0F &&
                 tf::cpp::mean_edge_length(value) == 4.0F
             ? 0
             : 1;
}
