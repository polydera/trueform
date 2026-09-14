#include <trueform/cpp/topology/connect_edges_to_paths.hpp>
#include <trueform/cpp/topology/label_connected_components.hpp>

#include <cstdint>
#include <utility>

namespace {

auto edges() -> tf::cpp::nd_array<std::int32_t> {
  tf::buffer<std::int32_t> storage;
  storage.allocate(4);
  storage[0] = 0;
  storage[1] = 1;
  storage[2] = 1;
  storage[3] = 2;
  return tf::cpp::nd_array<std::int32_t>::from_buffer(std::move(storage),
                                                      {2, 2});
}

} // namespace

int main() {
  const auto paths = tf::cpp::connect_edges_to_paths(edges());
  const auto labels = tf::cpp::label_connected_components(edges());
  return static_cast<int>(paths.size()) + labels.n_components;
}
