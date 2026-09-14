#include <trueform/cpp/topology/async/connect_edges_to_paths.hpp>
#include <trueform/cpp/topology/connect_edges_to_paths.hpp>

#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  auto output = storage.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

} // namespace

int main() {
  const auto paths = tf::cpp::connect_edges_to_paths(
      make_array<std::int64_t>({0, 1, 1, 2}, {2, 2}));
  static_assert(std::is_same_v<
                std::decay_t<decltype(paths)>,
                tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>>);
  return paths.size() == 1 && paths.data().length() == 3 ? 0 : 1;
}
