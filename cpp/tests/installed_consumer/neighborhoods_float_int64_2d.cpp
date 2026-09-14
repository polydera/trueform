#include <trueform/cpp/topology/async/neighborhoods.hpp>
#include <trueform/cpp/topology/neighborhoods.hpp>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <utility>

namespace {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  std::copy(values.begin(), values.end(), storage.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

} // namespace

int main() {
  using index_type = std::int64_t;
  const auto connectivity =
      tf::cpp::offset_blocked_buffer<index_type, index_type>::create(
          make_array<index_type>({0, 2, 5, 8, 10}, {5}),
          make_array<index_type>({1, 2, 0, 2, 3, 0, 1, 3, 1, 2}, {10}));
  const auto points = make_array<float>({0, 0, 1, 0, 0.5F, 1, 1.5F, 1}, {4, 2});
  const auto neighborhoods =
      tf::cpp::neighborhoods<index_type, float, 2>(connectivity, points, 1.5F);
  const index_type expected_offsets[]{0, 2, 5, 8, 10};
  const index_type expected_data[]{1, 2, 0, 2, 3, 0, 1, 3, 1, 2};
  return neighborhoods.offsets().length() == 5 &&
                 neighborhoods.data().length() == 10 &&
                 std::equal(neighborhoods.offsets().begin(),
                            neighborhoods.offsets().end(),
                            std::begin(expected_offsets)) &&
                 std::equal(neighborhoods.data().begin(),
                            neighborhoods.data().end(),
                            std::begin(expected_data))
             ? 0
             : 1;
}
