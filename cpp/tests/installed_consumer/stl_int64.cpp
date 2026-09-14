#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/io/read_stl.hpp>
#include <trueform/cpp/io/write_stl.hpp>

#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>
#include <type_traits>

int main() {
  using index_type = std::int64_t;
  using storage_type = tf::polygons_buffer<index_type, float, 3, 3>;
  constexpr std::string_view source = "solid triangle\n"
                                      " facet normal 0 0 1\n"
                                      "  outer loop\n"
                                      "   vertex 0 0 0\n"
                                      "   vertex 1 0 0\n"
                                      "   vertex 0 1 0\n"
                                      "  endloop\n"
                                      " endfacet\n"
                                      "endsolid triangle\n";

  auto value = tf::cpp::read_stl<index_type>(
      reinterpret_cast<const std::int8_t *>(source.data()), source.size());
  static_assert(std::is_same_v<decltype(value), storage_type>);
  if (value.size() != 1 || value.points_buffer().size() != 3 ||
      value.faces_buffer().data_buffer().size() != 3) {
    std::cerr << "STL int64 parse mismatch: faces=" << value.size()
              << " points=" << value.points_buffer().size() << '\n';
    return 1;
  }
  for (const auto point_id : value.faces_buffer().data_buffer())
    if (point_id < index_type{0} || point_id >= index_type{3})
      return 1;

  const consumer::owned_mesh<index_type, float, 3> held{std::move(value)};
  const auto encoded =
      tf::cpp::write_stl<index_type, float, 3, 3>(held.mesh());
  const auto decoded =
      tf::cpp::read_stl<index_type>(encoded.raw_data(), encoded.length());
  return encoded.length() == 134 && decoded.size() == 1 &&
                 decoded.points_buffer().size() == 3
             ? 0
             : 2;
}
