#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/io/read_obj.hpp>
#include <trueform/cpp/io/write_obj.hpp>

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

int main() {
  using real_type = float;
  using index_type = std::int64_t;
  constexpr auto mixed = tf::dynamic_size;
  using storage_type = tf::polygons_buffer<index_type, real_type, 3, mixed>;
  constexpr std::string_view source = "v 0 0 0\n"
                                      "v 1 0 0\n"
                                      "v 1 1 0\n"
                                      "v 0 1 0\n"
                                      "v 2 0 0\n"
                                      "f 1 2 3\n"
                                      "f 1 3 4\n"
                                      "f 1 2 5 3\n";

  auto storage = tf::cpp::read_obj<index_type, real_type, mixed>(
      reinterpret_cast<const std::int8_t *>(source.data()), source.size());
  static_assert(std::is_same_v<decltype(storage), storage_type>);
  if (storage.size() != 3 ||
      storage.faces_buffer().offsets_buffer()[3] != index_type{10} ||
      storage.points_buffer().size() != 5)
    return 1;

  const consumer::owned_mesh<index_type, real_type, 3, mixed> mesh{
      std::move(storage)};
  const auto encoded =
      tf::cpp::write_obj<index_type, real_type, 3, mixed>(mesh.mesh());
  const auto decoded = tf::cpp::read_obj<index_type, real_type, mixed>(
      encoded.raw_data(), encoded.length());
  return decoded.size() == 3 &&
                 decoded.faces_buffer().offsets_buffer()[3] == index_type{10} &&
                 decoded.points_buffer().size() == 5
             ? 0
             : 2;
}
