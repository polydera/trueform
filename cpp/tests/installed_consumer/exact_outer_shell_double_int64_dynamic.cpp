#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/csg/outer_shell.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;
  constexpr auto mixed = tf::dynamic_size;

  const consumer::owned_mesh<index_type, real_type, 3, mixed> source{
      polygons_of<index_type, real_type, 3>(
          {0, 4, 7, 10, 13, 16},
          {0, 3, 2, 1, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4},
          {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1})};

  const auto shell = tf::cpp::outer_shell<index_type, real_type>(source.mesh());
  // an uncut face is emitted verbatim, so the shell states the arity its
  // operand did
  static_assert(
      std::is_same_v<std::decay_t<decltype(shell)>,
                     tf::polygons_buffer<index_type, real_type, 3, mixed>>);

  if (shell.size() == 0 || shell.points_buffer().size() == 0)
    return 2;

  const auto offsets = shell.faces_buffer().offsets_buffer();
  if (offsets.size() != shell.size() + 1)
    return 3;
  for (std::size_t face = 0; face + 1 < offsets.size(); ++face)
    if (offsets[face + 1] <= offsets[face])
      return 4;
  if (offsets[offsets.size() - 1] !=
      static_cast<index_type>(shell.faces_buffer().data_buffer().size()))
    return 5;

  return 0;
}
