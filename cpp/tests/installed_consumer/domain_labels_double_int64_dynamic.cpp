#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/topology/domain_labels.hpp>

#include <cstdint>
#include <iostream>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;
  using result_type = tf::cpp::domain_labels_result<index_type>;

  const consumer::owned_mesh<index_type, real_type, 3, tf::dynamic_size> owned{
      polygons_of<index_type, real_type, 3>(
          {0, 3, 6, 9, 12}, {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2},
          {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1})};
  const auto value = owned.mesh();

  const auto included =
      tf::cpp::make_domain_labels<index_type, real_type, 3>(value);
  const auto excluded = tf::cpp::make_domain_labels<index_type, real_type, 3>(
      value, tf::domain_config::exclude_outer_shell);
  static_assert(std::is_same_v<std::decay_t<decltype(included)>, result_type>);

  auto included_labels_valid = true;
  for (const auto label : included.labels())
    included_labels_valid &= label >= included.valid_label_begin() &&
                             label < included.valid_label_end();
  auto excluded_labels_valid = true;
  for (const auto label : excluded.labels())
    excluded_labels_valid &= label >= excluded.valid_label_begin() &&
                             label <= excluded.sentinel_label();

  const auto ok =
      included.number_of_faces() == 4 &&
      included.labels().raw_shape() == tf::small_vector<int, 3>{4, 2} &&
      included.number_of_domains() == 2 && included.has_outer_shell_domain() &&
      included_labels_valid && excluded.number_of_faces() == 4 &&
      excluded.number_of_domains() == 1 && !excluded.has_outer_shell_domain() &&
      excluded.outer_shell_label() == excluded.sentinel_label() &&
      excluded_labels_valid;
  if (!ok)
    std::cerr << "double/int64 dynamic domain labels mismatch\n";
  return ok ? 0 : 1;
}
