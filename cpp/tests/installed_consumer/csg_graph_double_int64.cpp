#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/csg/csg_created_points.hpp>
#include <trueform/cpp/csg/csg_domains.hpp>
#include <trueform/cpp/csg/csg_graph.hpp>
#include <trueform/cpp/csg/csg_intersection_curves.hpp>
#include <trueform/cpp/csg/csg_mesh.hpp>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
auto arrays_equal(const tf::cpp::nd_array<T> &first,
                  const tf::cpp::nd_array<T> &second) -> bool {
  if (first.raw_shape() != second.raw_shape() ||
      first.length() != second.length())
    return false;
  for (std::size_t index = 0; index < first.length(); ++index)
    if (first[index] != second[index])
      return false;
  return true;
}

template <typename First, typename Second>
auto buffers_equal(const First &first, const Second &second) -> bool {
  if (first.size() != second.size())
    return false;
  for (std::size_t index = 0; index < first.size(); ++index)
    if (first[index] != second[index])
      return false;
  return true;
}

} // namespace

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;
  using mesh_type = tf::cpp::mesh<index_type, real_type, 3>;
  using graph_type = tf::cpp::csg_graph<index_type, real_type>;

  static_assert(std::is_same_v<
                tf::curves_buffer<std::int32_t, real_type, 3>,
                tf::curves_buffer<tf::cpp::default_index_t, real_type, 3>>);
  static_assert(
      std::is_same_v<tf::cpp::csg_mesh_labeled_result<std::int32_t, real_type>,
                     tf::cpp::csg_mesh_labeled_result<tf::cpp::default_index_t,
                                                      real_type>>);
  static_assert(std::is_same_v<
                tf::cpp::csg_mesh_index_map_result<std::int32_t, real_type>,
                tf::cpp::csg_mesh_index_map_result<tf::cpp::default_index_t,
                                                   real_type>>);
  static_assert(
      std::is_same_v<
          tf::cpp::csg_domains_result<std::int32_t, real_type>,
          tf::cpp::csg_domains_result<tf::cpp::default_index_t, real_type>>);
  static_assert(std::is_same_v<
                tf::cpp::csg_domains_labeled_result<index_type, real_type>,
                tf::cpp::csg_domains_labeled_result<index_type, real_type>>);
  static_assert(std::is_same_v<
                tf::cpp::csg_domains_index_map_result<index_type, real_type>,
                tf::cpp::csg_domains_index_map_result<index_type, real_type>>);

  const auto first = negative_tetrahedron<index_type, real_type>();
  auto second = negative_tetrahedron<index_type, real_type>();
  // the storage is the caller's, so a move of it is stated on the cache
  auto &shifted = second.polygons.points_buffer().data_buffer();
  for (std::size_t point = 0; point + 2 < shifted.size(); point += 3) {
    shifted[point] += real_type{0.2};
    shifted[point + 1] += real_type{0.1};
    shifted[point + 2] += real_type{0.1};
  }
  second.cache.points_changed();

  const std::vector<mesh_type> inputs{first.mesh(), second.mesh()};
  auto graph = tf::cpp::make_csg_graph(inputs);
  static_assert(std::is_same_v<decltype(graph), graph_type>);

  const auto expression = tf::csg::op(0) | tf::csg::op(1);
  const auto config = tf::domain_config::exclude_outer_shell |
                      tf::domain_config::ignore_open_fragments;
  const auto full = tf::cpp::make_csg_mesh(graph);
  const auto full_labeled = tf::cpp::make_csg_mesh_with_labels(graph);
  const auto merged = tf::cpp::make_csg_mesh(graph, expression);
  const auto repeated = tf::cpp::make_csg_mesh(graph, expression);
  const auto labeled = tf::cpp::make_csg_mesh_with_labels(graph, expression);
  const auto mapped = tf::cpp::make_csg_mesh_with_index_map(graph, expression);
  const auto created = tf::cpp::csg_created_points(graph);
  const auto curves = tf::cpp::csg_intersection_curves(graph);
  const auto domains = tf::cpp::make_csg_domains(graph, config);
  const auto selected_domains =
      tf::cpp::make_csg_domains(graph, expression, config);
  const auto domain_labels =
      tf::cpp::make_csg_domains_with_labels(graph, config);
  const auto selected_labels =
      tf::cpp::make_csg_domains_with_labels(graph, expression, config);
  const auto domain_map =
      tf::cpp::make_csg_domains_with_index_map(graph, config);
  const auto selected_map =
      tf::cpp::make_csg_domains_with_index_map(graph, expression, config);

  static_assert(std::is_same_v<std::decay_t<decltype(labeled.tag_labels)>,
                               tf::cpp::nd_array<index_type>>);
  static_assert(std::is_same_v<std::decay_t<decltype(mapped.point_labels)>,
                               tf::cpp::nd_array<index_type>>);
  static_assert(std::is_same_v<std::decay_t<decltype(curves)>,
                               tf::curves_buffer<index_type, real_type, 3>>);

  const auto ok =
      full.size() > 0 && merged.size() > 0 &&
      buffers_equal(consumer::face_indices_of(merged),
                    consumer::face_indices_of(repeated)) &&
      buffers_equal(merged.points_buffer().data_buffer(),
                    repeated.points_buffer().data_buffer()) &&
      full_labeled.mesh.size() == full.size() &&
      full_labeled.tag_labels.length() == full.size() &&
      labeled.mesh.size() == merged.size() &&
      labeled.tag_labels.length() == merged.size() &&
      mapped.mesh.size() == merged.size() &&
      mapped.number_of_tags == index_type{2} &&
      mapped.point_f_offsets.length() == 3 && created.shape_at(1) == 3 &&
      created.shape_at(0) > 0 && curves.size() > 0 &&
      domains.ids.length() == domains.meshes.size() &&
      selected_domains.ids.length() == selected_domains.meshes.size() &&
      arrays_equal(domains.ids, domain_labels.ids) &&
      arrays_equal(domains.ids, domain_map.ids) &&
      arrays_equal(selected_domains.ids, selected_labels.ids) &&
      arrays_equal(selected_domains.ids, selected_map.ids) &&
      domain_map.inclusion.shape_at(1) == 2 &&
      selected_map.inclusion.shape_at(1) == 2;
  if (!ok)
    std::cerr
        << "double/int64 CSG graph or repeated result consumers mismatch\n";
  return ok ? 0 : 1;
}
