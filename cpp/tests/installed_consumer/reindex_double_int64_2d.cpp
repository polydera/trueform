#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/reindex.hpp>

#include <cstdint>
#include <future>
#include <initializer_list>
#include <iostream>
#include <type_traits>

namespace {

using namespace trueform_installed_test;
using mesh_storage = tf::polygons_buffer<std::int64_t, double, 2, 3>;
using mixed_storage =
    tf::polygons_buffer<std::int64_t, double, 2, tf::dynamic_size>;
using edge_storage = tf::segments_buffer<std::int64_t, double, 2>;
using mesh_result = tf::cpp::reindexed_mesh_result<std::int64_t, double, 2>;

using tf::cpp::nd_array;

template <typename T>
auto has_values(const nd_array<T> &value, std::initializer_list<T> expected)
    -> bool {
  if (value.length() != expected.size())
    return false;
  auto actual = value.begin();
  for (const auto item : expected)
    if (*actual++ != item)
      return false;
  return true;
}

auto points() -> nd_array<double> {
  return make_array<double>(
      {0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 2.0, 0.0, 2.5, 1.0, 3.0, 0.0}, {6, 2});
}

auto faces() -> nd_array<std::int64_t> {
  return make_array<std::int64_t>({0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4}, {4, 3});
}

auto dynamic_faces()
    -> tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t> {
  return tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
      make_array<std::int64_t>({0, 3, 6, 9, 12}, {5}),
      make_array<std::int64_t>({0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4}, {12}));
}

template <typename Storage>
auto selected_faces_ok(const Storage &value) -> bool {
  return value.size() == 2 && value.points_buffer().size() == 6;
}

template <typename Storage>
auto selected_point_faces_ok(const Storage &value) -> bool {
  const auto indices = consumer::face_indices_of(value);
  const std::int64_t expected[]{0, 1, 2, 0, 2, 1};
  if (value.size() != 2 || value.points_buffer().size() != 3 ||
      indices.size() != 6)
    return false;
  for (std::size_t index = 0; index < indices.size(); ++index)
    if (indices[index] != expected[index])
      return false;
  return true;
}

auto check_v3_sync_and_async() -> bool {
  const auto fixed_faces = faces();
  const auto fixed_points = points();
  const auto ids = make_array<std::int64_t>({2, 0}, {2});
  const auto mask = make_array<std::int8_t>({1, 0, 1, 0}, {4});
  const auto reversed_duplicate_point_ids =
      make_array<std::int64_t>({5, 4, 3, 5, 4}, {5});
  const auto point_mask = make_array<std::int8_t>({0, 0, 0, 1, 1, 1}, {6});

  const auto by_ids = tf::cpp::reindexed_by_ids<std::int64_t, double, 2, 3>(
      fixed_faces, fixed_points, ids);
  const auto by_ids_maps =
      tf::cpp::reindexed_by_ids_with_maps<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, ids);
  const auto by_mask = tf::cpp::reindexed_by_mask<std::int64_t, double, 2, 3>(
      fixed_faces, fixed_points, mask);
  const auto by_mask_maps =
      tf::cpp::reindexed_by_mask_with_maps<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, mask);
  const auto by_point_ids =
      tf::cpp::reindexed_by_ids_on_points<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, reversed_duplicate_point_ids);
  const auto by_point_mask =
      tf::cpp::reindexed_by_mask_on_points<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, point_mask);

  auto pending_ids =
      tf::cpp::async::reindexed_by_ids<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, ids);
  auto pending_ids_maps =
      tf::cpp::async::reindexed_by_ids_with_maps<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, ids);
  auto pending_mask =
      tf::cpp::async::reindexed_by_mask<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, mask);
  auto pending_mask_maps =
      tf::cpp::async::reindexed_by_mask_with_maps<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, mask);
  auto pending_point_ids =
      tf::cpp::async::reindexed_by_ids_on_points<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, reversed_duplicate_point_ids);
  auto pending_point_mask =
      tf::cpp::async::reindexed_by_mask_on_points<std::int64_t, double, 2, 3>(
          fixed_faces, fixed_points, point_mask);

  static_assert(std::is_same_v<std::decay_t<decltype(by_ids)>, mesh_storage>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(by_ids_maps)>, mesh_result>);
  static_assert(
      std::is_same_v<decltype(pending_ids), std::future<mesh_storage>>);
  static_assert(
      std::is_same_v<decltype(pending_ids_maps), std::future<mesh_result>>);

  return selected_faces_ok(by_ids) && selected_faces_ok(by_ids_maps.mesh) &&
         has_values(by_ids_maps.face_map.kept_ids,
                    {std::int64_t{2}, std::int64_t{0}}) &&
         selected_faces_ok(by_mask) && selected_faces_ok(by_mask_maps.mesh) &&
         has_values(by_mask_maps.face_map.kept_ids,
                    {std::int64_t{0}, std::int64_t{2}}) &&
         selected_point_faces_ok(by_point_ids) &&
         selected_point_faces_ok(by_point_mask) &&
         selected_faces_ok(pending_ids.get()) &&
         selected_faces_ok(pending_ids_maps.get().mesh) &&
         selected_faces_ok(pending_mask.get()) &&
         selected_faces_ok(pending_mask_maps.get().mesh) &&
         selected_point_faces_ok(pending_point_ids.get()) &&
         selected_point_faces_ok(pending_point_mask.get());
}

auto check_dynamic_tuple_sync_and_async() -> bool {
  const auto dynamic = dynamic_faces();
  const auto tuple_points = points();
  const auto ids = make_array<std::int64_t>({2, 0}, {2});
  const auto mask = make_array<std::int8_t>({1, 0, 1, 0}, {4});
  const auto reversed_duplicate_point_ids =
      make_array<std::int64_t>({5, 4, 3, 5, 4}, {5});
  const auto point_mask = make_array<std::int8_t>({0, 0, 0, 1, 1, 1}, {6});

  const auto by_ids = tf::cpp::reindexed_by_ids<std::int64_t, double, 2>(
      dynamic, tuple_points, ids);
  const auto by_ids_maps =
      tf::cpp::reindexed_by_ids_with_maps<std::int64_t, double, 2>(
          dynamic, tuple_points, ids);
  const auto by_mask = tf::cpp::reindexed_by_mask<std::int64_t, double, 2>(
      dynamic, tuple_points, mask);
  const auto by_mask_maps =
      tf::cpp::reindexed_by_mask_with_maps<std::int64_t, double, 2>(
          dynamic, tuple_points, mask);
  const auto by_point_ids =
      tf::cpp::reindexed_by_ids_on_points<std::int64_t, double, 2>(
          dynamic, tuple_points, reversed_duplicate_point_ids);
  const auto by_point_mask =
      tf::cpp::reindexed_by_mask_on_points<std::int64_t, double, 2>(
          dynamic, tuple_points, point_mask);

  auto pending_ids = tf::cpp::async::reindexed_by_ids<std::int64_t, double, 2>(
      dynamic, tuple_points, ids);
  auto pending_ids_maps =
      tf::cpp::async::reindexed_by_ids_with_maps<std::int64_t, double, 2>(
          dynamic, tuple_points, ids);
  auto pending_mask =
      tf::cpp::async::reindexed_by_mask<std::int64_t, double, 2>(
          dynamic, tuple_points, mask);
  auto pending_mask_maps =
      tf::cpp::async::reindexed_by_mask_with_maps<std::int64_t, double, 2>(
          dynamic, tuple_points, mask);
  auto pending_point_ids =
      tf::cpp::async::reindexed_by_ids_on_points<std::int64_t, double, 2>(
          dynamic, tuple_points, reversed_duplicate_point_ids);
  auto pending_point_mask =
      tf::cpp::async::reindexed_by_mask_on_points<std::int64_t, double, 2>(
          dynamic, tuple_points, point_mask);

  static_assert(std::is_same_v<std::decay_t<decltype(by_ids)>, mixed_storage>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(by_ids_maps.mesh)>, mixed_storage>);
  static_assert(std::is_same_v<std::decay_t<decltype(by_mask)>, mixed_storage>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(by_point_ids)>, mixed_storage>);

  return selected_faces_ok(by_ids) && selected_faces_ok(by_ids_maps.mesh) &&
         selected_faces_ok(by_mask) && selected_faces_ok(by_mask_maps.mesh) &&
         selected_point_faces_ok(by_point_ids) &&
         selected_point_faces_ok(by_point_mask) &&
         selected_faces_ok(pending_ids.get()) &&
         selected_faces_ok(pending_ids_maps.get().mesh) &&
         selected_faces_ok(pending_mask.get()) &&
         selected_faces_ok(pending_mask_maps.get().mesh) &&
         selected_point_faces_ok(pending_point_ids.get()) &&
         selected_point_faces_ok(pending_point_mask.get());
}

auto check_reversed_duplicate_point_maps() -> bool {
  const consumer::owned_mesh<std::int64_t, double, 2> owned{
      polygons_of<std::int64_t, double, 2>(
          {0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4},
          {0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 2.0, 0.0, 2.5, 1.0, 3.0, 0.0})};
  const auto input = owned.mesh();
  const auto point_ids = make_array<std::int64_t>({5, 4, 3, 5, 4}, {5});
  const auto selected =
      tf::cpp::reindexed_by_ids_on_points_with_maps<std::int64_t, double, 2>(
          input, point_ids);
  auto pending =
      tf::cpp::async::reindexed_by_ids_on_points_with_maps<std::int64_t, double,
                                                           2>(input, point_ids);
  const auto reapplied = tf::cpp::reindexed<std::int64_t, double, 2>(
      input, selected.face_map, selected.point_map);
  auto pending_reapplied = tf::cpp::async::reindexed<std::int64_t, double, 2>(
      input, selected.face_map, selected.point_map);
  const auto async_selected = pending.get();
  return selected_point_faces_ok(selected.mesh) &&
         has_values(selected.face_map.kept_ids,
                    {std::int64_t{2}, std::int64_t{3}}) &&
         has_values(selected.point_map.kept_ids,
                    {std::int64_t{3}, std::int64_t{4}, std::int64_t{5}}) &&
         selected_point_faces_ok(async_selected.mesh) &&
         has_values(async_selected.point_map.kept_ids,
                    {std::int64_t{3}, std::int64_t{4}, std::int64_t{5}}) &&
         selected_point_faces_ok(reapplied) &&
         selected_point_faces_ok(pending_reapplied.get());
}

auto check_mixed_concatenation() -> bool {
  const consumer::owned_mesh<std::int32_t, float, 2> low_owned{
      polygons_of<std::int32_t, float, 2>(
          {0, 1, 2}, {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F})};
  const consumer::owned_mesh<std::int64_t, double, 2, tf::dynamic_size>
      high_owned{polygons_of<std::int64_t, double, 2>(
          {0, 4}, {0, 1, 2, 3}, {10.0, 0.0, 11.0, 0.0, 11.0, 1.0, 10.0, 1.0})};
  const auto low_mesh = low_owned.mesh();
  const auto high_mesh = high_owned.mesh();

  const auto forward = tf::cpp::concatenate_meshes(low_mesh, high_mesh);
  const auto reverse = tf::cpp::concatenate_meshes(high_mesh, low_mesh);
  auto pending_forward =
      tf::cpp::async::concatenate_meshes(low_mesh, high_mesh);
  auto pending_reverse =
      tf::cpp::async::concatenate_meshes(high_mesh, low_mesh);

  const consumer::owned_edge_mesh<std::int32_t, float, 2> low_edge_owned{
      segments_of<std::int32_t, float, 2>({0, 1}, {0.0F, 0.0F, 1.0F, 0.0F})};
  const consumer::owned_edge_mesh<std::int64_t, double, 2> high_edge_owned{
      segments_of<std::int64_t, double, 2>({0, 1}, {10.0, 0.0, 11.0, 0.0})};
  const auto low_edges = low_edge_owned.edge_mesh();
  const auto high_edges = high_edge_owned.edge_mesh();
  const auto edge_forward =
      tf::cpp::concatenate_edge_meshes(low_edges, high_edges);
  const auto edge_reverse =
      tf::cpp::concatenate_edge_meshes(high_edges, low_edges);

  // one triangle carrier beside one mixed carrier can only be stated mixed
  static_assert(std::is_same_v<std::decay_t<decltype(forward)>, mixed_storage>);
  static_assert(std::is_same_v<std::decay_t<decltype(reverse)>, mixed_storage>);
  static_assert(std::is_same_v<std::decay_t<decltype(pending_forward.get())>,
                               mixed_storage>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(edge_forward)>, edge_storage>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(edge_reverse)>, edge_storage>);

  return forward.size() == 2 && forward.points_buffer().size() == 7 &&
         consumer::face_indices_of(forward)[3] == std::int64_t{3} &&
         consumer::face_indices_of(reverse)[4] == std::int64_t{4} &&
         edge_forward.size() == 2 &&
         edge_forward.edges_buffer().data_buffer()[2] == std::int64_t{2} &&
         edge_reverse.size() == 2 &&
         edge_reverse.edges_buffer().data_buffer()[2] == std::int64_t{2};
}

auto check_typed_domain_split() -> bool {
  const consumer::owned_mesh<std::int64_t, double, 2, tf::dynamic_size> owned{
      polygons_of<std::int64_t, double, 2>(
          {0, 3, 6, 9, 12}, {0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4},
          {0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 2.0, 0.0, 2.5, 1.0, 3.0, 0.0})};
  const auto input = owned.mesh();
  const auto label_values =
      make_array<std::int64_t>({0, 2, 0, 2, 1, 2, 1, 2}, {4, 2});
  const tf::cpp::domain_labels_result<std::int64_t> labels(label_values, 2, -1);
  const auto split = tf::cpp::split_into_domains(input, labels);
  auto pending = tf::cpp::async::split_into_domains(input, labels);
  using result_type =
      tf::cpp::split_domains_result<std::int64_t, double, 2, tf::dynamic_size>;
  static_assert(std::is_same_v<std::decay_t<decltype(split)>, result_type>);
  static_assert(std::is_same_v<decltype(pending), std::future<result_type>>);
  const auto async_split = pending.get();
  return split.components.size() == 2 &&
         has_values(split.labels, {std::int64_t{0}, std::int64_t{1}}) &&
         split.components[0].size() == 2 && split.components[1].size() == 2 &&
         async_split.components.size() == 2 &&
         has_values(async_split.labels, {std::int64_t{0}, std::int64_t{1}});
}

} // namespace

int main() {
  const auto ok = check_v3_sync_and_async() &&
                  check_dynamic_tuple_sync_and_async() &&
                  check_reversed_duplicate_point_maps() &&
                  check_mixed_concatenation() && check_typed_domain_split();
  if (!ok)
    std::cerr << "double/int64/2D typed reindex archive verification failed\n";
  return ok ? 0 : 1;
}
