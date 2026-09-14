#include "axis_shard_test_utils.hpp"
#include <trueform/cpp/core/common_index.hpp>
#include <trueform/cpp/core/index_map.hpp>
#include <trueform/cpp/core/index_type.hpp>

#include <trueform/cpp/core/build_face_membership.hpp>
#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/core/offset_blocked_buffer.hpp>
#include <trueform/cpp/csg/async/outer_shell.hpp>
#include <trueform/cpp/csg/csg_graph.hpp>
#include <trueform/cpp/csg/csg_mesh.hpp>
#include <trueform/cpp/csg/outer_shell.hpp>
#include <trueform/cpp/geometry/area.hpp>
#include <trueform/cpp/geometry/async/area.hpp>
#include <trueform/cpp/geometry/async/chamfer_error.hpp>
#include <trueform/cpp/geometry/async/fit_icp.hpp>
#include <trueform/cpp/geometry/async/fit_knn.hpp>
#include <trueform/cpp/geometry/async/fit_obb.hpp>
#include <trueform/cpp/geometry/async/fit_rigid.hpp>
#include <trueform/cpp/geometry/async/mean_edge_length.hpp>
#include <trueform/cpp/geometry/async/normals.hpp>
#include <trueform/cpp/geometry/async/point_normals.hpp>
#include <trueform/cpp/geometry/async/symmetric_chamfer_error.hpp>
#include <trueform/cpp/geometry/chamfer_error.hpp>
#include <trueform/cpp/geometry/fit_icp.hpp>
#include <trueform/cpp/geometry/fit_knn.hpp>
#include <trueform/cpp/geometry/fit_obb.hpp>
#include <trueform/cpp/geometry/fit_rigid.hpp>
#include <trueform/cpp/geometry/make_box_mesh.hpp>
#include <trueform/cpp/geometry/mean_edge_length.hpp>
#include <trueform/cpp/geometry/normals.hpp>
#include <trueform/cpp/geometry/point_normals.hpp>
#include <trueform/cpp/geometry/symmetric_chamfer_error.hpp>
#include <trueform/cpp/io/read_obj.hpp>
#include <trueform/cpp/io/read_stl.hpp>
#include <trueform/cpp/io/write_obj.hpp>
#include <trueform/cpp/io/write_stl.hpp>
#include <trueform/cpp/spatial/async/closest_metric_point_pair.hpp>
#include <trueform/cpp/spatial/async/distance.hpp>
#include <trueform/cpp/spatial/async/intersects.hpp>
#include <trueform/cpp/spatial/async/ray_cast.hpp>
#include <trueform/cpp/spatial/async/transformed.hpp>
#include <trueform/cpp/spatial/closest_metric_point_pair.hpp>
#include <trueform/cpp/spatial/distance.hpp>
#include <trueform/cpp/spatial/intersects.hpp>
#include <trueform/cpp/spatial/primitive.hpp>
#include <trueform/cpp/spatial/ray_cast.hpp>
#include <trueform/cpp/spatial/transformed.hpp>
#include <trueform/cpp/topology.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

template <typename> struct function_argument;

template <typename Result, typename Argument>
struct function_argument<Result (*)(Argument)> {
  using type = std::remove_reference_t<Argument>;
};

template <typename T> using array_storage_t = tf::buffer<T>;

template <typename T>
using array_shape_t = std::decay_t<
    decltype(std::declval<const tf::cpp::nd_array<T> &>().raw_shape())>;

template <typename T>
auto make_array(std::initializer_list<T> values, array_shape_t<T> shape)
    -> tf::cpp::nd_array<T> {
  array_storage_t<T> storage;
  storage.allocate(values.size());
  auto output = storage.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

template <typename Real, std::size_t Dims>
auto measurement_triangle() -> tf::cpp::primitive<Real, Dims> {
  auto points = [&] {
    if constexpr (Dims == 2)
      return make_array<Real>({0, 0, 4, 0, 0, 3}, {3, 2});
    else
      return make_array<Real>({0, 0, 0, 4, 0, 0, 0, 3, 0}, {3, 3});
  }();
  return tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::triangle,
                                        std::move(points));
}

template <typename Real, std::size_t Dims>
auto measurement_triangle_batch() -> tf::cpp::primitive<Real, Dims> {
  auto points = [&] {
    if constexpr (Dims == 2)
      return make_array<Real>({0, 0, 4, 0, 0, 3, 0, 0, 8, 0, 0, 6}, {2, 3, 2});
    else
      return make_array<Real>(
          {0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 8, 0, 0, 0, 6, 0}, {2, 3, 3});
  }();
  return tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::triangle,
                                        std::move(points));
}

template <typename Real, std::size_t Dims>
auto check_primitive_measurement_symbols() -> bool {
  const auto triangle = measurement_triangle<Real, Dims>();
  const auto triangles = measurement_triangle_batch<Real, Dims>();

  const auto sync_area = tf::cpp::area(triangle);
  const auto sync_batch_area = tf::cpp::area(triangles);
  auto pending_area = tf::cpp::async::area(triangle);
  auto pending_batch_area = tf::cpp::async::area(triangles);
  const auto sync_mean = tf::cpp::mean_edge_length(triangle);
  const auto sync_batch_mean = tf::cpp::mean_edge_length(triangles);
  auto pending_mean = tf::cpp::async::mean_edge_length(triangle);
  auto pending_batch_mean = tf::cpp::async::mean_edge_length(triangles);

  static_assert(
      std::is_same_v<decltype(sync_area), const tf::cpp::area_result<Real>>);
  static_assert(std::is_same_v<decltype(pending_area),
                               std::future<tf::cpp::area_result<Real>>>);
  static_assert(std::is_same_v<decltype(pending_mean), std::future<Real>>);

  const auto async_area = pending_area.get();
  const auto async_batch_area = pending_batch_area.get();
  const auto async_mean = pending_mean.get();
  const auto async_batch_mean = pending_batch_mean.get();
  const auto sync_batch = sync_batch_area.batch();
  const auto async_batch = async_batch_area.batch();

  return sync_area.is_scalar() && sync_area.scalar() == Real{6} &&
         async_area.is_scalar() && async_area.scalar() == Real{6} &&
         sync_batch_area.is_batch() && sync_batch.ndim() == 1 &&
         sync_batch.shape_at(0) == 2 && sync_batch[0] == Real{6} &&
         sync_batch[1] == Real{24} && async_batch_area.is_batch() &&
         async_batch.ndim() == 1 && async_batch.shape_at(0) == 2 &&
         async_batch[0] == Real{6} && async_batch[1] == Real{24} &&
         sync_mean == Real{4} && async_mean == Real{4} &&
         sync_batch_mean == Real{6} && async_batch_mean == Real{6};
}

template <typename Real> auto check_primitive_normal_symbols() -> bool {
  const auto triangle = measurement_triangle<Real, 3>();
  const auto triangles = measurement_triangle_batch<Real, 3>();
  const auto sync_normal = tf::cpp::normals(triangle);
  const auto sync_normals = tf::cpp::normals(triangles);
  auto pending_normal = tf::cpp::async::normals(triangle);
  auto pending_normals = tf::cpp::async::normals(triangles);

  static_assert(
      std::is_same_v<decltype(sync_normal), const tf::cpp::nd_array<Real>>);
  static_assert(std::is_same_v<decltype(pending_normal),
                               std::future<tf::cpp::nd_array<Real>>>);

  const auto async_normal = pending_normal.get();
  const auto async_normals = pending_normals.get();
  const auto has_single_shape = [](const auto &value) {
    return value.ndim() == 1 && value.shape_at(0) == 3 && value[0] == Real{0} &&
           value[1] == Real{0} && value[2] == Real{1};
  };
  const auto has_batch_shape = [](const auto &value) {
    if (value.ndim() != 2 || value.shape_at(0) != 2 || value.shape_at(1) != 3)
      return false;
    for (std::size_t index = 0; index < 6; ++index) {
      const auto expected = index % 3 == 2 ? Real{1} : Real{0};
      if (value[index] != expected)
        return false;
    }
    return true;
  };
  return has_single_shape(sync_normal) && has_single_shape(async_normal) &&
         has_batch_shape(sync_normals) && has_batch_shape(async_normals);
}

template <typename Real, std::size_t Dims>
auto make_translation() -> tf::cpp::nd_array<Real> {
  constexpr auto side = Dims + 1;
  array_storage_t<Real> storage;
  storage.allocate(side * side);
  for (std::size_t row = 0; row < side; ++row)
    for (std::size_t column = 0; column < side; ++column)
      storage[row * side + column] = row == column ? Real{1} : Real{0};
  storage[Dims] = Real{5};
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(storage), {static_cast<int>(side), static_cast<int>(side)});
}

template <typename Index, typename Real, std::size_t Dims>
auto check_mesh_carrier() -> bool {
  using namespace trueform_installed_test;
  const auto storage = [] {
    if constexpr (Dims == 2)
      return polygons_of<Index, Real, Dims>({0, 1, 2}, {0, 0, 2, 0, 0, 3});
    else
      return polygons_of<Index, Real, Dims>({0, 1, 2},
                                            {0, 0, 0, 2, 0, 0, 0, 3, 4});
  }();
  consumer::owned_mesh<Index, Real, Dims> owned{storage};
  const auto value = owned.mesh();
  tf::cpp::build_tree(value);
  tf::cpp::build_face_membership(value);
  const auto membership = value.face_membership();
  // a second assembly over the same cache is answered, not rebuilt: the cache
  // remembers what the geometry determines
  const auto again = owned.mesh();
  return value.number_of_faces() == 1 && value.number_of_points() == 3 &&
         consumer::face_indices_of(owned.polygons)[2] == Index{2} &&
         owned.cache.is_tree_fresh(value.geometry()) &&
         owned.cache.is_face_membership_fresh(value.geometry()) &&
         membership.size() == 3 && membership[2][0] == Index{0} &&
         again.face_membership().size() == 3 &&
         owned.cache.face_membership_build_count() == 1 &&
         owned.cache.tree_build_count() == 1;
}

template <typename Real, std::size_t Dims>
auto check_point_cloud_carrier() -> bool {
  using namespace trueform_installed_test;
  const auto storage = [] {
    if constexpr (Dims == 2)
      return points_of<Real, Dims>({0, 0, 2, 3});
    else
      return points_of<Real, Dims>({0, 0, 0, 2, 3, 4});
  }();
  consumer::owned_point_cloud<Real, Dims> owned{storage};
  const auto value = owned.point_cloud();
  tf::cpp::build_tree(value);
  const auto again = owned.point_cloud();
  return value.number_of_points() == 2 &&
         owned.cache.is_tree_fresh(value.geometry()) &&
         again.tree().ids().size() == 2 && owned.cache.tree_build_count() == 1;
}

template <typename Real, std::size_t Dims>
auto distance_point(Real offset) -> tf::cpp::primitive<Real, Dims> {
  if constexpr (Dims == 2)
    return tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::point,
                                          make_array<Real>({0, offset}, {2}));
  else
    return tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::point, make_array<Real>({0, 0, offset}, {3}));
}

template <typename Index, typename Real, std::size_t Dims>
auto distance_mesh(Real offset = Real{})
    -> consumer::owned_mesh<Index, Real, Dims> {
  using namespace trueform_installed_test;
  if constexpr (Dims == 2)
    return {polygons_of<Index, Real, Dims>(
        {0, 1, 2}, {-1, -1 + offset, 1, -1 + offset, 0, 1 + offset})};
  else
    return {polygons_of<Index, Real, Dims>(
        {0, 1, 2}, {-1, -1, offset, 1, -1, offset, 0, 1, offset})};
}

template <typename Real, std::size_t Dims>
auto check_primitive_distance_shard() -> bool {
  const auto first = distance_point<Real, Dims>(Real{});
  const auto second = distance_point<Real, Dims>(Real{2});
  return tf::cpp::distance(first, second).scalar() == Real{2} &&
         tf::cpp::async::distance2(first, second).get().scalar() == Real{4};
}

template <std::size_t Dims>
auto check_mixed_primitive_distance_shard() -> bool {
  const auto first = distance_point<float, Dims>(0.0F);
  const auto second = distance_point<double, Dims>(2.0);
  return tf::cpp::distance(first, second).scalar() == 2.0 &&
         tf::cpp::async::distance2(second, first).get().scalar() == 4.0;
}

template <typename Real, std::size_t Dims>
auto check_int64_distance_shard() -> bool {
  const auto owned = distance_mesh<std::int64_t, Real, Dims>();
  const auto mesh = owned.mesh();
  const auto query = distance_point<Real, Dims>(Real{2});
  const auto expected = Dims == 2 ? Real{1} : Real{2};
  return tf::cpp::distance(mesh, query).scalar() == expected &&
         tf::cpp::async::distance2(mesh, query).get().scalar() ==
             expected * expected;
}

template <std::size_t Dims> auto check_mixed_int64_distance_shard() -> bool {
  const auto first_owned = distance_mesh<std::int64_t, float, Dims>();
  const auto second_owned = distance_mesh<std::int64_t, double, Dims>(4.0);
  const auto first = first_owned.mesh();
  const auto second = second_owned.mesh();
  auto sync_rejected = false;
  auto async_rejected = false;
  try {
    static_cast<void>(tf::cpp::distance(first, second));
  } catch (const std::invalid_argument &) {
    sync_rejected = true;
  }
  try {
    static_cast<void>(tf::cpp::async::distance2(second, first).get());
  } catch (const std::invalid_argument &) {
    async_rejected = true;
  }
  return sync_rejected && async_rejected;
}

auto check_supplemental_distance_shards() -> bool {
  return check_primitive_distance_shard<float, 2>() &&
         check_primitive_distance_shard<double, 2>() &&
         check_mixed_primitive_distance_shard<2>() &&
         check_int64_distance_shard<float, 2>() &&
         check_int64_distance_shard<double, 2>() &&
         check_mixed_int64_distance_shard<2>() &&
         check_int64_distance_shard<float, 3>() &&
         check_int64_distance_shard<double, 3>() &&
         check_mixed_int64_distance_shard<3>();
}

template <typename Index> auto check_uniform_offset_blocks() -> bool {
  const auto uniform = make_array<Index>({0, 1, 2, 3, 4, 5}, {2, 3});
  const auto member =
      tf::cpp::offset_blocked_buffer<Index, Index>::from_uniform(uniform);
  const auto facade = tf::cpp::as_offset_blocked(uniform);
  return member.size() == 2 && member.offsets()[2] == Index{6} &&
         member.get(1)[0] == Index{3} && facade.size() == 2 &&
         facade.offsets()[1] == Index{3} && facade.data()[5] == Index{5};
}

auto check_connected_component_symbols() -> bool {
  auto dense32 = make_array<std::int32_t>({1, 0, -1, -1}, {2, 2});
  auto dense64 = make_array<std::int64_t>({1, 0, -1, -1}, {2, 2});
  auto variable32 =
      tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
          make_array<std::int32_t>({0, 1, 2}, {3}),
          make_array<std::int32_t>({1, 0}, {2}));
  auto variable64 =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
          make_array<std::int64_t>({0, 1, 2}, {3}),
          make_array<std::int64_t>({1, 0}, {2}));

  const auto dense32_default = tf::cpp::label_connected_components(dense32);
  const auto dense32_hint = tf::cpp::label_connected_components(dense32, 1000);
  const auto variable32_default =
      tf::cpp::label_connected_components(variable32);
  const auto variable32_hint =
      tf::cpp::label_connected_components(variable32, 1000);
  const auto dense64_default = tf::cpp::label_connected_components(dense64);
  const auto dense64_hint = tf::cpp::label_connected_components(dense64, 1000);
  const auto variable64_default =
      tf::cpp::label_connected_components(variable64);
  const auto variable64_hint =
      tf::cpp::label_connected_components(variable64, 1000);

  auto async_dense32_default =
      tf::cpp::async::label_connected_components(dense32);
  auto async_dense32_hint =
      tf::cpp::async::label_connected_components(dense32, 1000);
  auto async_variable32_default =
      tf::cpp::async::label_connected_components(variable32);
  auto async_variable32_hint =
      tf::cpp::async::label_connected_components(variable32, 1000);
  auto async_dense64_default =
      tf::cpp::async::label_connected_components(dense64);
  auto async_dense64_hint =
      tf::cpp::async::label_connected_components(dense64, 1000);
  auto async_variable64_default =
      tf::cpp::async::label_connected_components(variable64);
  auto async_variable64_hint =
      tf::cpp::async::label_connected_components(variable64, 1000);

  using wide_result = tf::cpp::connected_components_result<std::int64_t>;
  static_assert(std::is_same_v<decltype(dense32_default),
                               const tf::cpp::connected_components_result<>>);
  static_assert(std::is_same_v<decltype(dense64_default), const wide_result>);
  static_assert(
      std::is_same_v<decltype(async_dense64_hint), std::future<wide_result>>);

  return dense32_default.n_components == 1 && dense32_hint.n_components == 1 &&
         variable32_default.n_components == 1 &&
         variable32_hint.n_components == 1 &&
         dense64_default.n_components == 1 && dense64_hint.n_components == 1 &&
         variable64_default.n_components == 1 &&
         variable64_hint.n_components == 1 &&
         async_dense32_default.get().n_components == 1 &&
         async_dense32_hint.get().n_components == 1 &&
         async_variable32_default.get().n_components == 1 &&
         async_variable32_hint.get().n_components == 1 &&
         async_dense64_default.get().n_components == 1 &&
         async_dense64_hint.get().n_components == 1 &&
         async_variable64_default.get().n_components == 1 &&
         async_variable64_hint.get().n_components == 1;
}

template <typename Real, std::size_t Dims>
auto check_transformed_registration_symbols() -> bool {
  const auto storage = [] {
    if constexpr (Dims == 2)
      return trueform_installed_test::points_of<Real, Dims>(
          {0, 0, 1, 0, 0, 1, 1, 1});
    else
      return trueform_installed_test::points_of<Real, Dims>(
          {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1});
  }();
  const consumer::owned_point_cloud<Real, Dims> source_owned{storage};
  const consumer::owned_point_cloud<Real, Dims> target_owned{storage};
  const auto source = source_owned.point_cloud();
  const auto target = target_owned.point_cloud();
  tf::cpp::fit_knn_options<Real> options;
  options.k = 3;
  options.sigma = Real{0.25};
  const auto fit = tf::cpp::fit_knn(source, target, options);
  tf::cpp::fit_icp_options<Real> icp_options;
  icp_options.max_iterations = 0;
  icp_options.n_samples = 0;
  icp_options.k = 0;
  const auto icp = tf::cpp::fit_icp(source, target, icp_options);
  const auto rigid = tf::cpp::fit_rigid(source, target);
  const auto obb =
      tf::cpp::fit_obb(source, target, tf::cpp::fit_obb_options{0});
  const auto symmetric = tf::cpp::symmetric_chamfer_error(source, target);
  const auto directed = tf::cpp::chamfer_error(source, target);

  auto matrix = make_translation<Real, Dims>();
  auto point = tf::cpp::primitive<Real, Dims>(
      tf::cpp::primitive_kind::point, Dims == 2
                                          ? make_array<Real>({0, 0}, {2})
                                          : make_array<Real>({0, 0, 0}, {3}));
  const auto transformed = tf::cpp::transformed(point, matrix);
  const auto async_transformed =
      tf::cpp::async::transformed(point, matrix).get();
  const auto async_fit = tf::cpp::async::fit_knn(source, target, options).get();

  return fit.raw_shape() ==
             tf::small_vector<int, 3>{static_cast<int>(Dims + 1),
                                      static_cast<int>(Dims + 1)} &&
         async_fit.raw_shape() == fit.raw_shape() &&
         icp.raw_shape() == fit.raw_shape() &&
         rigid.raw_shape() == fit.raw_shape() &&
         obb.raw_shape() == fit.raw_shape() && symmetric == Real{0} &&
         directed == Real{0} && transformed.data()[Dims - 1] == Real{0} &&
         async_transformed.data()[0] == Real{5};
}

template <typename Real0, typename Real1>
auto check_2d_intersects_symbols() -> bool {
  const auto point = tf::cpp::primitive<Real0, 2>(
      tf::cpp::primitive_kind::point, make_array<Real0>({0, 0}, {2}));
  const auto box = tf::cpp::primitive<Real1, 2>(
      tf::cpp::primitive_kind::aabb, make_array<Real1>({-1, -1, 1, 1}, {2, 2}));
  return tf::cpp::intersects(point, box).scalar() &&
         tf::cpp::async::intersects(point, box).get().scalar();
}

template <typename Real0, typename Real1>
auto check_2d_closest_metric_point_pair_symbols() -> bool {
  using result_real = std::common_type_t<Real0, Real1>;
  using result_type = std::variant<
      tf::cpp::closest_metric_point_pair_result<result_real>,
      tf::cpp::closest_metric_point_pair_batch_result<result_real>>;
  const auto point = tf::cpp::primitive<Real0, 2>(
      tf::cpp::primitive_kind::point, make_array<Real0>({0, 2}, {2}));
  const auto segment =
      tf::cpp::primitive<Real1, 2>(tf::cpp::primitive_kind::segment,
                                   make_array<Real1>({1, 0, 3, 0}, {2, 2}));

  const auto sync_result = tf::cpp::closest_metric_point_pair(point, segment);
  auto pending_result =
      tf::cpp::async::closest_metric_point_pair(point, segment);
  static_assert(std::is_same_v<decltype(sync_result), const result_type>);
  static_assert(
      std::is_same_v<decltype(pending_result), std::future<result_type>>);

  const auto is_exact_pair = [](const result_type &value) {
    const auto *result =
        std::get_if<tf::cpp::closest_metric_point_pair_result<result_real>>(
            &value);
    return result && result->point0.ndim() == 1 &&
           result->point0.shape_at(0) == 2 &&
           result->point0[0] == result_real{0} &&
           result->point0[1] == result_real{2} && result->point1.ndim() == 1 &&
           result->point1.shape_at(0) == 2 &&
           result->point1[0] == result_real{1} &&
           result->point1[1] == result_real{0} &&
           result->distance2 == result_real{5};
  };
  return is_exact_pair(sync_result) && is_exact_pair(pending_result.get());
}

template <typename RayReal, typename TargetReal>
auto check_2d_ray_cast_symbols() -> bool {
  using result_real = std::common_type_t<RayReal, TargetReal>;
  const auto ray = tf::cpp::primitive<RayReal, 2>(
      tf::cpp::primitive_kind::ray, make_array<RayReal>({0, 0, 1, 0}, {2, 2}));
  const auto segment = tf::cpp::primitive<TargetReal, 2>(
      tf::cpp::primitive_kind::segment,
      make_array<TargetReal>({3, -1, 3, 1}, {2, 2}));

  const auto sync_result = tf::cpp::ray_cast(ray, segment);
  auto pending_result = tf::cpp::async::ray_cast(ray, segment);
  static_assert(
      std::is_same_v<decltype(sync_result),
                     const tf::cpp::ray_cast_primitive_result<result_real>>);
  static_assert(std::is_same_v<
                decltype(pending_result),
                std::future<tf::cpp::ray_cast_primitive_result<result_real>>>);

  const auto async_result = pending_result.get();
  const auto is_exact_hit = [](const auto &result) {
    return result.is_scalar() && result.scalar().hit &&
           result.scalar().t == result_real{3} &&
           result.scalar().element_id == -1;
  };
  return is_exact_hit(sync_result) && is_exact_hit(async_result);
}

template <typename Real> auto check_path_writers() -> bool {
  const consumer::owned_mesh<tf::cpp::default_index_t, Real, 3> owned{
      trueform_installed_test::polygons_of<tf::cpp::default_index_t, Real, 3>(
          {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  const auto value = owned.mesh();
  const auto stem = std::filesystem::temp_directory_path() /
                    (std::string("trueform-installed-writer-") +
                     std::to_string(sizeof(Real)));
  const auto obj = std::filesystem::path(stem.string() + ".obj");
  const auto stl = std::filesystem::path(stem.string() + ".stl");
  std::error_code error;
  std::filesystem::remove(obj, error);
  std::filesystem::remove(stl, error);
  const auto obj_ok = tf::cpp::write_obj(value, stem);
  const auto stl_ok = tf::cpp::write_stl(value, stem);
  const auto result = obj_ok && stl_ok && std::filesystem::file_size(obj) > 0 &&
                      std::filesystem::file_size(stl) == 134;
  std::filesystem::remove(obj, error);
  std::filesystem::remove(stl, error);
  return result;
}

auto check_generalized_io_symbols() -> bool {
  using index_type = std::int64_t;
  constexpr std::string_view obj_source = "v 0 0 0\n"
                                          "v 1 0 0\n"
                                          "v 1 1 0\n"
                                          "v 0 1 0\n"
                                          "f 1 2 3\n"
                                          "f 1 3 4 2\n";
  constexpr auto mixed = tf::dynamic_size;
  const consumer::owned_mesh<index_type, float, 3, mixed> dynamic_obj{
      tf::cpp::read_obj<index_type, float, mixed>(
          reinterpret_cast<const std::int8_t *>(obj_source.data()),
          obj_source.size())};
  const auto obj_bytes =
      tf::cpp::write_obj<index_type, float, 3, mixed>(dynamic_obj.mesh());
  const auto obj_roundtrip = tf::cpp::read_obj<index_type, float, mixed>(
      obj_bytes.raw_data(), obj_bytes.length());
  if (obj_roundtrip.size() != 2 ||
      obj_roundtrip.faces_buffer().offsets_buffer()[2] != index_type{7})
    return false;

  constexpr std::string_view stl_source = "solid triangle\n"
                                          " facet normal 0 0 1\n"
                                          "  outer loop\n"
                                          "   vertex 0 0 0\n"
                                          "   vertex 1 0 0\n"
                                          "   vertex 0 1 0\n"
                                          "  endloop\n"
                                          " endfacet\n"
                                          "endsolid triangle\n";
  const consumer::owned_mesh<index_type, float, 3> stl{
      tf::cpp::read_stl<index_type>(
          reinterpret_cast<const std::int8_t *>(stl_source.data()),
          stl_source.size())};
  const auto stl_bytes =
      tf::cpp::write_stl<index_type, float, 3, 3>(stl.mesh());
  return stl.polygons.size() == 1 && stl.polygons.points_buffer().size() == 3 &&
         stl_bytes.length() == 134;
}

auto check_mesh_normal_symbols() -> bool {
  using index_type = std::int64_t;
  const consumer::owned_mesh<index_type, double, 3> owned{
      trueform_installed_test::polygons_of<index_type, double, 3>(
          {0, 1, 2, 1, 3, 2}, {0, 0, 0, 1, 0, 0, 0.5, 1, 0, 1.5, 1, 0})};
  const auto value = owned.mesh();
  const auto faces = tf::cpp::normals<index_type, double, 3>(value);
  const auto points = tf::cpp::point_normals<index_type, double, 3>(value);
  return faces.raw_shape() == tf::small_vector<int, 3>{2, 3} &&
         points.raw_shape() == tf::small_vector<int, 3>{4, 3} &&
         faces[2] == 1.0 && faces[5] == 1.0 && points[2] == 1.0 &&
         points[11] == 1.0;
}

template <typename Index, typename Real, std::size_t Dims>
auto check_analysis_mesh() -> bool {
  const auto owned =
      trueform_installed_test::open_triangle<Index, Real, Dims>();
  const auto value = owned.mesh();
  return !tf::cpp::is_closed<Index, Real, Dims>(value) &&
         tf::cpp::is_open<Index, Real, Dims>(value) &&
         tf::cpp::is_manifold<Index, Real, Dims>(value) &&
         !tf::cpp::is_non_manifold<Index, Real, Dims>(value) &&
         tf::cpp::non_manifold_edges<Index, Real, Dims>(value).length() == 0;
}

auto check_typed_topology_link_symbols() -> bool {
  using index_type = std::int64_t;
  const auto faces = make_array<index_type>({0, 1, 2, 1, 3, 2}, {2, 3});
  const auto membership = tf::cpp::cell_membership(faces, index_type{4});
  const auto manifold = tf::cpp::manifold_edge_link(faces, membership);
  const auto links = tf::cpp::face_link(faces, membership);
  return manifold.raw_shape() == tf::small_vector<int, 3>{2, 3} &&
         manifold[1] == index_type{1} && manifold[5] == index_type{0} &&
         links.offsets().length() == 3 && links.data().length() == 2 &&
         links.data()[0] == index_type{1} && links.data()[1] == index_type{0};
}

template <typename Real, std::size_t Dims>
auto check_primitive_carrier() -> bool {
  auto data = [&] {
    if constexpr (Dims == 2)
      return make_array<Real>({0, 0, 3, 4}, {2, 2});
    else
      return make_array<Real>({0, 0, 0, 3, 4, 5}, {2, 3});
  }();
  auto value = tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::point,
                                              std::move(data));
  const auto second = value.at(1);
  const auto first_slice = value.slice(0, 1);
  const auto shallow = value.shallow_copy();
  const auto deep = value.deep_copy();
  auto plane_behavior_is_valid = true;
  if constexpr (Dims == 2) {
    try {
      static_cast<void>(tf::cpp::primitive<Real, Dims>(
          tf::cpp::primitive_kind::plane, make_array<Real>({0, 1, 0}, {3})));
      plane_behavior_is_valid = false;
    } catch (const std::invalid_argument &) {
    }
  } else {
    const auto plane = tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::plane, make_array<Real>({0, 0, 1, 0}, {4}));
    plane_behavior_is_valid = plane.element_stride() == 4;
  }
  return value.kind() == tf::cpp::primitive_kind::point && value.is_batch() &&
         value.count() == 2 && value.element_stride() == Dims &&
         value.data().shape_at(1) == static_cast<int>(Dims) &&
         !second.is_batch() && second.data()[0] == Real{3} &&
         first_slice.is_batch() && first_slice.count() == 1 &&
         shallow.data().raw_data() == value.data().raw_data() &&
         deep.data().raw_data() != value.data().raw_data() &&
         plane_behavior_is_valid;
}

} // namespace

int main() {
  using real = double;

  static_assert(std::is_same_v<tf::cpp::mesh<tf::cpp::default_index_t, float>,
                               tf::cpp::mesh<std::int32_t, float, 3, 3>>);
  static_assert(std::is_same_v<tf::cpp::mesh<tf::cpp::default_index_t, double>,
                               tf::cpp::mesh<std::int32_t, double, 3, 3>>);
  static_assert(std::is_same_v<tf::cpp::point_cloud<float>,
                               tf::cpp::point_cloud<float, 3>>);
  static_assert(std::is_same_v<tf::cpp::point_cloud<double>,
                               tf::cpp::point_cloud<double, 3>>);
  static_assert(
      std::is_same_v<tf::cpp::primitive<float>, tf::cpp::primitive<float, 3>>);
  static_assert(std::is_same_v<tf::cpp::primitive<double>,
                               tf::cpp::primitive<double, 3>>);

  if (!check_uniform_offset_blocks<std::int32_t>() ||
      !check_uniform_offset_blocks<std::int64_t>()) {
    std::cerr << "installed-package uniform offset-block symbols failed\n";
    return 11;
  }
  if (!check_path_writers<float>() || !check_path_writers<double>()) {
    std::cerr << "installed-package path writer symbols failed\n";
    return 12;
  }
  if (!check_generalized_io_symbols()) {
    std::cerr << "installed-package generalized IO symbols failed\n";
    return 19;
  }
  if (!check_mesh_normal_symbols()) {
    std::cerr << "installed-package mesh normal symbols failed\n";
    return 20;
  }
  if (!check_analysis_mesh<std::int64_t, float, 2>() ||
      !check_analysis_mesh<std::int64_t, double, 3>()) {
    std::cerr << "installed-package typed topology analysis symbols failed\n";
    return 21;
  }
  if (!check_typed_topology_link_symbols()) {
    std::cerr << "installed-package typed topology link symbols failed\n";
    return 22;
  }
  if (!check_connected_component_symbols()) {
    std::cerr << "installed-package connected-component symbols failed\n";
    return 13;
  }
  if (!check_transformed_registration_symbols<float, 2>() ||
      !check_transformed_registration_symbols<float, 3>() ||
      !check_transformed_registration_symbols<double, 2>() ||
      !check_transformed_registration_symbols<double, 3>()) {
    std::cerr << "installed-package transformed/registration symbols failed\n";
    return 14;
  }
  if (!check_2d_intersects_symbols<float, float>() ||
      !check_2d_intersects_symbols<float, double>() ||
      !check_2d_intersects_symbols<double, float>() ||
      !check_2d_intersects_symbols<double, double>()) {
    std::cerr << "installed-package 2D intersection symbols failed\n";
    return 15;
  }
  if (!check_2d_ray_cast_symbols<float, float>() ||
      !check_2d_ray_cast_symbols<float, double>() ||
      !check_2d_ray_cast_symbols<double, float>() ||
      !check_2d_ray_cast_symbols<double, double>()) {
    std::cerr << "installed-package 2D ray-cast symbols failed\n";
    return 17;
  }
  if (!check_2d_closest_metric_point_pair_symbols<float, float>() ||
      !check_2d_closest_metric_point_pair_symbols<float, double>() ||
      !check_2d_closest_metric_point_pair_symbols<double, float>() ||
      !check_2d_closest_metric_point_pair_symbols<double, double>()) {
    std::cerr
        << "installed-package 2D closest-metric-point-pair symbols failed\n";
    return 18;
  }
  if (!check_primitive_measurement_symbols<float, 2>() ||
      !check_primitive_measurement_symbols<float, 3>() ||
      !check_primitive_measurement_symbols<double, 2>() ||
      !check_primitive_measurement_symbols<double, 3>() ||
      !check_primitive_normal_symbols<float>() ||
      !check_primitive_normal_symbols<double>()) {
    std::cerr << "installed-package primitive geometry symbols failed\n";
    return 16;
  }

  auto mesh_matrix_ok = true;
  mesh_matrix_ok &= check_mesh_carrier<std::int32_t, float, 2>();
  mesh_matrix_ok &= check_mesh_carrier<std::int32_t, float, 3>();
  mesh_matrix_ok &= check_mesh_carrier<std::int64_t, float, 2>();
  mesh_matrix_ok &= check_mesh_carrier<std::int64_t, float, 3>();
  mesh_matrix_ok &= check_mesh_carrier<std::int32_t, double, 2>();
  mesh_matrix_ok &= check_mesh_carrier<std::int32_t, double, 3>();
  mesh_matrix_ok &= check_mesh_carrier<std::int64_t, double, 2>();
  mesh_matrix_ok &= check_mesh_carrier<std::int64_t, double, 3>();
  if (!mesh_matrix_ok) {
    std::cerr << "installed-package mesh carrier matrix symbols failed\n";
    return 10;
  }

  auto point_cloud_matrix_ok = true;
  point_cloud_matrix_ok &= check_point_cloud_carrier<float, 2>();
  point_cloud_matrix_ok &= check_point_cloud_carrier<float, 3>();
  point_cloud_matrix_ok &= check_point_cloud_carrier<double, 2>();
  point_cloud_matrix_ok &= check_point_cloud_carrier<double, 3>();
  if (!point_cloud_matrix_ok) {
    std::cerr
        << "installed-package point-cloud carrier matrix symbols failed\n";
    return 7;
  }

  auto primitive_matrix_ok = true;
  primitive_matrix_ok &= check_primitive_carrier<float, 2>();
  primitive_matrix_ok &= check_primitive_carrier<float, 3>();
  primitive_matrix_ok &= check_primitive_carrier<double, 2>();
  primitive_matrix_ok &= check_primitive_carrier<double, 3>();
  if (!primitive_matrix_ok) {
    std::cerr << "installed-package primitive carrier matrix symbols failed\n";
    return 8;
  }

  if (!check_supplemental_distance_shards()) {
    std::cerr << "installed-package supplemental distance shards failed\n";
    return 11;
  }

  const consumer::owned_mesh<tf::cpp::default_index_t, real, 3> first{
      tf::cpp::make_box_mesh(real{2}, real{2}, real{2})};
  consumer::owned_mesh<tf::cpp::default_index_t, real, 3> second{
      tf::cpp::make_box_mesh(real{2}, real{2}, real{2})};
  // the storage is the caller's: it writes through it and then says what moved
  auto &second_points = second.polygons.points_buffer().data_buffer();
  for (std::size_t point = 0; point < second_points.size(); point += 3)
    second_points[point] += real{1};
  second.cache.points_changed();

  const std::vector<tf::cpp::mesh<tf::cpp::default_index_t, real, 3, 3>> inputs{
      first.mesh(), second.mesh()};
  auto graph = tf::cpp::make_csg_graph(inputs);
  auto merged = tf::cpp::make_csg_mesh(graph, tf::csg::op(0) | tf::csg::op(1));
  if (merged.size() == 0) {
    std::cerr << "installed-package synchronous CSG call failed\n";
    return 1;
  }

  const consumer::owned_mesh<tf::cpp::default_index_t, real, 3> shell_source{
      tf::cpp::make_box_mesh(real{2}, real{2}, real{2})};
  auto pending_shell = tf::cpp::async::outer_shell(shell_source.mesh());
  static_assert(
      std::is_same_v<decltype(pending_shell),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     real, 3, 3>>>);
  auto shell = pending_shell.get();
  if (shell.size() == 0) {
    std::cerr << "installed-package default-future outer-shell call failed\n";
    return 2;
  }

  static_assert(std::is_same_v<tf::cpp::default_index_t, std::int32_t>);
  static_assert(
      std::is_same_v<tf::cpp::index_map<>, tf::cpp::index_map<std::int32_t>>);
  static_assert(
      std::is_same_v<tf::cpp::common_index_t<std::int32_t, std::int64_t>,
                     std::int64_t>);
  array_storage_t<std::int32_t> triangle_storage;
  triangle_storage.allocate(3);
  triangle_storage[0] = 0;
  triangle_storage[1] = 1;
  triangle_storage[2] = 2;
  auto fixed_faces = tf::cpp::nd_array<std::int32_t>::from_buffer(
      std::move(triangle_storage), {1, 3});
  if (fixed_faces.shape_at(0) != 1 || fixed_faces.shape_at(1) != 3) {
    std::cerr
        << "installed-package int32 fixed face connectivity call failed\n";
    return 4;
  }

  const consumer::owned_mesh<tf::cpp::default_index_t, double, 3,
                             tf::dynamic_size>
      dynamic_owned{
          trueform_installed_test::polygons_of<tf::cpp::default_index_t, double,
                                               3>(
              {0, 3, 7}, {0, 1, 2, 1, 3, 4, 2},
              {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0})};
  const auto dynamic_mesh = dynamic_owned.mesh();
  const auto read_indices = [&] {
    std::size_t count = 0;
    for (const auto face : dynamic_mesh.faces())
      count += face.size();
    return count;
  }();
  tf::cpp::build_tree(dynamic_mesh);
  tf::cpp::build_face_membership(dynamic_mesh);
  if (dynamic_mesh.number_of_faces() != 2 || read_indices != 7 ||
      !dynamic_owned.cache.is_tree_fresh(dynamic_mesh.geometry()) ||
      !dynamic_owned.cache.is_face_membership_fresh(dynamic_mesh.geometry())) {
    std::cerr << "installed-package dynamic mesh carrier call failed\n";
    return 9;
  }

  array_storage_t<std::int64_t> offset_storage;
  offset_storage.allocate(2);
  offset_storage[0] = 0;
  offset_storage[1] = 1;
  array_storage_t<std::int64_t> identity_storage;
  identity_storage.allocate(1);
  identity_storage[0] =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  auto index_blocks =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
          tf::cpp::nd_array<std::int64_t>::from_buffer(
              std::move(offset_storage)),
          tf::cpp::nd_array<std::int64_t>::from_buffer(
              std::move(identity_storage)));
  if (!index_blocks.is_valid() || index_blocks.size() != 1 ||
      index_blocks.get(0)[0] <= std::numeric_limits<std::int32_t>::max()) {
    std::cerr << "installed-package int64 index storage call failed\n";
    return 3;
  }

  using wide_index_map_type = tf::cpp::index_map<std::int64_t>;
  using wide_index_map_buffer_type = typename function_argument<
      decltype(&wide_index_map_type::from_index_map_buffer)>::type;
  auto convert_wide_index_map = &wide_index_map_type::from_index_map_buffer;
  wide_index_map_buffer_type wide_index_map_storage;
  wide_index_map_storage.f().allocate(3);
  wide_index_map_storage.f()[0] = 0;
  wide_index_map_storage.f()[1] = 3;
  wide_index_map_storage.f()[2] = 1;
  wide_index_map_storage.kept_ids().allocate(2);
  wide_index_map_storage.kept_ids()[0] = 0;
  wide_index_map_storage.kept_ids()[1] = 2;
  auto wide_index_map =
      convert_wide_index_map(std::move(wide_index_map_storage));
  auto wide_index_map_shallow = wide_index_map;
  auto wide_index_map_deep = wide_index_map.deep_copy();
  wide_index_map = {};
  if (wide_index_map.is_valid() || !wide_index_map_shallow.is_valid() ||
      !wide_index_map_deep.is_valid() ||
      wide_index_map_shallow.f[1] != std::int64_t{3} ||
      wide_index_map_deep.f[1] != std::int64_t{3} ||
      wide_index_map_deep.kept_ids[1] != std::int64_t{2}) {
    std::cerr << "installed-package int64 basic index map symbol failed\n";
    return 6;
  }

  array_storage_t<std::int64_t> face_offset_storage;
  face_offset_storage.allocate(2);
  face_offset_storage[0] = 0;
  face_offset_storage[1] = 4;
  array_storage_t<std::int64_t> face_index_storage;
  face_index_storage.allocate(4);
  face_index_storage[0] = 0;
  face_index_storage[1] = 1;
  face_index_storage[2] = 2;
  face_index_storage[3] =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  auto dynamic_faces =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
          tf::cpp::nd_array<std::int64_t>::from_buffer(
              std::move(face_offset_storage)),
          tf::cpp::nd_array<std::int64_t>::from_buffer(
              std::move(face_index_storage)));
  if (dynamic_faces.size() != 1 || dynamic_faces.data().length() != 4 ||
      dynamic_faces.data()[3] <= std::numeric_limits<std::int32_t>::max()) {
    std::cerr
        << "installed-package int64 dynamic face connectivity call failed\n";
    return 5;
  }

  return 0;
}
