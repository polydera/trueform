/*
 * Direct-upstream-VTK utilities for the public tf::cpp examples.
 *
 * These adapters deliberately deep-copy facade-owned arrays into VTK. They do
 * not depend on trueform/vtk or on header-only Trueform algorithms.
 */
#pragma once

#include <trueform/core/curves_buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/static_size.hpp>
#include <trueform/core/transformation_view.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/core/offset_blocked_buffer.hpp>
#include <trueform/cpp/io/read_stl.hpp>

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace trueform::vtk_examples {

template <typename T>
auto make_array(std::vector<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  return make_array(std::vector<T>(values), std::move(shape));
}

template <typename Real> using matrix4 = std::array<Real, 16>;
template <typename Real> using point3 = std::array<Real, 3>;

template <typename Real> auto identity_matrix() -> matrix4<Real> {
  return {Real{1}, Real{0}, Real{0}, Real{0}, Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real>
auto multiply(const matrix4<Real> &left, const matrix4<Real> &right)
    -> matrix4<Real> {
  matrix4<Real> result{};
  for (int row = 0; row != 4; ++row)
    for (int column = 0; column != 4; ++column)
      for (int k = 0; k != 4; ++k)
        result[static_cast<std::size_t>(row * 4 + column)] +=
            left[static_cast<std::size_t>(row * 4 + k)] *
            right[static_cast<std::size_t>(k * 4 + column)];
  return result;
}

template <typename Real>
auto translation(Real x, Real y, Real z) -> matrix4<Real> {
  auto result = identity_matrix<Real>();
  result[3] = x;
  result[7] = y;
  result[11] = z;
  return result;
}

template <typename Real>
auto transform_point(const matrix4<Real> &matrix, const point3<Real> &point)
    -> point3<Real> {
  return {matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] +
              matrix[3],
          matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] +
              matrix[7],
          matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] +
              matrix[11]};
}

template <typename Real>
auto from_nd_array(const tf::cpp::nd_array<Real> &array) -> matrix4<Real> {
  if (array.ndim() != 2 || array.shape_at(0) != 4 || array.shape_at(1) != 4)
    throw std::invalid_argument("expected a 4x4 transformation");
  matrix4<Real> result;
  std::copy_n(array.begin(), 16, result.begin());
  return result;
}

template <typename Real>
auto set_vtk_matrix(vtkMatrix4x4 *target, const matrix4<Real> &source) -> void {
  for (int row = 0; row != 4; ++row)
    for (int column = 0; column != 4; ++column)
      target->SetElement(row, column,
                         static_cast<double>(source[row * 4 + column]));
  target->Modified();
}

template <typename Real>
auto from_vtk_matrix(vtkMatrix4x4 *source) -> matrix4<Real> {
  matrix4<Real> result;
  for (int row = 0; row != 4; ++row)
    for (int column = 0; column != 4; ++column)
      result[row * 4 + column] =
          static_cast<Real>(source->GetElement(row, column));
  return result;
}

template <typename Real> auto random_rotation_matrix() -> matrix4<Real> {
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<Real> distribution(Real{0}, Real{1});
  const auto u0 = distribution(generator);
  const auto u1 = distribution(generator);
  const auto u2 = distribution(generator);
  constexpr auto two_pi = Real{6.283185307179586476925286766559};
  const auto q0 = std::sqrt(Real{1} - u0) * std::sin(two_pi * u1);
  const auto q1 = std::sqrt(Real{1} - u0) * std::cos(two_pi * u1);
  const auto q2 = std::sqrt(u0) * std::sin(two_pi * u2);
  const auto q3 = std::sqrt(u0) * std::cos(two_pi * u2);
  auto result = identity_matrix<Real>();
  result[0] = Real{1} - Real{2} * (q2 * q2 + q3 * q3);
  result[1] = Real{2} * (q1 * q2 - q0 * q3);
  result[2] = Real{2} * (q1 * q3 + q0 * q2);
  result[4] = Real{2} * (q1 * q2 + q0 * q3);
  result[5] = Real{1} - Real{2} * (q1 * q1 + q3 * q3);
  result[6] = Real{2} * (q2 * q3 - q0 * q1);
  result[8] = Real{2} * (q1 * q3 - q0 * q2);
  result[9] = Real{2} * (q2 * q3 + q0 * q1);
  result[10] = Real{1} - Real{2} * (q1 * q1 + q2 * q2);
  return result;
}

template <typename Real> auto rotation_x(Real radians) -> matrix4<Real> {
  auto result = identity_matrix<Real>();
  result[5] = result[10] = std::cos(radians);
  result[6] = -std::sin(radians);
  result[9] = std::sin(radians);
  return result;
}

template <typename Real> auto rotation_y(Real radians) -> matrix4<Real> {
  auto result = identity_matrix<Real>();
  result[0] = result[10] = std::cos(radians);
  result[2] = std::sin(radians);
  result[8] = -std::sin(radians);
  return result;
}

/// @brief The assembly, over storage and a cache this scope holds.
///
/// Every entry takes the mesh, and a mesh is one reading of geometry the caller
/// owns, so this is the two-argument spelling of the one entrance — nothing is
/// bundled and nothing is owned here.
template <typename Index, typename Real, std::size_t Ngon>
auto mesh_over(const tf::polygons_buffer<Index, Real, 3, Ngon> &polygons,
               tf::cpp::cache<Index, Real, 3, Ngon> &cache)
    -> tf::cpp::mesh<Index, Real, 3, Ngon> {
  return {polygons.faces(), polygons.points(), cache};
}

/// Storage of its own, holding the same coordinates: what a cloud read off a
/// mesh's points takes when it is to outlive them.
template <typename Real>
auto copied_points(const tf::points_buffer<Real, 3> &storage)
    -> tf::points_buffer<Real, 3> {
  tf::points_buffer<Real, 3> result;
  result.data_buffer() = storage.data_buffer();
  return result;
}

/// The array tier takes `nd_array`, so a point batch or a scalar field built
/// from a mesh's own coordinates is copied into one.
template <typename Real>
auto points_array(const tf::points_buffer<Real, 3> &storage)
    -> tf::cpp::nd_array<Real> {
  const auto &coordinates = storage.data_buffer();
  tf::buffer<Real> values;
  values.allocate(coordinates.size());
  std::copy(coordinates.begin(), coordinates.end(), values.begin());
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(values), {static_cast<int>(storage.size()), 3});
}

template <typename Real>
auto point_mean(const tf::points_buffer<Real, 3> &storage) -> point3<Real> {
  const auto points = storage.points();
  if (points.size() == 0)
    throw std::invalid_argument("expected a non-empty point set");
  point3<Real> result{};
  for (const auto point : points)
    for (int axis = 0; axis != 3; ++axis)
      result[static_cast<std::size_t>(axis)] +=
          point[static_cast<std::size_t>(axis)];
  for (auto &coordinate : result)
    coordinate /= static_cast<Real>(points.size());
  return result;
}

template <typename Real>
auto point_bounds(const tf::points_buffer<Real, 3> &storage)
    -> std::pair<point3<Real>, point3<Real>> {
  const auto points = storage.points();
  if (points.size() == 0)
    throw std::invalid_argument("expected a non-empty point set");
  point3<Real> lower{points[0][0], points[0][1], points[0][2]};
  auto upper = lower;
  for (const auto point : points)
    for (std::size_t axis = 0; axis != 3; ++axis) {
      lower[axis] = std::min(lower[axis], point[axis]);
      upper[axis] = std::max(upper[axis], point[axis]);
    }
  return {lower, upper};
}

template <typename Real>
auto center_and_scale_matrix(const tf::points_buffer<Real, 3> &points,
                             Real target_radius = Real{10}) -> matrix4<Real> {
  const auto [lower, upper] = point_bounds(points);
  const point3<Real> center{(lower[0] + upper[0]) / Real{2},
                            (lower[1] + upper[1]) / Real{2},
                            (lower[2] + upper[2]) / Real{2}};
  const auto dx = upper[0] - lower[0];
  const auto dy = upper[1] - lower[1];
  const auto dz = upper[2] - lower[2];
  const auto radius = std::sqrt(dx * dx + dy * dy + dz * dz) / Real{2};
  if (!(radius > Real{0}))
    throw std::invalid_argument("cannot scale a zero-radius point set");
  const auto scale = target_radius / radius;
  auto result = identity_matrix<Real>();
  result[0] = result[5] = result[10] = scale;
  result[3] = -center[0] * scale;
  result[7] = -center[1] * scale;
  result[11] = -center[2] * scale;
  return result;
}

/// The coordinates move, so a caller that already asked something of a cache
/// over this storage says `points_changed()` afterwards.
template <typename Index, typename Real, std::size_t Ngon>
auto materialize_center_and_scale(
    tf::polygons_buffer<Index, Real, 3, Ngon> &polygons,
    Real target_radius = Real{10}) -> void {
  const auto transform =
      center_and_scale_matrix(polygons.points_buffer(), target_radius);
  for (auto point : polygons.points()) {
    const auto mapped =
        transform_point(transform, point3<Real>{point[0], point[1], point[2]});
    for (std::size_t axis = 0; axis != 3; ++axis)
      point[axis] = mapped[axis];
  }
}

template <typename Index, typename Real, std::size_t Ngon>
auto to_vtk_polydata(const tf::polygons_buffer<Index, Real, 3, Ngon> &mesh)
    -> vtkSmartPointer<vtkPolyData> {
  auto result = vtkSmartPointer<vtkPolyData>::New();
  auto points = vtkSmartPointer<vtkPoints>::New();
  const auto source_points = mesh.points();
  points->SetDataType(std::is_same_v<Real, float> ? VTK_FLOAT : VTK_DOUBLE);
  points->SetNumberOfPoints(static_cast<vtkIdType>(source_points.size()));
  for (std::size_t point = 0; point != source_points.size(); ++point)
    points->SetPoint(static_cast<vtkIdType>(point), source_points[point][0],
                     source_points[point][1], source_points[point][2]);

  auto cells = vtkSmartPointer<vtkCellArray>::New();
  std::vector<vtkIdType> ids;
  for (const auto face : mesh.faces()) {
    ids.clear();
    ids.reserve(static_cast<std::size_t>(face.size()));
    for (const auto id : face) {
      if (id < 0 ||
          static_cast<std::uint64_t>(id) >
              static_cast<std::uint64_t>(std::numeric_limits<vtkIdType>::max()))
        throw std::overflow_error("mesh index does not fit vtkIdType");
      ids.push_back(static_cast<vtkIdType>(id));
    }
    cells->InsertNextCell(static_cast<vtkIdType>(ids.size()), ids.data());
  }
  result->SetPoints(points);
  result->SetPolys(cells);
  return result;
}

/// A curve result is core's own storage, and an array-taking entry takes the
/// facade's two array carriers, so this is where a caller crosses between
/// them: a copy of the three flat arrays it is.
template <typename Blocks> auto as_offset_blocks(const Blocks &blocks) {
  using index_type = std::decay_t<decltype(blocks.data_buffer()[0])>;
  const auto &offsets = blocks.offsets_buffer();
  const auto &data = blocks.data_buffer();
  return tf::cpp::offset_blocked_buffer<index_type, index_type>::create(
      make_array<index_type>(
          std::vector<index_type>(offsets.begin(), offsets.end()),
          {static_cast<int>(offsets.size())}),
      make_array<index_type>(std::vector<index_type>(data.begin(), data.end()),
                             {static_cast<int>(data.size())}));
}

template <typename Curves>
auto curves_to_vtk_polydata(const Curves &curves)
    -> vtkSmartPointer<vtkPolyData> {
  auto result = vtkSmartPointer<vtkPolyData>::New();
  auto points = vtkSmartPointer<vtkPoints>::New();
  const auto source_points = curves.points();
  using real_type =
      std::remove_cv_t<std::remove_reference_t<decltype(source_points[0][0])>>;
  points->SetDataType(std::is_same_v<real_type, float> ? VTK_FLOAT
                                                       : VTK_DOUBLE);
  points->SetNumberOfPoints(static_cast<vtkIdType>(source_points.size()));
  for (std::size_t point = 0; point != source_points.size(); ++point)
    points->SetPoint(static_cast<vtkIdType>(point), source_points[point][0],
                     source_points[point][1], source_points[point][2]);
  auto lines = vtkSmartPointer<vtkCellArray>::New();
  std::vector<vtkIdType> ids;
  for (const auto path : curves.paths()) {
    ids.clear();
    ids.reserve(path.size());
    for (const auto id : path) {
      if (id < 0 ||
          static_cast<std::uint64_t>(id) >
              static_cast<std::uint64_t>(std::numeric_limits<vtkIdType>::max()))
        throw std::overflow_error("curve index does not fit vtkIdType");
      ids.push_back(static_cast<vtkIdType>(id));
    }
    lines->InsertNextCell(static_cast<vtkIdType>(ids.size()), ids.data());
  }
  result->SetPoints(points);
  result->SetLines(lines);
  return result;
}

template <typename Index, typename Real>
auto to_vtk_polydata(const tf::curves_buffer<Index, Real, 3> &curves)
    -> vtkSmartPointer<vtkPolyData> {
  return curves_to_vtk_polydata(curves);
}

inline auto replace_polydata(vtkPolyData *target, vtkPolyData *source) -> void {
  if (source &&
      (source->GetNumberOfPoints() != 0 || source->GetNumberOfCells() != 0))
    target->ShallowCopy(source);
  else
    target->Initialize();
  target->Modified();
}

inline auto make_actor(vtkPolyData *polydata,
                       std::array<double, 3> color = {0.8, 0.8, 0.8})
    -> vtkSmartPointer<vtkActor> {
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(polydata);
  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(color.data());
  return actor;
}

inline auto make_text_actor(const std::string &text, int font_size = 38,
                            std::array<double, 2> position = {0.03, 0.5},
                            bool right_justified = false)
    -> vtkSmartPointer<vtkTextActor> {
  auto actor = vtkSmartPointer<vtkTextActor>::New();
  actor->SetInput(text.c_str());
  actor->GetTextProperty()->SetFontSize(font_size);
  actor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
  actor->GetTextProperty()->SetVerticalJustificationToCentered();
  if (right_justified)
    actor->GetTextProperty()->SetJustificationToRight();
  else
    actor->GetTextProperty()->SetJustificationToLeft();
  actor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  actor->SetPosition(position.data());
  return actor;
}

inline auto make_text_strip(vtkRenderer *main_renderer, double height = 0.12)
    -> vtkSmartPointer<vtkRenderer> {
  main_renderer->SetViewport(0.0, height, 1.0, 1.0);
  main_renderer->SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0);
  auto text = vtkSmartPointer<vtkRenderer>::New();
  text->SetViewport(0.0, 0.0, 1.0, height);
  text->SetBackground(0.090, 0.143, 0.173);
  text->InteractiveOff();
  return text;
}

class rolling_average {
  std::vector<double> _samples;
  std::size_t _capacity;
  std::size_t _next = 0;

public:
  explicit rolling_average(std::size_t capacity) : _capacity(capacity) {
    _samples.reserve(capacity);
  }
  auto add(double value) -> void {
    if (_samples.size() != _capacity)
      _samples.push_back(value);
    else {
      _samples[_next] = value;
      _next = (_next + 1) % _capacity;
    }
  }
  auto average() const -> double {
    if (_samples.empty())
      return 0.0;
    return std::accumulate(_samples.begin(), _samples.end(), 0.0) /
           static_cast<double>(_samples.size());
  }
  auto empty() const -> bool { return _samples.empty(); }
};

inline auto format_milliseconds(double seconds) -> std::string {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << seconds * 1e3 << " ms";
  return stream.str();
}

inline auto format_microseconds(double seconds) -> std::string {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << seconds * 1e6 << " us";
  return stream.str();
}

template <typename Clock = std::chrono::steady_clock>
inline auto elapsed_seconds(typename Clock::time_point start) -> double {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// @brief The same geometry stated at mixed arity.
///
/// The arity is the storage's own type, so this is NEW storage and not a mode a
/// caller switches: a mixed carrier is what `read_obj` gives at data birth, and
/// this is how an example states one from triangles.
template <typename Index, typename Real>
auto as_mixed_mesh(const tf::polygons_buffer<Index, Real, 3, 3> &source)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> result;
  auto &offsets = result.faces_buffer().offsets_buffer();
  offsets.allocate(source.size() + 1);
  for (std::size_t face = 0; face != offsets.size(); ++face)
    offsets[face] = static_cast<Index>(face * 3);
  result.faces_buffer().data_buffer() = source.faces_buffer().data_buffer();
  result.points_buffer().data_buffer() = source.points_buffer().data_buffer();
  return result;
}

/// @brief What an example holds when it shows a mesh.
///
/// Core's own storage, the cache that remembers what was built for it, and the
/// placement this instance is drawn at. `mesh()` is the assembly every entry
/// takes; a copy is its own geometry, with its own cache.
struct mesh_actor_data {
  using index_type = tf::cpp::default_index_t;
  using mesh_type = tf::cpp::mesh<index_type, float>;

  tf::polygons_buffer<index_type, float, 3, 3> polygons;
  mutable tf::cpp::cache<index_type, float> cache;
  matrix4<float> placement = identity_matrix<float>();
  vtkSmartPointer<vtkPolyData> polydata;
  vtkSmartPointer<vtkActor> actor;
  vtkSmartPointer<vtkMatrix4x4> matrix;

  auto mesh() const -> mesh_type {
    return {polygons.faces(), polygons.points(), cache,
            tf::make_transformation_view<3>(placement.data())};
  }
};

inline auto make_mesh_actor_data(
    tf::polygons_buffer<mesh_actor_data::index_type, float, 3, 3> polygons,
    const matrix4<float> &transform) -> mesh_actor_data {
  auto polydata = to_vtk_polydata(polygons);
  auto actor = make_actor(polydata);
  auto matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  set_vtk_matrix(matrix, transform);
  actor->SetUserMatrix(matrix);
  return {std::move(polygons), {},
          transform,           std::move(polydata),
          std::move(actor),    std::move(matrix)};
}

inline auto load_stl_actor(const std::filesystem::path &path,
                           const point3<float> &position,
                           bool random_rotation = true,
                           float target_radius = 10.0F) -> mesh_actor_data {
  auto polygons = tf::cpp::read_stl(path.string());
  auto transform =
      center_and_scale_matrix(polygons.points_buffer(), target_radius);
  if (random_rotation)
    transform = multiply(random_rotation_matrix<float>(), transform);
  transform =
      multiply(translation(position[0], position[1], position[2]), transform);
  return make_mesh_actor_data(std::move(polygons), transform);
}

inline auto share_mesh_actor(const mesh_actor_data &source,
                             const point3<float> &position,
                             bool random_rotation = true,
                             float target_radius = 10.0F) -> mesh_actor_data {
  // Storage is a value: a copy is its own geometry, and it gets its own cache.
  auto polygons = source.polygons;
  auto transform =
      center_and_scale_matrix(polygons.points_buffer(), target_radius);
  if (random_rotation)
    transform = multiply(random_rotation_matrix<float>(), transform);
  transform =
      multiply(translation(position[0], position[1], position[2]), transform);
  return make_mesh_actor_data(std::move(polygons), transform);
}

inline auto rotate_around_world_point(const matrix4<float> &current,
                                      const point3<float> &center,
                                      const matrix4<float> &rotation)
    -> matrix4<float> {
  return multiply(
      translation(center[0], center[1], center[2]),
      multiply(
          rotation,
          multiply(translation(-center[0], -center[1], -center[2]), current)));
}

/// The frame is a tag, so an instance that moves changes only where it is
/// placed: the geometry and everything the cache built for it stand.
inline auto synchronize(mesh_actor_data &data, const matrix4<float> &transform)
    -> void {
  set_vtk_matrix(data.matrix, transform);
  data.placement = transform;
}

} // namespace trueform::vtk_examples
