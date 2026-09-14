/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once

#include <array>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <stdexcept>
#include <trueform/core/point.hpp>
#include <trueform/core/transformation_view.hpp>
#include <trueform/volume/volume.hpp>

namespace tf::py {

template <typename RealT>
auto to_volume_point(const std::array<double, 3> &a) -> tf::point<RealT, 3> {
  return tf::point<RealT, 3>{static_cast<RealT>(a[0]),
                             static_cast<RealT>(a[1]),
                             static_cast<RealT>(a[2])};
}

/// The layer's standard input shape for a scalar grid: the ndarray handle
/// retains the Python owner of the samples, the `tf::volume` view ranges over
/// that memory, and the tiny grid metadata (dims, spacing, origin) is held by
/// value. No sample is copied anywhere.
///
/// The carrier's two types cross the boundary separately: `T` is what NumPy
/// holds (int16 CT counts included) and `Coord` is the grid the samples stand
/// on, so an integer field keeps a floating millimetre spacing.
///
/// The samples argument is bound `noconvert`: an array of another dtype is
/// refused here rather than converted, because a conversion would produce a
/// temporary for this borrowing view to outlive. Choosing the row that matches
/// the caller's dtype is the facade's job.
template <typename T, typename Coord> class volume_wrapper {
public:
  volume_wrapper(nanobind::ndarray<nanobind::numpy, T, nanobind::ndim<1>,
                                   nanobind::c_contig>
                     samples,
                 std::array<int, 3> dims, std::array<double, 3> spacing,
                 std::array<double, 3> origin)
      : _samples(samples), _dims(dims),
        _spacing(to_volume_point<Coord>(spacing)),
        _origin(to_volume_point<Coord>(origin)) {
    for (int i = 0; i < 3; ++i)
      if (_dims[i] < 1)
        throw std::invalid_argument(
            "dims must be at least 1 along each axis");
    if (view().voxel_count() == 0)
      throw std::invalid_argument("volume must contain at least one sample");
    if (view().voxel_count() != _samples.size())
      throw std::invalid_argument(
          "samples size must equal dims[0] * dims[1] * dims[2]");
  }

  auto view() const {
    return tf::make_volume(_samples.data(), _dims, _spacing, _origin);
  }

  auto dims() const -> const std::array<int, 3> & { return _dims; }

  auto spacing() const -> std::array<double, 3> {
    return {static_cast<double>(_spacing[0]), static_cast<double>(_spacing[1]),
            static_cast<double>(_spacing[2])};
  }
  auto set_spacing(std::array<double, 3> spacing) -> void {
    _spacing = to_volume_point<Coord>(spacing);
  }

  auto origin() const -> std::array<double, 3> {
    return {static_cast<double>(_origin[0]), static_cast<double>(_origin[1]),
            static_cast<double>(_origin[2])};
  }
  auto set_origin(std::array<double, 3> origin) -> void {
    _origin = to_volume_point<Coord>(origin);
  }

  auto has_transformation() const -> bool {
    return _transformation.has_value();
  }

  auto transformation() const
      -> std::optional<nanobind::ndarray<nanobind::numpy, Coord,
                                         nanobind::shape<4, 4>,
                                         nanobind::c_contig>> {
    return _transformation;
  }

  auto transformation_view() const {
    const auto &trans = *_transformation;
    return tf::make_transformation_view<3>(trans.data());
  }

  auto set_transformation(nanobind::ndarray<nanobind::numpy, Coord,
                                            nanobind::shape<4, 4>,
                                            nanobind::c_contig>
                              transformation_array) -> void {
    _transformation = transformation_array;
  }

  auto clear_transformation() -> void { _transformation.reset(); }

  /// How many samples the grid holds; zero when an axis has no extent, which
  /// is what the constructor refuses.
  auto voxel_count() const -> std::size_t { return view().voxel_count(); }

  auto samples_array() const -> const nanobind::ndarray<
      nanobind::numpy, T, nanobind::ndim<1>, nanobind::c_contig> & {
    return _samples;
  }

private:
  nanobind::ndarray<nanobind::numpy, T, nanobind::ndim<1>,
                    nanobind::c_contig>
      _samples;
  std::array<int, 3> _dims;
  tf::point<Coord, 3> _spacing;
  tf::point<Coord, 3> _origin;
  std::optional<nanobind::ndarray<nanobind::numpy, Coord,
                                  nanobind::shape<4, 4>, nanobind::c_contig>>
      _transformation;
};

template <typename T, typename Coord>
auto register_volume_wrapper(nanobind::module_ &m, const char *name) -> void {
  using wrapper_t = volume_wrapper<T, Coord>;
  nanobind::class_<wrapper_t>(m, name)
      .def(nanobind::init<nanobind::ndarray<nanobind::numpy, T,
                                            nanobind::ndim<1>,
                                            nanobind::c_contig>,
                          std::array<int, 3>, std::array<double, 3>,
                          std::array<double, 3>>(),
           nanobind::arg("samples").noconvert(), nanobind::arg("dims"),
           nanobind::arg("spacing"), nanobind::arg("origin"))
      .def("dims", &wrapper_t::dims)
      .def("spacing", &wrapper_t::spacing)
      .def("set_spacing", &wrapper_t::set_spacing)
      .def("origin", &wrapper_t::origin)
      .def("set_origin", &wrapper_t::set_origin)
      .def("voxel_count", &wrapper_t::voxel_count)
      .def("samples_array", &wrapper_t::samples_array)
      .def("has_transformation", &wrapper_t::has_transformation)
      .def("transformation", &wrapper_t::transformation)
      .def("set_transformation", &wrapper_t::set_transformation)
      .def("clear_transformation", &wrapper_t::clear_transformation);
}

} // namespace tf::py
