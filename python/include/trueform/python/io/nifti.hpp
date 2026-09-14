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

#include "../util/make_numpy_array.hpp"
#include "../volume/to_python_volume.hpp"
#include "../volume/volume_wrapper.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <string>
#include <trueform/core/frame.hpp>
#include <trueform/io/nifti_file.hpp>
#include <trueform/io/read_nifti.hpp>
#include <trueform/io/write_nifti.hpp>
#include <trueform/volume/volume_buffer.hpp>
#include <utility>

namespace tf::py {

/// The fact a refusal names, in the reader's own words; the entry that was
/// asked says its own name in front of it.
inline auto nifti_status_fact(tf::nifti_status status) -> const char * {
  switch (status) {
  case tf::nifti_status::ok:
    return "ok";
  case tf::nifti_status::unreadable:
    return "the file cannot be read";
  case tf::nifti_status::bad_magic:
    return "a foreign magic, not a NIfTI-1 file";
  case tf::nifti_status::header_pair:
    return "a two-file header pair (.hdr/.img) is not supported";
  case tf::nifti_status::truncated:
    return "truncated bytes";
  case tf::nifti_status::unsupported_datatype:
    return "an unsupported dtype";
  case tf::nifti_status::unsupported_dims:
    return "a real fourth dimension";
  case tf::nifti_status::inconsistent_header:
    return "a self-contradictory header";
  }
  return "refused";
}

inline auto nifti_refusal(const char *entry, tf::nifti_status status)
    -> std::invalid_argument {
  return std::invalid_argument(std::string(entry) + ": " +
                               nifti_status_fact(status));
}

/// One sample row read: the file's samples in `T`, its grid, and the pose the
/// axis-aligned grid could not absorb (none when the file is unposed).
template <typename T> auto read_nifti_impl(const std::string &path) {
  auto file = tf::read_nifti<T>(path);
  if (!file)
    throw nifti_refusal("read_nifti", file.status);
  auto pose = file.posed
                  ? nanobind::cast(make_numpy_array(file.frame.transformation()))
                  : nanobind::object(nanobind::none());
  auto field = to_python_volume(std::move(file.volume));
  return nanobind::make_tuple(field[0], field[1], field[2], field[3],
                              std::move(pose), file.reflecting);
}

/// The writer is buffer-shaped, so the wrapper's borrowed field is copied
/// once at the IO boundary; a posed volume writes its pose as the sform.
template <typename T, typename Coord>
auto write_nifti_impl(volume_wrapper<T, Coord> &vol, const std::string &path)
    -> bool {
  const auto buffer = tf::make_volume_buffer(vol.view());
  return vol.has_transformation()
             ? tf::write_nifti(buffer, tf::make_frame(vol.transformation_view()),
                               path)
             : tf::write_nifti(buffer, path);
}

/// The NIfTI entries of one sample row, registered beside that row's other
/// entries so `(T, Coord)` is stated once, by the row itself.
template <typename T, typename Coord>
auto register_volume_nifti(nanobind::module_ &m, const char *suffix) -> void {
  m.def((std::string("read_nifti_") + suffix).c_str(), &read_nifti_impl<T>,
        nanobind::arg("path"),
        "Read a NIfTI-1 volume's samples in this row's sample type.");
  m.def((std::string("write_nifti_") + suffix).c_str(),
        &write_nifti_impl<T, Coord>, nanobind::arg("volume"),
        nanobind::arg("path"),
        "Write a volume as NIfTI-1; true when the file was written.");
}

auto register_io_nifti(nanobind::module_ &m) -> void;

} // namespace tf::py
