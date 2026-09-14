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

#include "trueform/python/io/nifti.hpp"
#include <array>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <string>
#include <trueform/io/read_nifti.hpp>

namespace tf::py {
namespace {

auto read_nifti_header_impl(const std::string &path) {
  const auto header = tf::read_nifti_header(path);
  if (!header)
    throw nifti_refusal("read_nifti_header", header.status);
  const std::array<int, 3> dims = header.dims;
  const std::array<double, 3> spacing{static_cast<double>(header.spacing[0]),
                                      static_cast<double>(header.spacing[1]),
                                      static_cast<double>(header.spacing[2])};
  return nanobind::make_tuple(
      static_cast<int>(header.datatype), dims, spacing,
      static_cast<double>(header.slope), static_cast<double>(header.intercept),
      static_cast<int>(header.units), header.posed, header.reflecting);
}

} // namespace

auto register_io_nifti(nanobind::module_ &m) -> void {
  m.def("read_nifti_header", &read_nifti_header_impl, nanobind::arg("path"),
        "The header facts of a NIfTI-1 file, read without its samples.");
}

} // namespace tf::py
