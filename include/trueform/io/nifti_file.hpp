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
#include "../core/frame.hpp"
#include "../core/point.hpp"
#include "../core/transformation.hpp"
#include "../volume/volume_buffer.hpp"
#include <array>
#include <cstdint>

namespace tf {

/// @ingroup io
/// @brief Why a NIfTI read refused, or `ok`.
enum class nifti_status : std::uint8_t {
  ok,
  unreadable,
  bad_magic,
  header_pair,
  truncated,
  unsupported_datatype,
  unsupported_dims,
  inconsistent_header
};

/// @ingroup io
/// @brief The sample type a NIfTI file stores.
enum class nifti_datatype : std::uint8_t {
  uint8,
  int16,
  uint16,
  int32,
  float32,
  float64,
  unsupported
};

/// @ingroup io
/// @brief The spatial unit a NIfTI file states for its grid.
enum class nifti_units : std::uint8_t {
  unknown,
  meters,
  millimeters,
  micrometers
};

/// @ingroup io
/// @brief The header facts of a NIfTI-1 file, read without its samples.
///
/// `slope` and `intercept` are the file's scl fields verbatim; scaling
/// applies on read when `slope` is nonzero and the pair is not (1, 0).
/// The grid is in the file's stated `units`, unconverted.
struct nifti_header_info {
  nifti_status status = nifti_status::unreadable;
  nifti_datatype datatype = nifti_datatype::unsupported;
  std::array<int, 3> dims{};
  tf::point<float, 3> spacing{1.f, 1.f, 1.f};
  float slope = 0.f;
  float intercept = 0.f;
  nifti_units units = nifti_units::unknown;
  /// The file states an affine the axis-aligned grid cannot absorb.
  bool posed = false;
  /// The stated affine flips handedness (its determinant is negative).
  bool reflecting = false;
  explicit operator bool() const { return status == nifti_status::ok; }
};

/// @ingroup io
/// @brief Payload returned by `read_nifti<T>`.
///
/// `volume` holds the samples in `T` on a grid in the file's stated unit;
/// `frame` poses it in the file's world space and is identity-valued when
/// `posed` is false. A frame tag always transforms per point, so branch on
/// `posed` — consume the volume bare when false, compose
/// `file.volume.volume() | tf::tag(file.frame)` when true. `reflecting`
/// says the pose flips handedness.
template <typename T = float> struct nifti_file {
  tf::volume_buffer<T, float, 3> volume;
  tf::frame<float, 3> frame =
      tf::make_frame(tf::make_identity_transformation<float, 3>());
  bool posed = false;
  bool reflecting = false;
  nifti_status status = nifti_status::unreadable;
  explicit operator bool() const { return status == nifti_status::ok; }
};

} // namespace tf
