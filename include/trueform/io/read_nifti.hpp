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
#include "../core/buffer.hpp"
#include "../core/frame.hpp"
#include "../core/range.hpp"
#include "../volume/volume_buffer.hpp"
#include "./file/gzip.hpp"
#include "./file/mapped_file.hpp"
#include "./nifti/convert_nifti_samples.hpp"
#include "./nifti/nifti_affine.hpp"
#include "./nifti/nifti_header.hpp"
#include "./nifti_file.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tf::io::nifti {

struct parsed_nifti {
  tf::nifti_header_info info;
  nifti_pose pose;
  bool swap = false;
  bool scale = false;
  std::size_t data_offset = 0;
  std::size_t count = 0;
};

inline auto parse_nifti(tf::range<const char *, tf::dynamic_size> bytes,
                        bool header_only) -> parsed_nifti {
  parsed_nifti parsed;
  auto refuse = [&](tf::nifti_status status) -> parsed_nifti & {
    parsed.info.status = status;
    return parsed;
  };
  if (bytes.size() < header_size)
    return refuse(tf::nifti_status::truncated);
  const auto *header = bytes.begin();
  const auto *magic = header + at_magic;
  if (magic[0] == 'n' && magic[1] == 'i' && magic[2] == '1' && magic[3] == 0)
    return refuse(tf::nifti_status::header_pair);
  if (magic[0] != 'n' || magic[1] != '+' || magic[2] != '1' || magic[3] != 0)
    return refuse(tf::nifti_status::bad_magic);
  if (field<std::int32_t>(header, at_sizeof_hdr, false) ==
      std::int32_t(header_size))
    parsed.swap = false;
  else if (field<std::int32_t>(header, at_sizeof_hdr, true) ==
           std::int32_t(header_size))
    parsed.swap = true;
  else
    return refuse(tf::nifti_status::bad_magic);
  const auto swap = parsed.swap;

  parsed.info.datatype =
      datatype_of_code(field<std::int16_t>(header, at_datatype, swap));
  if (parsed.info.datatype == tf::nifti_datatype::unsupported)
    return refuse(tf::nifti_status::unsupported_datatype);
  const auto bitpix = field<std::int16_t>(header, at_bitpix, swap);
  if (bitpix != 0 &&
      bitpix != std::int16_t(8 * bytes_of_datatype(parsed.info.datatype)))
    return refuse(tf::nifti_status::inconsistent_header);

  const auto ndim = field<std::int16_t>(header, at_dim, swap);
  if (ndim < 1 || ndim > 7)
    return refuse(tf::nifti_status::unsupported_dims);
  parsed.count = 1;
  for (std::int16_t axis = 1; axis <= ndim; ++axis) {
    const auto extent = field<std::int16_t>(header, at_dim + 2 * axis, swap);
    if (extent < 1 || (axis > 3 && extent > 1))
      return refuse(tf::nifti_status::unsupported_dims);
    if (axis <= 3) {
      parsed.info.dims[std::size_t(axis - 1)] = extent;
      if (parsed.count > std::size_t(-1) / std::size_t(extent))
        return refuse(tf::nifti_status::unsupported_dims);
      parsed.count *= std::size_t(extent);
    }
  }
  for (std::size_t axis = std::size_t(ndim); axis < 3; ++axis)
    parsed.info.dims[axis] = 1;

  parsed.info.slope = field<float>(header, at_scl_slope, swap);
  parsed.info.intercept = field<float>(header, at_scl_inter, swap);
  parsed.scale = std::isfinite(parsed.info.slope) &&
                 std::isfinite(parsed.info.intercept) &&
                 parsed.info.slope != 0.f &&
                 !(parsed.info.slope == 1.f && parsed.info.intercept == 0.f);
  switch (field<char>(header, at_xyzt_units, swap) & 0x07) {
  case 1:
    parsed.info.units = tf::nifti_units::meters;
    break;
  case 2:
    parsed.info.units = tf::nifti_units::millimeters;
    break;
  case 3:
    parsed.info.units = tf::nifti_units::micrometers;
    break;
  default:
    parsed.info.units = tf::nifti_units::unknown;
    break;
  }

  parsed.pose = decompose_pose(header, swap);
  parsed.info.spacing = parsed.pose.spacing;
  parsed.info.posed = parsed.pose.posed;
  parsed.info.reflecting = parsed.pose.reflecting;

  if (!header_only) {
    const auto vox_offset = double(field<float>(header, at_vox_offset, swap));
    if (!std::isfinite(vox_offset) || vox_offset < double(single_file_offset) ||
        vox_offset > double(bytes.size()))
      return refuse(tf::nifti_status::truncated);
    parsed.data_offset = std::size_t(vox_offset);
    const auto sample_bytes = bytes_of_datatype(parsed.info.datatype);
    if (parsed.count > (std::size_t(-1) - parsed.data_offset) / sample_bytes ||
        parsed.data_offset + parsed.count * sample_bytes > bytes.size())
      return refuse(tf::nifti_status::truncated);
  }
  parsed.info.status = tf::nifti_status::ok;
  return parsed;
}

} // namespace tf::io::nifti

namespace tf {

/// @ingroup io
/// @brief Read a NIfTI-1 header's facts without its samples.
///
/// A gzipped file costs only its head. Use the facts to state the `T` a
/// full read should keep the samples in.
inline auto read_nifti_header(tf::range<const char *, tf::dynamic_size> bytes)
    -> nifti_header_info {
  if (io::is_gzip(bytes)) {
    const auto head = io::gzip_inflated_prefix(bytes, io::nifti::header_size);
    if (head.size() < io::nifti::header_size) {
      nifti_header_info info;
      info.status = nifti_status::truncated;
      return info;
    }
    return io::nifti::parse_nifti(tf::make_range(head.begin(), head.size()),
                                  true)
        .info;
  }
  return io::nifti::parse_nifti(bytes, true).info;
}

/// @overload
inline auto read_nifti_header(tf::range<char *, tf::dynamic_size> bytes)
    -> nifti_header_info {
  return read_nifti_header(
      tf::make_range(static_cast<const char *>(bytes.begin()), bytes.size()));
}

/// @overload
inline auto read_nifti_header(std::string_view path) -> nifti_header_info {
  io::mapped_file map(path);
  if (!map) {
    nifti_header_info info;
    info.status = nifti_status::unreadable;
    return info;
  }
  return read_nifti_header(tf::make_range(map.data(), map.size()));
}

/// @ingroup io
/// @brief Read a NIfTI-1 volume (.nii bytes, gzipped or plain) from memory.
///
/// The samples land in `T` — converted when the file's dtype differs or the
/// header states scl scaling (computed in double, cast once); kept bit-exact
/// when it matches. The default keeps every consumer running without a
/// dtype dispatch; state the file's own type (see @ref read_nifti_header)
/// for bit-exact large int32/float64 samples. The grid is in the file's
/// stated spatial unit, unconverted; a nonzero bitpix disagreeing with the
/// datatype refuses as `inconsistent_header` (a zero bitpix is tolerated —
/// the datatype is the authority); `posed` says whether `frame` carries
/// an orientation the grid could not absorb — a non-orthogonal sform makes
/// that frame non-rigid, so metric consumers must not assume rigidity.
/// A gzipped file must hold one member; trailing input refuses as
/// truncated.
template <typename T = float>
auto read_nifti(tf::range<const char *, tf::dynamic_size> bytes)
    -> nifti_file<T> {
  static_assert(io::nifti::datatype_of<T>() != nifti_datatype::unsupported,
                "NIfTI stores uint8, int16, uint16, int32, float32 or "
                "float64 samples");
  nifti_file<T> file;
  tf::buffer<char> inflated;
  if (io::is_gzip(bytes)) {
    inflated = io::gzip_inflated(bytes);
    if (inflated.size() == 0) {
      file.status = nifti_status::truncated;
      return file;
    }
    bytes = tf::make_range(
        static_cast<const char *>(inflated.begin()), inflated.size());
  }
  const auto parsed = io::nifti::parse_nifti(bytes, false);
  file.status = parsed.info.status;
  if (file.status != nifti_status::ok)
    return file;

  file.volume = tf::volume_buffer<T, float, 3>(
      parsed.info.dims, parsed.pose.spacing, parsed.pose.origin);
  file.posed = parsed.pose.posed;
  file.reflecting = parsed.pose.reflecting;
  if (file.posed)
    file.frame = tf::make_frame(parsed.pose.world);

  const auto *samples = bytes.begin() + parsed.data_offset;
  auto *out = file.volume.samples_buffer().begin();
  const auto convert = [&](auto sample) {
    using source_t = decltype(sample);
    io::nifti::convert_samples<source_t>(samples, out, parsed.count,
                                         parsed.swap, parsed.scale,
                                         parsed.info.slope,
                                         parsed.info.intercept);
  };
  switch (parsed.info.datatype) {
  case nifti_datatype::uint8:
    convert(std::uint8_t{});
    break;
  case nifti_datatype::int16:
    convert(std::int16_t{});
    break;
  case nifti_datatype::uint16:
    convert(std::uint16_t{});
    break;
  case nifti_datatype::int32:
    convert(std::int32_t{});
    break;
  case nifti_datatype::float32:
    convert(float{});
    break;
  default:
    convert(double{});
    break;
  }
  return file;
}

/// @overload
template <typename T = float>
auto read_nifti(tf::range<char *, tf::dynamic_size> bytes) -> nifti_file<T> {
  return read_nifti<T>(
      tf::make_range(static_cast<const char *>(bytes.begin()), bytes.size()));
}

/// @ingroup io
/// @brief Read a NIfTI-1 volume from a `.nii` or `.nii.gz` file.
///
/// The file must not be modified for the duration of the read.
template <typename T = float>
auto read_nifti(std::string_view path) -> nifti_file<T> {
  io::mapped_file map(path);
  if (!map) {
    nifti_file<T> file;
    file.status = nifti_status::unreadable;
    return file;
  }
  return read_nifti<T>(tf::make_range(map.data(), map.size()));
}

} // namespace tf
