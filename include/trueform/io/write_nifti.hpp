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
#include "../core/frame_like.hpp"
#include "../core/point.hpp"
#include "../core/range.hpp"
#include "../core/transformation.hpp"
#include "../volume/volume_buffer.hpp"
#include "./file/gzip.hpp"
#include "./nifti/nifti_affine.hpp"
#include "./nifti/nifti_header.hpp"
#include "./nifti_file.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>

namespace tf::io::nifti {

template <typename T, typename Coord, typename Transformation>
auto serialize_nifti(const tf::volume_buffer<T, Coord, 3> &volume,
                     const Transformation &world) -> tf::buffer<char> {
  for (const auto extent : volume.dims())
    if (extent > 32767)
      return {};
  constexpr std::size_t data_offset = single_file_offset;
  tf::buffer<char> out;
  out.allocate(data_offset + volume.voxel_count() * sizeof(T));
  auto *header = out.begin();
  std::memset(header, 0, data_offset);

  put_field(header, at_sizeof_hdr, std::int32_t(header_size));
  put_field(header, at_dim, std::int16_t(3));
  for (std::size_t axis = 0; axis < 7; ++axis)
    put_field(header, at_dim + 2 * (axis + 1),
              axis < 3 ? std::int16_t(volume.dims()[axis]) : std::int16_t(1));
  put_field(header, at_datatype, code_of_datatype(datatype_of<T>()));
  put_field(header, at_bitpix, std::int16_t(8 * sizeof(T)));

  tf::point<float, 3> spacing, origin;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    spacing[axis] = float(volume.spacing()[axis]);
    origin[axis] = float(volume.origin()[axis]);
  }
  put_field(header, at_pixdim, 1.f);
  for (std::size_t axis = 0; axis < 3; ++axis)
    put_field(header, at_pixdim + 4 * (axis + 1), spacing[axis]);
  put_field(header, at_vox_offset, float(data_offset));
  put_field(header, at_scl_slope, 1.f);
  put_field(header, at_scl_inter, 0.f);
  put_field(header, at_xyzt_units, char(2));
  put_field(header, at_sform_code, std::int16_t(1));

  float srow[12];
  compose_srows(spacing, origin, world, srow);
  for (std::size_t i = 0; i < 12; ++i)
    put_field(header, at_srow + 4 * i, srow[i]);
  header[at_magic] = 'n';
  header[at_magic + 1] = '+';
  header[at_magic + 2] = '1';

  std::memcpy(out.begin() + data_offset, volume.samples_buffer().begin(),
              volume.voxel_count() * sizeof(T));
  return out;
}

inline auto write_bytes(tf::range<const char *, tf::dynamic_size> bytes,
                        std::string_view path) -> bool {
  if (bytes.size() == 0)
    return false;
  const auto gz = path.size() > 3 &&
                  path.substr(path.size() - 3) == std::string_view{".gz"};
  tf::buffer<char> deflated;
  if (gz) {
    deflated = io::gzip_deflated(bytes);
    if (deflated.size() == 0)
      return false;
    bytes = tf::make_range(static_cast<const char *>(deflated.begin()),
                           deflated.size());
  }
  std::ofstream file{std::string(path), std::ios::binary};
  if (!file)
    return false;
  file.write(bytes.begin(), std::streamsize(bytes.size()));
  return file.good();
}

} // namespace tf::io::nifti

namespace tf {

/// @ingroup io
/// @brief Serialize a volume to NIfTI-1 bytes, posed by the stated frame.
///
/// The grid's spacing and origin compose with the frame into the file's
/// sform; samples are written in their own type, unscaled, the spatial
/// unit stated as millimetres. An extent NIfTI-1 cannot represent (past
/// 32767) refuses with an empty buffer.
template <typename T, typename Coord, typename FramePolicy>
auto write_nifti_to_buffer(const tf::volume_buffer<T, Coord, 3> &volume,
                           const tf::frame_like<3, FramePolicy> &frame)
    -> tf::buffer<char> {
  static_assert(io::nifti::datatype_of<T>() != nifti_datatype::unsupported,
                "NIfTI stores uint8, int16, uint16, int32, float32 or "
                "float64 samples");
  return io::nifti::serialize_nifti(volume, frame.transformation());
}

/// @overload
template <typename T, typename Coord>
auto write_nifti_to_buffer(const tf::volume_buffer<T, Coord, 3> &volume)
    -> tf::buffer<char> {
  static_assert(io::nifti::datatype_of<T>() != nifti_datatype::unsupported,
                "NIfTI stores uint8, int16, uint16, int32, float32 or "
                "float64 samples");
  return io::nifti::serialize_nifti(
      volume, tf::make_identity_transformation<float, 3>());
}

/// @ingroup io
/// @brief Write a volume to a `.nii` or `.nii.gz` file, posed by the frame.
///
/// False when the volume cannot be represented or the file cannot be
/// written; nothing is created on refusal. A gzip member's trailer size is
/// modulo 4 GiB, so a payload past that writes a legal file this library's
/// own reader refuses.
template <typename T, typename Coord, typename FramePolicy>
auto write_nifti(const tf::volume_buffer<T, Coord, 3> &volume,
                 const tf::frame_like<3, FramePolicy> &frame,
                 std::string_view path) -> bool {
  const auto bytes = write_nifti_to_buffer(volume, frame);
  return io::nifti::write_bytes(tf::make_range(bytes.begin(), bytes.size()),
                                path);
}

/// @overload
template <typename T, typename Coord>
auto write_nifti(const tf::volume_buffer<T, Coord, 3> &volume,
                 std::string_view path) -> bool {
  const auto bytes = write_nifti_to_buffer(volume);
  return io::nifti::write_bytes(tf::make_range(bytes.begin(), bytes.size()),
                                path);
}

} // namespace tf
