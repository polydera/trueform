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
#include "../nifti_file.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tf::io::nifti {

/// The NIfTI-1 layout this tier reads and writes, stated once: the header
/// is 348 bytes, and these are the offsets of every field either side
/// touches.
inline constexpr std::size_t header_size = 348;
inline constexpr std::size_t at_sizeof_hdr = 0;
inline constexpr std::size_t at_dim = 40;
inline constexpr std::size_t at_datatype = 70;
inline constexpr std::size_t at_bitpix = 72;
inline constexpr std::size_t at_pixdim = 76;
inline constexpr std::size_t at_vox_offset = 108;
inline constexpr std::size_t at_scl_slope = 112;
inline constexpr std::size_t at_scl_inter = 116;
inline constexpr std::size_t at_xyzt_units = 123;
inline constexpr std::size_t at_qform_code = 252;
inline constexpr std::size_t at_sform_code = 254;
inline constexpr std::size_t at_quatern = 256;
inline constexpr std::size_t at_qoffset = 268;
inline constexpr std::size_t at_srow = 280;
inline constexpr std::size_t at_magic = 344;
/// A single-file (`n+1`) member's samples start at 352 or later.
inline constexpr std::size_t single_file_offset = 352;

template <typename T> auto swapped_bytes(T value) -> T {
  unsigned char bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  for (std::size_t i = 0; i < sizeof(T) / 2; ++i) {
    const auto keep = bytes[i];
    bytes[i] = bytes[sizeof(T) - 1 - i];
    bytes[sizeof(T) - 1 - i] = keep;
  }
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

template <typename T>
auto field(const char *header, std::size_t offset, bool swap) -> T {
  T value;
  std::memcpy(&value, header + offset, sizeof(T));
  return swap ? swapped_bytes(value) : value;
}

template <typename T>
auto put_field(char *header, std::size_t offset, T value) -> void {
  std::memcpy(header + offset, &value, sizeof(T));
}

inline auto datatype_of_code(std::int16_t code) -> tf::nifti_datatype {
  switch (code) {
  case 2:
    return tf::nifti_datatype::uint8;
  case 4:
    return tf::nifti_datatype::int16;
  case 8:
    return tf::nifti_datatype::int32;
  case 16:
    return tf::nifti_datatype::float32;
  case 64:
    return tf::nifti_datatype::float64;
  case 512:
    return tf::nifti_datatype::uint16;
  default:
    return tf::nifti_datatype::unsupported;
  }
}

inline auto code_of_datatype(tf::nifti_datatype datatype) -> std::int16_t {
  switch (datatype) {
  case tf::nifti_datatype::uint8:
    return 2;
  case tf::nifti_datatype::int16:
    return 4;
  case tf::nifti_datatype::int32:
    return 8;
  case tf::nifti_datatype::float32:
    return 16;
  case tf::nifti_datatype::float64:
    return 64;
  case tf::nifti_datatype::uint16:
    return 512;
  default:
    return 0;
  }
}

inline auto bytes_of_datatype(tf::nifti_datatype datatype) -> std::size_t {
  switch (datatype) {
  case tf::nifti_datatype::uint8:
    return 1;
  case tf::nifti_datatype::int16:
  case tf::nifti_datatype::uint16:
    return 2;
  case tf::nifti_datatype::int32:
  case tf::nifti_datatype::float32:
    return 4;
  case tf::nifti_datatype::float64:
    return 8;
  default:
    return 0;
  }
}

template <typename T> constexpr auto datatype_of() -> tf::nifti_datatype {
  if constexpr (std::is_same_v<T, std::uint8_t>)
    return tf::nifti_datatype::uint8;
  else if constexpr (std::is_same_v<T, std::int16_t>)
    return tf::nifti_datatype::int16;
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return tf::nifti_datatype::uint16;
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return tf::nifti_datatype::int32;
  else if constexpr (std::is_same_v<T, float>)
    return tf::nifti_datatype::float32;
  else if constexpr (std::is_same_v<T, double>)
    return tf::nifti_datatype::float64;
  else
    return tf::nifti_datatype::unsupported;
}

} // namespace tf::io::nifti
