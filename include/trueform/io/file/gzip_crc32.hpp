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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../external/miniz.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace tf::io::gz {

inline auto crc32_matrix_times(const std::uint32_t *matrix,
                               std::uint32_t vector) -> std::uint32_t {
  std::uint32_t sum = 0;
  while (vector != 0) {
    if ((vector & 1U) != 0)
      sum ^= *matrix;
    vector >>= 1;
    ++matrix;
  }
  return sum;
}

inline auto crc32_matrix_square(std::uint32_t *square,
                                const std::uint32_t *matrix) -> void {
  for (std::size_t i = 0; i < 32; ++i)
    square[i] = crc32_matrix_times(matrix, matrix[i]);
}

/// Concatenate two post-conditioned CRC-32 values without revisiting bytes
/// (zlib's `crc32_combine`: the polynomial's GF(2) transition matrix, raised
/// to the second run's length by repeated squaring).
inline auto crc32_combine(std::uint32_t first, std::uint32_t second,
                          std::size_t second_size) -> std::uint32_t {
  if (second_size == 0)
    return first;
  std::uint32_t even[32];
  std::uint32_t odd[32];
  odd[0] = 0xedb88320U;
  std::uint32_t row = 1;
  for (std::size_t i = 1; i < 32; ++i) {
    odd[i] = row;
    row <<= 1;
  }
  crc32_matrix_square(even, odd);
  crc32_matrix_square(odd, even);
  do {
    crc32_matrix_square(even, odd);
    if ((second_size & 1U) != 0)
      first = crc32_matrix_times(even, first);
    second_size >>= 1;
    if (second_size == 0)
      break;
    crc32_matrix_square(odd, even);
    if ((second_size & 1U) != 0)
      first = crc32_matrix_times(odd, first);
    second_size >>= 1;
  } while (second_size != 0);
  return first ^ second;
}

/// The ARM CRC32 instructions carry the gzip polynomial itself, so the
/// accelerated path and the miniz fallback answer the same value.
inline auto crc32_serial(const char *bytes, std::size_t size)
    -> std::uint32_t {
#if defined(__ARM_FEATURE_CRC32) && defined(MINIZ_LITTLE_ENDIAN) &&            \
    MINIZ_LITTLE_ENDIAN
  std::uint32_t crc = 0xffffffffU;
  while (size != 0 &&
         (reinterpret_cast<std::uintptr_t>(bytes) & std::uintptr_t{3}) != 0) {
    crc = __crc32b(crc, static_cast<unsigned char>(*bytes++));
    --size;
  }
#if defined(__aarch64__)
  while (size >= 8) {
    std::uint64_t word;
    std::memcpy(&word, bytes, 8);
    crc = __crc32d(crc, word);
    bytes += 8;
    size -= 8;
  }
#endif
  while (size >= 4) {
    std::uint32_t word;
    std::memcpy(&word, bytes, 4);
    crc = __crc32w(crc, word);
    bytes += 4;
    size -= 4;
  }
  while (size-- != 0)
    crc = __crc32b(crc, static_cast<unsigned char>(*bytes++));
  return ~crc;
#else
  return static_cast<std::uint32_t>(
      mz_crc32(MZ_CRC32_INIT,
               reinterpret_cast<const unsigned char *>(bytes), size));
#endif
}

struct crc32_piece {
  std::uint32_t check;
  std::size_t size;
};

/// CRC-32 is serial within a piece; independent pieces are combined in order.
inline auto gzip_crc32(const char *bytes, std::size_t size) -> std::uint32_t {
  constexpr std::size_t serial_below = 1U << 20;
  constexpr std::size_t target_piece_size = 4U << 20;
  if (size < serial_below)
    return crc32_serial(bytes, size);
  const auto available =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
  const auto part_count = std::min<std::size_t>(
      available * 2, (size + target_piece_size - 1) / target_piece_size);
  const auto piece_size = (size + part_count - 1) / part_count;
  tf::buffer<crc32_piece> pieces;
  pieces.allocate(part_count);
  tf::parallel_for(std::size_t{0}, part_count,
                   [&](std::size_t begin, std::size_t end) {
                     for (auto part = begin; part < end; ++part) {
                       const auto offset = part * piece_size;
                       const auto length = std::min(piece_size, size - offset);
                       pieces.begin()[part] = {
                           crc32_serial(bytes + offset, length), length};
                     }
                   });
  auto check = pieces.begin()->check;
  for (std::size_t part = 1; part < part_count; ++part)
    check = crc32_combine(check, pieces.begin()[part].check,
                          pieces.begin()[part].size);
  return check;
}

} // namespace tf::io::gz
