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
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../external/miniz.hpp"
#include "./gzip_crc32.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace tf::io {

inline auto is_gzip(tf::range<const char *, tf::dynamic_size> bytes) -> bool {
  return bytes.size() >= 2 && static_cast<unsigned char>(bytes[0]) == 0x1f &&
         static_cast<unsigned char>(bytes[1]) == 0x8b;
}

namespace gz {

/// One gzip member's raw deflate stream, located past the header's optional
/// fields and before the trailer; empty on a malformed member.
inline auto payload(tf::range<const char *, tf::dynamic_size> bytes)
    -> tf::range<const char *, tf::dynamic_size> {
  constexpr std::size_t header = 10;
  constexpr std::size_t trailer = 8;
  if (bytes.size() < header + trailer || !is_gzip(bytes))
    return {};
  const auto *u = reinterpret_cast<const unsigned char *>(bytes.begin());
  if (u[2] != 8)
    return {};
  const auto flags = u[3];
  std::size_t at = header;
  const auto end = bytes.size() - trailer;
  if ((flags & 0x04) != 0) {
    if (at + 2 > end)
      return {};
    at += 2 + (std::size_t(u[at]) | (std::size_t(u[at + 1]) << 8));
  }
  for (const auto bit : {0x08, 0x10})
    if ((flags & bit) != 0) {
      while (at < end && u[at] != 0)
        ++at;
      ++at;
    }
  if ((flags & 0x02) != 0)
    at += 2;
  if (at > end)
    return {};
  return tf::make_range(bytes.begin() + at, end - at);
}

/// Deflate expands at most 1032:1, so a trailer size past that bound over
/// the member's own payload is hostile, not data.
inline constexpr std::uint64_t max_expansion = 1032;

struct heap_deleter {
  auto operator()(void *block) const -> void { mz_free(block); }
};

} // namespace gz

/// @ingroup io
/// @brief Inflate one gzip member; an empty buffer signals refusal.
///
/// The trailer's size field is the allocation, admitted only within the
/// deflate expansion bound over the payload's own size; the crc is
/// verified against the inflated bytes; a member past 4 GiB refuses, and
/// so does input trailing past the one member — this reader speaks single
/// members only.
inline auto gzip_inflated(tf::range<const char *, tf::dynamic_size> bytes)
    -> tf::buffer<char> {
  const auto deflate = gz::payload(bytes);
  if (deflate.size() == 0)
    return {};
  std::uint32_t crc, isize;
  std::memcpy(&crc, deflate.end(), 4);
  std::memcpy(&isize, deflate.end() + 4, 4);
  if (std::uint64_t(isize) > std::uint64_t(deflate.size()) * gz::max_expansion)
    return {};
  tf::buffer<char> out;
  out.allocate(isize);
  const auto decompressor = std::make_unique<tinfl_decompressor>();
  tinfl_init(decompressor.get());
  auto consumed = deflate.size();
  auto produced = out.size();
  const auto status = tinfl_decompress(
      decompressor.get(),
      reinterpret_cast<const mz_uint8 *>(deflate.begin()), &consumed,
      reinterpret_cast<mz_uint8 *>(out.begin()),
      reinterpret_cast<mz_uint8 *>(out.begin()), &produced,
      TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
  if (status != TINFL_STATUS_DONE || produced != out.size() ||
      consumed != deflate.size())
    return {};
  const auto check = gz::gzip_crc32(out.begin(), out.size());
  if (check != crc)
    return {};
  return out;
}

/// @ingroup io
/// @brief Inflate only the first `want` bytes of a gzip member.
///
/// Streams through the inflate window and stops as soon as the prefix is in
/// hand, so peeking at a large member's head costs its head; empty on
/// refusal, the trailer unverified since the member is not read whole.
inline auto
gzip_inflated_prefix(tf::range<const char *, tf::dynamic_size> bytes,
                     std::size_t want) -> tf::buffer<char> {
  const auto deflate = gz::payload(bytes);
  if (deflate.size() == 0)
    return {};
  const auto decompressor = std::make_unique<tinfl_decompressor>();
  tinfl_init(decompressor.get());
  tf::buffer<char> window;
  window.allocate(TINFL_LZ_DICT_SIZE);
  tf::buffer<char> out;
  out.allocate(want);
  std::size_t filled = 0;
  std::size_t window_at = 0;
  const auto *next = reinterpret_cast<const mz_uint8 *>(deflate.begin());
  auto available = deflate.size();
  while (filled < want) {
    auto in_size = available;
    auto out_size = std::size_t{TINFL_LZ_DICT_SIZE} - window_at;
    const auto status = tinfl_decompress(
        decompressor.get(), next, &in_size,
        reinterpret_cast<mz_uint8 *>(window.begin()),
        reinterpret_cast<mz_uint8 *>(window.begin()) + window_at, &out_size,
        0);
    next += in_size;
    available -= in_size;
    const auto take = out_size < want - filled ? out_size : want - filled;
    std::memcpy(out.begin() + filled, window.begin() + window_at, take);
    filled += take;
    window_at = (window_at + out_size) & (TINFL_LZ_DICT_SIZE - 1);
    if (status <= TINFL_STATUS_DONE || out_size == 0)
      break;
  }
  if (filled < want)
    return {};
  return out;
}

/// @ingroup io
/// @brief Deflate bytes into one gzip member; an empty buffer signals refusal.
inline auto gzip_deflated(tf::range<const char *, tf::dynamic_size> bytes)
    -> tf::buffer<char> {
  const auto flags = tdefl_create_comp_flags_from_zip_params(
      MZ_DEFAULT_LEVEL, -MZ_DEFAULT_WINDOW_BITS, MZ_DEFAULT_STRATEGY);
  std::size_t deflated_size = 0;
  const std::unique_ptr<void, gz::heap_deleter> deflated{
      tdefl_compress_mem_to_heap(bytes.begin(), bytes.size(), &deflated_size,
                                 flags)};
  if (deflated == nullptr)
    return {};
  constexpr unsigned char header[10] = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255};
  const std::uint32_t crc = gz::gzip_crc32(bytes.begin(), bytes.size());
  const auto isize = static_cast<std::uint32_t>(bytes.size());
  tf::buffer<char> out;
  out.allocate(sizeof(header) + deflated_size + 8);
  std::memcpy(out.begin(), header, sizeof(header));
  std::memcpy(out.begin() + sizeof(header), deflated.get(), deflated_size);
  std::memcpy(out.begin() + sizeof(header) + deflated_size, &crc, 4);
  std::memcpy(out.begin() + sizeof(header) + deflated_size + 4, &isize, 4);
  return out;
}

} // namespace tf::io
