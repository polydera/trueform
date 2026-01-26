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
#include <cstdint>
#include <fstream>
#include <string>

namespace tf::io {
class stl_point_collector {
public:
  // Read file and append x,y,z floats into tf::buffer<T>.
  template <class T>
  auto read(const std::string_view &path, tf::buffer<T> &out_xyz) -> bool {
    auto f = open_file_(path);
    if (!f)
      return false;

    bool is_bin{};
    if (!detect_binary_(f, is_bin))
      return false;

    return is_bin ? read_binary_into_(f, out_xyz)
                  : read_ascii_into_(f, out_xyz);
  }

private:
  // ---------- File helpers ----------
  static auto open_file_(const std::string_view &path) -> std::ifstream {
    return std::ifstream(std::string(path), std::ios::binary);
  }

  static auto file_size_(std::ifstream &f, std::uint64_t &sz) -> bool {
    f.clear();
    auto cur = f.tellg();
    f.seekg(0, std::ios::end);
    if (!f)
      return false;
    sz = static_cast<std::uint64_t>(f.tellg());
    f.seekg(cur);
    return true;
  }

  // ---------- Format detection ----------
  static auto detect_binary_(std::ifstream &f, bool &is_bin) -> bool {
    std::uint64_t sz{};
    if (!file_size_(f, sz))
      return false;
    if (sz < 84) {
      is_bin = false;
      return true;
    }

    std::uint32_t tri_count{};
    f.clear();
    f.seekg(80, std::ios::beg);
    f.read(reinterpret_cast<char *>(&tri_count), 4);
    if (!f)
      return false;

    const std::uint64_t expected = 84ull + 50ull * tri_count;
    is_bin = (expected == sz);
    return true;
  }

  // ---------- Counting ----------
  static auto count_points_ascii_(std::ifstream &f, std::uint64_t &out_points)
      -> bool {
    f.clear();
    f.seekg(0, std::ios::beg);
    std::uint64_t count = 0;
    std::string line;
    while (std::getline(f, line)) {
      const char *b = skip_ws_(line.c_str());
      if (starts_with_vertex_(b)) {
        char *endp{};
        (void)std::strtof(b + 6, &endp);
        (void)std::strtof(endp, &endp);
        (void)std::strtof(endp, &endp);
        ++count;
      }
    }
    out_points = count;
    return true;
  }

  static auto read_binary_triangle_count_(std::ifstream &f,
                                          std::uint32_t &tri_count) -> bool {
    f.clear();
    f.seekg(80, std::ios::beg);
    f.read(reinterpret_cast<char *>(&tri_count), 4);
    return static_cast<bool>(f);
  }

  // ---------- Reading: Binary ----------
  template <class T>
  auto read_binary_into_(std::ifstream &f, tf::buffer<T> &out) -> bool {
    std::uint32_t tri_count{};
    if (!read_binary_triangle_count_(f, tri_count))
      return false;

    const std::size_t bytes = static_cast<std::size_t>(tri_count) * 50u;
    tf::buffer<unsigned char> payload;
    payload.allocate(bytes);
    f.clear();
    f.seekg(84, std::ios::beg);
    if (!f.read(reinterpret_cast<char *>(payload.begin()),
                static_cast<std::streamsize>(payload.size())))
      return false;

    const std::uint64_t total_floats =
        static_cast<std::uint64_t>(tri_count) * 9ull;
    reserve_more_(out, total_floats);
    const std::size_t base = out.size();
    out.allocate(base + static_cast<std::size_t>(total_floats));

    const unsigned char *tri = payload.begin();
    T *dst = &out[base];
    for (std::uint32_t i = 0; i < tri_count; ++i) {
      const float *fl = reinterpret_cast<const float *>(tri);
      dst[0] = static_cast<T>(fl[3]);
      dst[1] = static_cast<T>(fl[4]);
      dst[2] = static_cast<T>(fl[5]);
      dst[3] = static_cast<T>(fl[6]);
      dst[4] = static_cast<T>(fl[7]);
      dst[5] = static_cast<T>(fl[8]);
      dst[6] = static_cast<T>(fl[9]);
      dst[7] = static_cast<T>(fl[10]);
      dst[8] = static_cast<T>(fl[11]);
      tri += 50;
      dst += 9;
    }
    return true;
  }

  // ---------- Reading: ASCII ----------
  template <class T>
  auto read_ascii_into_(std::ifstream &f, tf::buffer<T> &out) -> bool {
    std::uint64_t vcount{};
    if (!count_points_ascii_(f, vcount))
      return false;

    const std::uint64_t floats_to_add = vcount * 3ull;
    reserve_more_(out, floats_to_add);
    const std::size_t base = out.size();
    out.allocate(base + static_cast<std::size_t>(floats_to_add));
    T *dst = &out[base];

    f.clear();
    f.seekg(0, std::ios::beg);
    std::string line;
    while (std::getline(f, line)) {
      const char *p = skip_ws_(line.c_str());
      if (!starts_with_vertex_(p))
        continue;

      p += 6; // skip "vertex"
      float x{}, y{}, z{};
      if (!parse_three_floats_(p, x, y, z))
        return false;

      *dst++ = static_cast<T>(x);
      *dst++ = static_cast<T>(y);
      *dst++ = static_cast<T>(z);
    }
    return true;
  }

  // ---------- Parse helpers ----------
  static auto parse_three_floats_(const char *p, float &x, float &y, float &z)
      -> bool {
    char *endp{};
    p = skip_ws_(p);
    x = std::strtof(p, &endp);
    if (endp == p)
      return false;
    p = skip_ws_(endp);
    y = std::strtof(p, &endp);
    if (endp == p)
      return false;
    p = skip_ws_(endp);
    z = std::strtof(p, &endp);
    if (endp == p)
      return false;
    return true;
  }

  static auto skip_ws_(const char *p) -> const char * {
    while (*p == ' ' || *p == '\t')
      ++p;
    return p;
  }
  static auto starts_with_vertex_(const char *p) -> bool {
    return (p[0] == 'v' && p[1] == 'e' && p[2] == 'r' && p[3] == 't' &&
            p[4] == 'e' && p[5] == 'x') &&
           (p[6] == ' ' || p[6] == '\t');
  }

  // ---------- Reserve helper ----------
  template <class T>
  static auto reserve_more_(tf::buffer<T> &out, std::uint64_t floats_to_add)
      -> void {
    out.reserve(out.size() + static_cast<std::size_t>(floats_to_add));
  }
};
} // namespace tf::io
