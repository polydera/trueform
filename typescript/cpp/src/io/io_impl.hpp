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

#include "trueform/io/read_obj.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/io/write_obj.hpp"
#include "trueform/io/write_stl.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <string>

namespace tf::ts::io {

// ============================================================================
// Result type — mesh-data read carries the coordinate dtype; faces stay int32.
// ============================================================================

template <typename Real> struct mesh_data_result_t {
  wasm_offset_blocked_buffer<int, int> faces;
  wasm_ndarray<Real> points;
};

// ============================================================================
// Copy a JS Uint8Array/ArrayBuffer into a NUL-terminated tf::buffer<char>.
// ============================================================================

inline auto copy_js_to_buffer(const emscripten::val &js_data)
    -> tf::buffer<char> {
  const auto len = js_data["length"].as<std::size_t>();
  tf::buffer<char> buf;
  buf.allocate(len + 1);
  auto view = emscripten::typed_memory_view(
      len, reinterpret_cast<unsigned char *>(buf.data()));
  emscripten::val(view).call<void>("set", js_data);
  buf[len] = '\0';
  return buf;
}

// ============================================================================
// Read STL — STL coordinates are float32 by spec; no Real dispatch.
// ============================================================================

inline auto sync_read_stl(const std::string &path) -> wasm_mesh<float> {
  return wasm_mesh<float>::from_polygons_buffer(tf::read_stl(path));
}

inline auto async_read_stl(const std::string &path) -> promise_t {
  return promise([path]() -> wasm_mesh<float> { return sync_read_stl(path); });
}

inline auto sync_read_stl_buffer(const emscripten::val &js_data)
    -> wasm_mesh<float> {
  auto buf = copy_js_to_buffer(js_data);
  return wasm_mesh<float>::from_polygons_buffer(
      tf::read_stl(tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
}

inline auto async_read_stl_buffer(const emscripten::val &js_data)
    -> promise_t {
  auto buf = copy_js_to_buffer(js_data);
  return promise([buf = std::move(buf)]() -> wasm_mesh<float> {
    return wasm_mesh<float>::from_polygons_buffer(
        tf::read_stl(tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
  });
}

// ============================================================================
// Read OBJ — coordinate dtype dispatches on Real.
// ============================================================================

template <typename Real>
auto sync_read_obj(const std::string &path) -> wasm_mesh<Real> {
  return wasm_mesh<Real>::from_polygons_buffer(
      tf::read_obj<3, int, Real>(path));
}

template <typename Real>
auto async_read_obj(const std::string &path) -> promise_t {
  return promise(
      [path]() -> wasm_mesh<Real> { return sync_read_obj<Real>(path); });
}

template <typename Real>
auto sync_read_obj_buffer(const emscripten::val &js_data) -> wasm_mesh<Real> {
  auto buf = copy_js_to_buffer(js_data);
  return wasm_mesh<Real>::from_polygons_buffer(tf::read_obj<3, int, Real>(
      tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
}

template <typename Real>
auto async_read_obj_buffer(const emscripten::val &js_data) -> promise_t {
  auto buf = copy_js_to_buffer(js_data);
  return promise([buf = std::move(buf)]() -> wasm_mesh<Real> {
    return wasm_mesh<Real>::from_polygons_buffer(tf::read_obj<3, int, Real>(
        tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
  });
}

template <typename Real>
auto sync_read_obj_buffer_data(const emscripten::val &js_data)
    -> mesh_data_result_t<Real> {
  auto buf = copy_js_to_buffer(js_data);
  auto poly = tf::read_obj<int, Real>(
      tf::make_range(buf.data(), buf.data() + buf.size() - 1));
  auto points_len = poly.points_buffer().data_buffer().size();
  return {
      wasm_offset_blocked_buffer<int, int>::from_buffer(
          std::move(poly.faces_buffer())),
      wasm_ndarray<Real>::from_buffer(
          std::move(poly.points_buffer().data_buffer()),
          {static_cast<int>(points_len / 3), 3}),
  };
}

template <typename Real>
auto async_read_obj_buffer_data(const emscripten::val &js_data) -> promise_t {
  auto buf = copy_js_to_buffer(js_data);
  return promise(
      [buf = std::move(buf)]() -> mesh_data_result_t<Real> {
        auto poly = tf::read_obj<int, Real>(
            tf::make_range(buf.data(), buf.data() + buf.size() - 1));
        auto points_len = poly.points_buffer().data_buffer().size();
        return {
            wasm_offset_blocked_buffer<int, int>::from_buffer(
                std::move(poly.faces_buffer())),
            wasm_ndarray<Real>::from_buffer(
                std::move(poly.points_buffer().data_buffer()),
                {static_cast<int>(points_len / 3), 3}),
        };
      });
}

// ============================================================================
// Write STL — output is float32 by spec; float64 input is downcast inside
// the writer (write_stl_to_buffer casts coordinates to float).
// ============================================================================

template <typename Real>
auto sync_write_stl_buffer(wasm_mesh<Real> &m) -> wasm_ndarray<std::int8_t> {
  auto poly = m.polygons_range();
  tf::buffer<std::int8_t> buf;
  if (m.has_transformation()) {
    buf = tf::write_stl_to_buffer<std::int8_t>(
        poly | tf::tag(m.transformation_view()));
  } else {
    buf = tf::write_stl_to_buffer<std::int8_t>(poly);
  }
  auto len = static_cast<int>(buf.size());
  return wasm_ndarray<std::int8_t>::from_buffer(std::move(buf), {len});
}

template <typename Real>
auto async_write_stl_buffer(wasm_mesh<Real> &m) -> promise_t {
  auto mesh = m.shallow_copy();
  return promise([mesh]() {
    return sync_write_stl_buffer<Real>(const_cast<wasm_mesh<Real> &>(mesh));
  });
}

// ============================================================================
// Write OBJ — output precision matches input Real (float32: %.9g, float64:
// %.17g).
// ============================================================================

template <typename Real>
auto sync_write_obj_buffer(wasm_mesh<Real> &m) -> wasm_ndarray<std::int8_t> {
  auto poly = m.polygons_range();
  tf::buffer<std::int8_t> buf;
  if (m.has_transformation()) {
    buf = tf::write_obj_to_buffer<std::int8_t>(
        poly | tf::tag(m.transformation_view()));
  } else {
    buf = tf::write_obj_to_buffer<std::int8_t>(poly);
  }
  auto len = static_cast<int>(buf.size());
  return wasm_ndarray<std::int8_t>::from_buffer(std::move(buf), {len});
}

template <typename Real>
auto async_write_obj_buffer(wasm_mesh<Real> &m) -> promise_t {
  auto mesh = m.shallow_copy();
  return promise([mesh]() {
    return sync_write_obj_buffer<Real>(const_cast<wasm_mesh<Real> &>(mesh));
  });
}

} // namespace tf::ts::io
