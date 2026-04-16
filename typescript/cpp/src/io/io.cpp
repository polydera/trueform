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

struct mesh_data_result {
  tf::ts::wasm_offset_blocked_buffer<int, int> faces;
  tf::ts::wasm_ndarray<float> points;
};

namespace {

auto sync_read_stl(const std::string &path) -> tf::ts::wasm_mesh {
  return tf::ts::wasm_mesh::from_polygons_buffer(tf::read_stl(path));
}

auto async_read_stl(const std::string &path) -> tf::ts::promise_t {
  return tf::ts::promise(
      [path]() -> tf::ts::wasm_mesh { return sync_read_stl(path); });
}

auto sync_read_obj(const std::string &path) -> tf::ts::wasm_mesh {
  return tf::ts::wasm_mesh::from_polygons_buffer(tf::read_obj<3>(path));
}

auto async_read_obj(const std::string &path) -> tf::ts::promise_t {
  return tf::ts::promise(
      [path]() -> tf::ts::wasm_mesh { return sync_read_obj(path); });
}

auto copy_js_to_buffer_(const emscripten::val &js_data)
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

auto sync_read_stl_buffer(const emscripten::val &js_data)
    -> tf::ts::wasm_mesh {
  auto buf = copy_js_to_buffer_(js_data);
  return tf::ts::wasm_mesh::from_polygons_buffer(
      tf::read_stl(tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
}

auto async_read_stl_buffer(const emscripten::val &js_data)
    -> tf::ts::promise_t {
  auto buf = copy_js_to_buffer_(js_data);
  return tf::ts::promise(
      [buf = std::move(buf)]() -> tf::ts::wasm_mesh {
        return tf::ts::wasm_mesh::from_polygons_buffer(
            tf::read_stl(tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
      });
}

auto sync_read_obj_buffer(const emscripten::val &js_data)
    -> tf::ts::wasm_mesh {
  auto buf = copy_js_to_buffer_(js_data);
  return tf::ts::wasm_mesh::from_polygons_buffer(
      tf::read_obj<3>(tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
}

auto async_read_obj_buffer(const emscripten::val &js_data)
    -> tf::ts::promise_t {
  auto buf = copy_js_to_buffer_(js_data);
  return tf::ts::promise(
      [buf = std::move(buf)]() -> tf::ts::wasm_mesh {
        return tf::ts::wasm_mesh::from_polygons_buffer(
            tf::read_obj<3>(tf::make_range(buf.data(), buf.data() + buf.size() - 1)));
      });
}

auto sync_read_obj_buffer_data(const emscripten::val &js_data)
    -> mesh_data_result {
  auto buf = copy_js_to_buffer_(js_data);
  auto poly = tf::read_obj<int>(
      tf::make_range(buf.data(), buf.data() + buf.size() - 1));
  auto points_len = poly.points_buffer().data_buffer().size();
  return {
      tf::ts::wasm_offset_blocked_buffer<int, int>::from_buffer(
          std::move(poly.faces_buffer())),
      tf::ts::wasm_ndarray<float>::from_buffer(
          std::move(poly.points_buffer().data_buffer()),
          {static_cast<int>(points_len / 3), 3}),
  };
}

auto async_read_obj_buffer_data(const emscripten::val &js_data)
    -> tf::ts::promise_t {
  auto buf = copy_js_to_buffer_(js_data);
  return tf::ts::promise(
      [buf = std::move(buf)]() -> mesh_data_result {
        auto poly = tf::read_obj<int>(
            tf::make_range(buf.data(), buf.data() + buf.size() - 1));
        auto points_len = poly.points_buffer().data_buffer().size();
        return {
            tf::ts::wasm_offset_blocked_buffer<int, int>::from_buffer(
                std::move(poly.faces_buffer())),
            tf::ts::wasm_ndarray<float>::from_buffer(
                std::move(poly.points_buffer().data_buffer()),
                {static_cast<int>(points_len / 3), 3}),
        };
      });
}

// ============================================================================
// Write STL / OBJ to buffer
// ============================================================================

auto sync_write_stl_buffer(tf::ts::wasm_mesh &m)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  auto poly = m.polygons_range();
  tf::buffer<std::int8_t> buf;
  if (m.has_transformation()) {
    buf = tf::write_stl_to_buffer<std::int8_t>(
        poly | tf::tag(m.transformation_view()));
  } else {
    buf = tf::write_stl_to_buffer<std::int8_t>(poly);
  }
  auto len = static_cast<int>(buf.size());
  return tf::ts::wasm_ndarray<std::int8_t>::from_buffer(std::move(buf), {len});
}

auto async_write_stl_buffer(tf::ts::wasm_mesh &m) -> tf::ts::promise_t {
  auto mesh = m.shallow_copy();
  return tf::ts::promise([mesh]() {
    return sync_write_stl_buffer(
        const_cast<tf::ts::wasm_mesh &>(mesh));
  });
}

auto sync_write_obj_buffer(tf::ts::wasm_mesh &m)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  auto poly = m.polygons_range();
  tf::buffer<std::int8_t> buf;
  if (m.has_transformation()) {
    buf = tf::write_obj_to_buffer<std::int8_t>(
        poly | tf::tag(m.transformation_view()));
  } else {
    buf = tf::write_obj_to_buffer<std::int8_t>(poly);
  }
  auto len = static_cast<int>(buf.size());
  return tf::ts::wasm_ndarray<std::int8_t>::from_buffer(std::move(buf), {len});
}

auto async_write_obj_buffer(tf::ts::wasm_mesh &m) -> tf::ts::promise_t {
  auto mesh = m.shallow_copy();
  return tf::ts::promise([mesh]() {
    return sync_write_obj_buffer(
        const_cast<tf::ts::wasm_mesh &>(mesh));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_io) {
  emscripten::value_object<mesh_data_result>("MeshDataResult")
      .field("faces", &mesh_data_result::faces)
      .field("points", &mesh_data_result::points);

  emscripten::function("read_stl", &sync_read_stl);
  emscripten::function("dispatch_read_stl", &async_read_stl);
  emscripten::function("read_stl_buffer", &sync_read_stl_buffer);
  emscripten::function("dispatch_read_stl_buffer", &async_read_stl_buffer);
  emscripten::function("read_obj", &sync_read_obj);
  emscripten::function("dispatch_read_obj", &async_read_obj);
  emscripten::function("read_obj_buffer", &sync_read_obj_buffer);
  emscripten::function("dispatch_read_obj_buffer", &async_read_obj_buffer);
  emscripten::function("read_obj_buffer_data", &sync_read_obj_buffer_data);
  emscripten::function("dispatch_read_obj_buffer_data", &async_read_obj_buffer_data);
  emscripten::function("write_stl_buffer", &sync_write_stl_buffer);
  emscripten::function("dispatch_write_stl_buffer", &async_write_stl_buffer);
  emscripten::function("write_obj_buffer", &sync_write_obj_buffer);
  emscripten::function("dispatch_write_obj_buffer", &async_write_obj_buffer);
}
