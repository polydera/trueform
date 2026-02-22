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
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <string>

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

} // namespace

EMSCRIPTEN_BINDINGS(trueform_io) {
  emscripten::function("read_stl", &sync_read_stl);
  emscripten::function("dispatch_read_stl", &async_read_stl);
  emscripten::function("read_stl_buffer", &sync_read_stl_buffer);
  emscripten::function("dispatch_read_stl_buffer", &async_read_stl_buffer);
  emscripten::function("read_obj", &sync_read_obj);
  emscripten::function("dispatch_read_obj", &async_read_obj);
  emscripten::function("read_obj_buffer", &sync_read_obj_buffer);
  emscripten::function("dispatch_read_obj_buffer", &async_read_obj_buffer);
}
