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

#include "trueform/ts/core/build_intersect_structures.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

#include <trueform/core/algorithm/parallel_transform.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/csg/csg_graph.hpp>
#include <trueform/csg/expression.hpp>
#include <trueform/csg/make_csg_domains.hpp>
#include <trueform/csg/make_csg_mesh.hpp>
#include <trueform/csg/make_intersection_curves.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/triangulation_type.hpp>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tf {
namespace ts {

// Boolean expressions cross the boundary as a flat postfix program of
// int32 codes: id >= 0 pushes op(id); -1/-2/-3 pop two and push
// or/and/difference; -4 pops one and pushes complement.
inline auto decode_expr(const std::vector<int> &program) -> tf::csg::expr {
  std::vector<tf::csg::expr> stack;
  for (int code : program) {
    if (code >= 0) {
      stack.push_back(tf::csg::op(code));
      continue;
    }
    if (code == -4) {
      if (stack.size() < 1)
        throw std::runtime_error("malformed csg expression program");
      auto a = std::move(stack.back());
      stack.pop_back();
      stack.push_back(~a);
      continue;
    }
    if (stack.size() < 2)
      throw std::runtime_error("malformed csg expression program");
    auto b = std::move(stack.back());
    stack.pop_back();
    auto a = std::move(stack.back());
    stack.pop_back();
    switch (code) {
    case -1:
      stack.push_back(a | b);
      break;
    case -2:
      stack.push_back(a & b);
      break;
    case -3:
      stack.push_back(a - b);
      break;
    default:
      throw std::runtime_error("unknown csg expression opcode");
    }
  }
  if (stack.size() != 1)
    throw std::runtime_error("malformed csg expression program");
  return std::move(stack.back());
}

// emscripten::val cannot cross threads: extract on the main thread,
// capture the resulting POD vector in async lambdas.
inline auto extract_int_vector(emscripten::val js_array) -> std::vector<int> {
  auto n = js_array["length"].as<int>();
  std::vector<int> out;
  out.reserve(std::size_t(n));
  for (int i = 0; i < n; ++i)
    out.push_back(js_array[i].as<int>());
  return out;
}

template <typename Real>
auto extract_csg_meshes(emscripten::val js_meshes)
    -> std::vector<wasm_mesh<Real>> {
  auto n = js_meshes["length"].as<int>();
  if (n < 2)
    throw std::runtime_error("csgGraph: need at least 2 meshes");
  std::vector<wasm_mesh<Real>> meshes;
  meshes.reserve(std::size_t(n));
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh<Real>>(emscripten::allow_raw_pointers()));
  return meshes;
}

// ============================================================================
// Result structs (registered per Real as value_objects)
// ============================================================================

template <typename Real> struct csg_mesh_labeled_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> tag_labels;
  wasm_ndarray<std::int32_t> face_labels;
};

template <typename Real> struct csg_mesh_index_map_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> point_tag_labels;
  wasm_ndarray<std::int32_t> point_labels;
  wasm_ndarray<std::int32_t> face_tag_labels;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_ndarray<std::int32_t> point_f_offsets;
  wasm_ndarray<std::int32_t> point_f_data;
  int n_original_points = 0;
  int n_original_faces = 0;
  int n_tags = 0;
  int n_output_points = 0;
};

template <typename Real> struct csg_domains_t {
  std::vector<wasm_mesh<Real>> meshes;
  wasm_ndarray<std::int32_t> ids;
};

template <typename Real> struct csg_domains_labeled_t {
  std::vector<wasm_mesh<Real>> meshes;
  wasm_ndarray<std::int32_t> ids;
  wasm_ndarray<std::int32_t> tag_offsets;
  wasm_ndarray<std::int32_t> tag_data;
  wasm_ndarray<std::int32_t> face_offsets;
  wasm_ndarray<std::int32_t> face_data;
};

template <typename Real> struct csg_domains_index_map_t {
  std::vector<wasm_mesh<Real>> meshes;
  wasm_ndarray<std::int32_t> ids;
  wasm_ndarray<std::int32_t> face_tag_offsets;
  wasm_ndarray<std::int32_t> face_tag_data;
  wasm_ndarray<std::int32_t> face_offsets;
  wasm_ndarray<std::int32_t> face_data;
  wasm_ndarray<std::int32_t> point_tag_offsets;
  wasm_ndarray<std::int32_t> point_tag_data;
  wasm_ndarray<std::int32_t> point_offsets;
  wasm_ndarray<std::int32_t> point_data;
  int n_original_points = 0;
  int n_tags = 0;
  int n_output_points = 0;
  wasm_ndarray<std::int8_t> inclusion; // [n_cells, n_tags], 0/1
};

// ============================================================================
// The sealed engine: shared_ptr two-layer handle like wasm_mesh, so handle
// copies (including async lambda captures) share one graph build.
// ============================================================================

template <typename Real> class wasm_csg_graph {
  static auto make_form(wasm_mesh<Real> &m,
                        tf::transformation<Real, 3> &identity) {
    auto fm = m.face_membership_range();
    auto mel = m.manifold_edge_link_range();
    auto tv = m.has_transformation()
                  ? m.transformation_view()
                  : tf::make_transformation_view<3>(&identity(0, 0));
    return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
           tf::tag(mel) | tf::tag(tv);
  }

  struct data_t {
    std::vector<wasm_mesh<Real>> meshes;
    tf::transformation<Real, 3> identity;
    using form_t = decltype(make_form(std::declval<wasm_mesh<Real> &>(),
                                      std::declval<tf::transformation<Real, 3>
                                                       &>()));
    std::vector<form_t> forms;
    using forms_range_t =
        decltype(tf::make_range(std::declval<std::vector<form_t> &>()));
    tf::csg_graph<forms_range_t, tf::none_t> graph;

    static auto make_forms(std::vector<wasm_mesh<Real>> &ms,
                           tf::transformation<Real, 3> &identity)
        -> std::vector<form_t> {
      build_intersect_structures_all(ms);
      std::vector<form_t> forms;
      forms.reserve(ms.size());
      for (auto &m : ms)
        forms.push_back(make_form(m, identity));
      return forms;
    }

    static auto make_is_sheet(const std::vector<int> &sheets, std::size_t n)
        -> tf::buffer<char> {
      tf::buffer<char> is_sheet;
      if (sheets.empty())
        return is_sheet;
      is_sheet.allocate(n);
      std::fill(is_sheet.begin(), is_sheet.end(), char(0));
      for (int id : sheets)
        is_sheet[std::size_t(id)] = char(1);
      return is_sheet;
    }

    data_t(std::vector<wasm_mesh<Real>> ms, const std::vector<int> &sheets,
           int mode, double tolerance, int triangulation)
        : meshes(std::move(ms)),
          identity(tf::make_identity_transformation<Real, 3>()),
          forms(make_forms(meshes, identity)),
          graph(tf::make_range(forms), {},
                tf::arrangement_config{
                    tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                         tolerance},
                    static_cast<tf::triangulation_type>(triangulation)},
                make_is_sheet(sheets, meshes.size())) {}
  };

  std::shared_ptr<data_t> _data;

  static auto blocks_to_arrays(tf::offset_block_buffer<int, int> &&blocks) {
    return std::make_pair(
        wasm_ndarray<std::int32_t>::from_buffer(
            std::move(blocks.offsets_buffer())),
        wasm_ndarray<std::int32_t>::from_buffer(
            std::move(blocks.data_buffer())));
  }

public:
  wasm_csg_graph() = default;

  static auto from_meshes(std::vector<wasm_mesh<Real>> meshes,
                          const std::vector<int> &sheets, int mode,
                          double tolerance, int triangulation)
      -> wasm_csg_graph {
    wasm_csg_graph g;
    g._data = std::make_shared<data_t>(std::move(meshes), sheets, mode,
                                       tolerance, triangulation);
    return g;
  }

  auto is_valid() const -> bool { return _data != nullptr; }
  auto destroy() -> void { _data.reset(); }

  auto created_points() const -> wasm_ndarray<Real> {
    const auto &created = _data->graph.created_points();
    tf::buffer<Real> out;
    out.allocate(created.size() * 3);
    const auto &conv = _data->graph.converter();
    tf::parallel_transform(
        tf::make_range(created),
        tf::make_blocked_range<3>(tf::make_range(out)),
        [&](const auto &ip) {
          auto p = conv.deconvert(ip);
          return std::array<Real, 3>{Real(p[0]), Real(p[1]), Real(p[2])};
        });
    auto n = static_cast<int>(created.size());
    return wasm_ndarray<Real>::from_buffer(std::move(out), {n, 3});
  }

  auto intersection_curves() const -> wasm_curves<Real> {
    return wasm_curves<Real>::from_curves_buffer(
        tf::make_intersection_curves(_data->graph));
  }

  auto mesh(const std::vector<int> &program) const -> wasm_mesh<Real> {
    if (program.empty()) {
      return wasm_mesh<Real>::from_polygons_buffer(
          tf::make_csg_mesh(_data->graph));
    }
    auto result = tf::make_csg_mesh(_data->graph, decode_expr(program));
    return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
  }

  auto mesh_with_labels(const std::vector<int> &program) const
      -> csg_mesh_labeled_t<Real> {
    if (program.empty()) {
      auto [result, tag_labels, face_labels] =
          tf::make_csg_mesh(_data->graph, tf::return_source_ids);
      return {wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
              wasm_ndarray<std::int32_t>::from_buffer(std::move(tag_labels)),
              wasm_ndarray<std::int32_t>::from_buffer(
                  std::move(face_labels))};
    }
    auto [result, tag_labels, face_labels] = tf::make_csg_mesh(
        _data->graph, decode_expr(program), tf::return_source_ids);
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(tag_labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
  }

  auto mesh_with_index_map(const std::vector<int> &program) const
      -> csg_mesh_index_map_t<Real> {
    if (program.empty())
      throw std::runtime_error(
          "the full-arrangement mesh has no index-map form; pass an "
          "expression");
    auto [result, imap] = tf::make_csg_mesh(_data->graph,
                                            decode_expr(program),
                                            tf::return_index_map);
    auto [pf_off, pf_data] = blocks_to_arrays(std::move(imap.point_f));
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
            wasm_ndarray<std::int32_t>::from_buffer(
                std::move(imap.point_tag_labels)),
            wasm_ndarray<std::int32_t>::from_buffer(
                std::move(imap.point_labels)),
            wasm_ndarray<std::int32_t>::from_buffer(
                std::move(imap.face_tag_labels)),
            wasm_ndarray<std::int32_t>::from_buffer(
                std::move(imap.face_labels)),
            std::move(pf_off),
            std::move(pf_data),
            int(imap.n_original_points),
            int(imap.n_original_faces),
            int(imap.n_tags),
            int(imap.n_output_points)};
  }

  auto domains(const std::vector<int> &program, int config) const
      -> csg_domains_t<Real> {
    auto run = [&](auto &&...expr_arg) -> csg_domains_t<Real> {
      auto [cells, ids] =
          tf::make_csg_domains(_data->graph, expr_arg...,
                               static_cast<tf::domain_config>(config));
      std::vector<wasm_mesh<Real>> meshes;
      meshes.reserve(cells.size());
      for (auto &cell : cells)
        meshes.push_back(
            wasm_mesh<Real>::from_polygons_buffer(std::move(cell)));
      return {std::move(meshes),
              wasm_ndarray<std::int32_t>::from_buffer(std::move(ids))};
    };
    if (program.empty())
      return run();
    return run(decode_expr(program));
  }

  auto domains_with_labels(const std::vector<int> &program, int config) const
      -> csg_domains_labeled_t<Real> {
    auto run = [&](auto &&...expr_arg) -> csg_domains_labeled_t<Real> {
      auto [cells, ids, tag_blocks, face_blocks] = tf::make_csg_domains(
          _data->graph, expr_arg..., static_cast<tf::domain_config>(config),
          tf::return_source_ids);
      std::vector<wasm_mesh<Real>> meshes;
      meshes.reserve(cells.size());
      for (auto &cell : cells)
        meshes.push_back(
            wasm_mesh<Real>::from_polygons_buffer(std::move(cell)));
      auto [t_off, t_data] = blocks_to_arrays(std::move(tag_blocks));
      auto [f_off, f_data] = blocks_to_arrays(std::move(face_blocks));
      return {std::move(meshes),
              wasm_ndarray<std::int32_t>::from_buffer(std::move(ids)),
              std::move(t_off), std::move(t_data), std::move(f_off),
              std::move(f_data)};
    };
    if (program.empty())
      return run();
    return run(decode_expr(program));
  }

  auto domains_with_index_map(const std::vector<int> &program,
                              int config) const
      -> csg_domains_index_map_t<Real> {
    auto run = [&](auto &&...expr_arg) -> csg_domains_index_map_t<Real> {
      auto [cells, ids, imap] = tf::make_csg_domains(
          _data->graph, expr_arg..., static_cast<tf::domain_config>(config),
          tf::return_index_map);
      std::vector<wasm_mesh<Real>> meshes;
      meshes.reserve(cells.size());
      for (auto &cell : cells)
        meshes.push_back(
            wasm_mesh<Real>::from_polygons_buffer(std::move(cell)));
      auto [ft_off, ft_data] = blocks_to_arrays(std::move(imap.face_tag_blocks));
      auto [f_off, f_data] = blocks_to_arrays(std::move(imap.face_blocks));
      auto [pt_off, pt_data] =
          blocks_to_arrays(std::move(imap.point_tag_blocks));
      auto [p_off, p_data] = blocks_to_arrays(std::move(imap.point_blocks));
      // block size read at runtime: never deduce it from a template
      // parameter (dynamic_size would read as -1)
      const int n_cells = static_cast<int>(imap.inclusion.size());
      const int n_ops = static_cast<int>(imap.inclusion.block_size());
      // TS "bool" dtype is int8-backed; bool and int8 share size,
      // alignment, and a type-erased deleter, so the storage moves
      static_assert(sizeof(bool) == 1 &&
                    alignof(bool) == alignof(std::int8_t));
      auto inclusion = wasm_ndarray<std::int8_t>::from_buffer(
          std::move(reinterpret_cast<tf::buffer<std::int8_t> &>(
              imap.inclusion.data_buffer())),
          {n_cells, n_ops});
      return {std::move(meshes),
              wasm_ndarray<std::int32_t>::from_buffer(std::move(ids)),
              std::move(ft_off), std::move(ft_data), std::move(f_off),
              std::move(f_data), std::move(pt_off), std::move(pt_data),
              std::move(p_off), std::move(p_data),
              int(imap.n_original_points), int(imap.n_tags),
              int(imap.n_output_points), std::move(inclusion)};
    };
    if (program.empty())
      return run();
    return run(decode_expr(program));
  }
};

// ============================================================================
// Sync entry points (main thread; extract val immediately)
// ============================================================================

template <typename Real>
auto sync_csg_graph(emscripten::val js_meshes, emscripten::val js_sheets,
                    int mode, double tolerance, int triangulation)
    -> wasm_csg_graph<Real> {
  return wasm_csg_graph<Real>::from_meshes(
      extract_csg_meshes<Real>(js_meshes), extract_int_vector(js_sheets),
      mode, tolerance, triangulation);
}

template <typename Real>
auto sync_csg_mesh(wasm_csg_graph<Real> &g, emscripten::val js_program)
    -> wasm_mesh<Real> {
  return g.mesh(extract_int_vector(js_program));
}

template <typename Real>
auto sync_csg_mesh_with_labels(wasm_csg_graph<Real> &g,
                               emscripten::val js_program)
    -> csg_mesh_labeled_t<Real> {
  return g.mesh_with_labels(extract_int_vector(js_program));
}

template <typename Real>
auto sync_csg_mesh_with_index_map(wasm_csg_graph<Real> &g,
                                  emscripten::val js_program)
    -> csg_mesh_index_map_t<Real> {
  return g.mesh_with_index_map(extract_int_vector(js_program));
}

template <typename Real>
auto sync_csg_domains(wasm_csg_graph<Real> &g, emscripten::val js_program,
                      int config) -> csg_domains_t<Real> {
  return g.domains(extract_int_vector(js_program), config);
}

template <typename Real>
auto sync_csg_domains_with_labels(wasm_csg_graph<Real> &g,
                                  emscripten::val js_program, int config)
    -> csg_domains_labeled_t<Real> {
  return g.domains_with_labels(extract_int_vector(js_program), config);
}

template <typename Real>
auto sync_csg_domains_with_index_map(wasm_csg_graph<Real> &g,
                                     emscripten::val js_program, int config)
    -> csg_domains_index_map_t<Real> {
  return g.domains_with_index_map(extract_int_vector(js_program), config);
}

template <typename Real>
auto sync_csg_created_points(wasm_csg_graph<Real> &g) -> wasm_ndarray<Real> {
  return g.created_points();
}

template <typename Real>
auto sync_csg_intersection_curves(wasm_csg_graph<Real> &g)
    -> wasm_curves<Real> {
  return g.intersection_curves();
}

// ============================================================================
// Async twins: extract emscripten::val on the main thread, capture the
// graph handle and POD vectors by copy; queries are const, so the const
// captures call straight through.
// ============================================================================

template <typename Real>
auto async_csg_graph(emscripten::val js_meshes, emscripten::val js_sheets,
                     int mode, double tolerance, int triangulation)
    -> promise_t {
  auto meshes = extract_csg_meshes<Real>(js_meshes);
  auto sheets = extract_int_vector(js_sheets);
  return promise([meshes = std::move(meshes), sheets = std::move(sheets),
                  mode, tolerance, triangulation]() {
    return wasm_csg_graph<Real>::from_meshes(meshes, sheets, mode, tolerance,
                                             triangulation);
  });
}

template <typename Real>
auto async_csg_intersection_curves(wasm_csg_graph<Real> &g) -> promise_t {
  return promise([g = g]() { return g.intersection_curves(); });
}

template <typename Real>
auto async_csg_mesh(wasm_csg_graph<Real> &g, emscripten::val js_program)
    -> promise_t {
  auto program = extract_int_vector(js_program);
  return promise(
      [g = g, program = std::move(program)]() { return g.mesh(program); });
}

template <typename Real>
auto async_csg_mesh_with_labels(wasm_csg_graph<Real> &g,
                                emscripten::val js_program) -> promise_t {
  auto program = extract_int_vector(js_program);
  return promise([g = g, program = std::move(program)]() {
    return g.mesh_with_labels(program);
  });
}

template <typename Real>
auto async_csg_mesh_with_index_map(wasm_csg_graph<Real> &g,
                                   emscripten::val js_program) -> promise_t {
  auto program = extract_int_vector(js_program);
  return promise([g = g, program = std::move(program)]() {
    return g.mesh_with_index_map(program);
  });
}

template <typename Real>
auto async_csg_domains(wasm_csg_graph<Real> &g, emscripten::val js_program,
                       int config) -> promise_t {
  auto program = extract_int_vector(js_program);
  return promise([g = g, program = std::move(program), config]() {
    return g.domains(program, config);
  });
}

template <typename Real>
auto async_csg_domains_with_labels(wasm_csg_graph<Real> &g,
                                   emscripten::val js_program, int config)
    -> promise_t {
  auto program = extract_int_vector(js_program);
  return promise([g = g, program = std::move(program), config]() {
    return g.domains_with_labels(program, config);
  });
}

template <typename Real>
auto async_csg_domains_with_index_map(wasm_csg_graph<Real> &g,
                                      emscripten::val js_program, int config)
    -> promise_t {
  auto program = extract_int_vector(js_program);
  return promise([g = g, program = std::move(program), config]() {
    return g.domains_with_index_map(program, config);
  });
}

} // namespace ts
} // namespace tf
