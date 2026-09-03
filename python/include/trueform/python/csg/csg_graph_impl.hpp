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
#include "../arrangement/arrangement_builders.hpp"
#include "../intersect/build_intersect_structures.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include "./csg_builders.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/algorithm/parallel_transform.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/csg/expression.hpp>
#include <trueform/csg/expression/selection.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/csg/make_csg_domains.hpp>
#include <trueform/csg/make_csg_mesh.hpp>
#include <trueform/csg/make_intersection_curves.hpp>
#include <trueform/csg/make_outer_shell.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/triangulation_type.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tf::py {

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

// A selection crosses as the expression program, an optional list of
// operand tags and the selection kind as an int: absent restriction =
// every form's surface; present = only those forms' faces reach the
// output, and an
// empty program is no expression at all (the embedded surface read).
// Kind 0 is the boundary read, 1 the inside read, which has no meaning
// without a region to be inside of.
inline auto decode_selection(const std::vector<int> &program,
                             const std::optional<std::vector<int>> &tags,
                             int kind) -> tf::csg::selection_t {
  if (kind == 1) {
    if (program.empty())
      throw std::runtime_error("an inside read requires an expression");
    return tf::csg::inside(tags ? *tags : std::vector<int>{},
                           decode_expr(program));
  }
  if (kind != 0)
    throw std::runtime_error("unknown csg selection kind");
  if (!tags)
    return tf::csg::selection_t(decode_expr(program));
  if (program.empty())
    return tf::csg::selection(*tags);
  return tf::csg::selection(*tags, decode_expr(program));
}

/// The sealed engine behind the Python CsgGraph: owns the input mesh
/// wrappers (their stored ndarrays keep numpy alive), the tagged forms
/// over them, and the tf::csg_graph built once at construction. The
/// Python facade holds the user-facing state (forms list, sheets,
/// config); only evaluation methods and created_points cross here.
template <typename Index, typename RealT, std::size_t Ngon>
class csg_graph_wrapper {
  using wrapper_t = mesh_wrapper<Index, RealT, Ngon, 3>;
  using form_type = form_t<Index, RealT, Ngon, 3>;
  using forms_type = forms_range_t<Index, RealT, Ngon, 3>;
  using graph_t = range_csg_graph_t<forms_type>;

  static auto make_forms(std::vector<wrapper_t> &wrappers)
      -> std::vector<form_type> {
    build_intersect_structures_all(wrappers);
    return tagged_forms(wrappers);
  }

public:
  csg_graph_wrapper(std::vector<wrapper_t> wrappers, std::vector<int> sheets,
                    int mode, double tolerance, int triangulation)
      : _wrappers(std::move(wrappers)), _forms(make_forms(_wrappers)),
        _graph(build_range_csg_graph(
            tf::make_range(_forms.data(), _forms.size()),
            tf::make_range(static_cast<const int *>(sheets.data()),
                           sheets.size()),
            tf::arrangement_config{
                tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                     tolerance},
                static_cast<tf::triangulation_type>(triangulation)})) {}

  auto created_points() {
    const auto &created = _graph.created_points();
    tf::points_buffer<RealT, 3> out;
    out.allocate(created.size());
    const auto &conv = _graph.converter();
    tf::parallel_transform(tf::make_range(created), out.points(),
                           [&](const auto &ip) {
                             auto p = conv.deconvert(ip);
                             return tf::point<RealT, 3>{RealT(p[0]),
                                                        RealT(p[1]),
                                                        RealT(p[2])};
                           });
    return make_numpy_array(std::move(out));
  }

  auto intersection_curves() {
    auto curves = tf::make_intersection_curves(_graph);
    auto [paths, pts] = make_numpy_array(std::move(curves));
    return nanobind::make_tuple(
        nanobind::make_tuple(paths.first, paths.second), std::move(pts));
  }

  auto outer_shell() {
    auto shell = tf::make_outer_shell(_graph);
    return make_numpy_array(std::move(shell));
  }

  auto mesh(const std::vector<int> &program,
            const std::optional<std::vector<int>> &selection, int kind) {
    if (program.empty() && !selection && kind == 0) {
      auto result = tf::make_csg_mesh(_graph);
      return make_numpy_array(std::move(result));
    }
    auto result =
        tf::make_csg_mesh(_graph, decode_selection(program, selection, kind));
    return make_numpy_array(std::move(result));
  }

  auto mesh_with_labels(const std::vector<int> &program,
                        const std::optional<std::vector<int>> &selection,
                        int kind) {
    auto pack = [](auto &&result, auto &&tag_labels, auto &&face_labels) {
      return nanobind::make_tuple(make_numpy_array(std::move(result)),
                                  make_numpy_array(std::move(tag_labels)),
                                  make_numpy_array(std::move(face_labels)));
    };
    if (program.empty() && !selection && kind == 0) {
      auto [result, tag_labels, face_labels] =
          tf::make_csg_mesh(_graph, tf::return_source_ids);
      return pack(std::move(result), std::move(tag_labels),
                  std::move(face_labels));
    }
    auto [result, tag_labels, face_labels] =
        tf::make_csg_mesh(_graph, decode_selection(program, selection, kind),
                          tf::return_source_ids);
    return pack(std::move(result), std::move(tag_labels),
                std::move(face_labels));
  }

  auto mesh_with_index_map(const std::vector<int> &program,
                           const std::optional<std::vector<int>> &selection,
                           int kind) {
    if (program.empty() && !selection && kind == 0)
      throw std::runtime_error(
          "the full-arrangement mesh has no index-map form; pass an "
          "expression or a selection");
    auto [result, imap] =
        tf::make_csg_mesh(_graph, decode_selection(program, selection, kind),
                          tf::return_index_map);
    auto point_f = make_numpy_array(std::move(imap.point_f));
    return nanobind::make_tuple(
        make_numpy_array(std::move(result)),
        make_numpy_array(std::move(imap.point_tag_labels)),
        make_numpy_array(std::move(imap.point_labels)),
        make_numpy_array(std::move(imap.face_tag_labels)),
        make_numpy_array(std::move(imap.face_labels)),
        nanobind::make_tuple(point_f.first, point_f.second),
        make_numpy_array(std::move(imap.uncut_faces)),
        imap.n_original_points, imap.n_tags, imap.n_output_points);
  }

  auto
  domains_with_index_map(const std::vector<int> &program, int config,
                         const std::optional<std::vector<int>> &selection) {
    auto run = [&](auto &&...expr_arg) {
      auto [cells, ids, imap] = tf::make_csg_domains(
          _graph, expr_arg..., static_cast<tf::domain_config>(config),
          tf::return_index_map);
      nanobind::list out;
      for (auto &cell : cells)
        out.append(make_numpy_array(std::move(cell)));
      auto ftb = make_numpy_array(std::move(imap.face_tag_blocks));
      auto fb = make_numpy_array(std::move(imap.face_blocks));
      auto ptb = make_numpy_array(std::move(imap.point_tag_blocks));
      auto pb = make_numpy_array(std::move(imap.point_blocks));
      return nanobind::make_tuple(
          std::move(out), make_numpy_array(std::move(ids)),
          nanobind::make_tuple(ftb.first, ftb.second),
          nanobind::make_tuple(fb.first, fb.second),
          nanobind::make_tuple(ptb.first, ptb.second),
          nanobind::make_tuple(pb.first, pb.second), imap.n_original_points,
          imap.n_tags, imap.n_output_points,
          make_numpy_array(std::move(imap.inclusion)));
    };
    if (program.empty() && !selection)
      return run();
    return run(decode_selection(program, selection, 0));
  }

  auto domains(const std::vector<int> &program, int config,
               const std::optional<std::vector<int>> &selection) {
    auto run = [&](auto &&...expr_arg) {
      auto [cells, ids] =
          tf::make_csg_domains(_graph, expr_arg...,
                               static_cast<tf::domain_config>(config));
      nanobind::list out;
      for (auto &cell : cells)
        out.append(make_numpy_array(std::move(cell)));
      return nanobind::make_tuple(std::move(out),
                                  make_numpy_array(std::move(ids)));
    };
    if (program.empty() && !selection)
      return run();
    return run(decode_selection(program, selection, 0));
  }

  auto domains_with_labels(const std::vector<int> &program, int config,
                           const std::optional<std::vector<int>> &selection) {
    auto run = [&](auto &&...expr_arg) {
      auto [cells, ids, tag_blocks, face_blocks] = tf::make_csg_domains(
          _graph, expr_arg..., static_cast<tf::domain_config>(config),
          tf::return_source_ids);
      nanobind::list out;
      for (auto &cell : cells)
        out.append(make_numpy_array(std::move(cell)));
      auto tags = make_numpy_array(std::move(tag_blocks));
      auto faces = make_numpy_array(std::move(face_blocks));
      return nanobind::make_tuple(
          std::move(out), make_numpy_array(std::move(ids)),
          nanobind::make_tuple(tags.first, tags.second),
          nanobind::make_tuple(faces.first, faces.second));
    };
    if (program.empty() && !selection)
      return run();
    return run(decode_selection(program, selection, 0));
  }

private:
  std::vector<wrapper_t> _wrappers;
  std::vector<form_type> _forms;
  graph_t _graph;
};

template <typename Index, typename RealT, std::size_t Ngon>
auto register_csg_graph(nanobind::module_ &m, const char *name) -> void {
  using G = csg_graph_wrapper<Index, RealT, Ngon>;
  using W = mesh_wrapper<Index, RealT, Ngon, 3>;
  nanobind::class_<G>(m, name)
      .def(nanobind::init<std::vector<W>, std::vector<int>, int, double,
                          int>(),
           nanobind::arg("meshes"), nanobind::arg("sheets"),
           nanobind::arg("mode"), nanobind::arg("tolerance"),
           nanobind::arg("triangulation"))
      .def("created_points", &G::created_points)
      .def("intersection_curves", &G::intersection_curves)
      .def("outer_shell", &G::outer_shell)
      .def("mesh", &G::mesh, nanobind::arg("program"),
           nanobind::arg("selection").none() = nanobind::none(),
           nanobind::arg("kind") = 0)
      .def("mesh_with_labels", &G::mesh_with_labels,
           nanobind::arg("program"),
           nanobind::arg("selection").none() = nanobind::none(),
           nanobind::arg("kind") = 0)
      .def("mesh_with_index_map", &G::mesh_with_index_map,
           nanobind::arg("program"),
           nanobind::arg("selection").none() = nanobind::none(),
           nanobind::arg("kind") = 0)
      .def("domains_with_index_map", &G::domains_with_index_map,
           nanobind::arg("program"), nanobind::arg("config"),
           nanobind::arg("selection").none() = nanobind::none())
      .def("domains", &G::domains, nanobind::arg("program"),
           nanobind::arg("config"),
           nanobind::arg("selection").none() = nanobind::none())
      .def("domains_with_labels", &G::domains_with_labels,
           nanobind::arg("program"), nanobind::arg("config"),
           nanobind::arg("selection").none() = nanobind::none());
}

} // namespace tf::py
