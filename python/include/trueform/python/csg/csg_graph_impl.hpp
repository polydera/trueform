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
#include "../intersect/build_intersect_structures.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/algorithm/parallel_transform.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/csg/csg_graph.hpp>
#include <trueform/csg/expression.hpp>
#include <trueform/cut/arrangement_config.hpp>
#include <trueform/csg/make_csg_domains.hpp>
#include <trueform/csg/make_csg_mesh.hpp>
#include <trueform/csg/make_intersection_curves.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/triangulation_type.hpp>

#include <cstdint>
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

/// The sealed engine behind the Python CsgGraph: owns the input mesh
/// wrappers (their stored ndarrays keep numpy alive), the tagged forms
/// over them, and the tf::csg_graph built once at construction. The
/// Python facade holds the user-facing state (forms list, sheets,
/// config); only evaluation methods and created_points cross here.
template <typename Index, typename RealT> class csg_graph_wrapper {
  using wrapper_t = mesh_wrapper<Index, RealT, 3, 3>;

  static auto make_form(wrapper_t &w) {
    auto fm = w.face_membership();
    auto mel = w.manifold_edge_link();
    tf::transformation<RealT, 3> tv =
        w.has_transformation()
            ? tf::transformation<RealT, 3>(w.transformation_view())
            : tf::make_identity_transformation<RealT, 3>();
    return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
           tf::tag(mel) | tf::tag(tv);
  }

  using form_t = decltype(make_form(std::declval<wrapper_t &>()));
  using forms_range_t =
      decltype(tf::make_range(std::declval<std::vector<form_t> &>()));
  using graph_t = tf::csg_graph<forms_range_t, tf::none_t>;

  static auto make_forms(std::vector<wrapper_t> &wrappers)
      -> std::vector<form_t> {
    build_intersect_structures_all(wrappers);
    std::vector<form_t> forms;
    forms.reserve(wrappers.size());
    for (auto &w : wrappers)
      forms.push_back(make_form(w));
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

public:
  csg_graph_wrapper(std::vector<wrapper_t> wrappers, std::vector<int> sheets,
                    int mode, double tolerance, int triangulation)
      : _wrappers(std::move(wrappers)), _forms(make_forms(_wrappers)),
        _graph(tf::make_range(_forms), {},
               tf::arrangement_config{
                   tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                        tolerance},
                   static_cast<tf::triangulation_type>(triangulation)},
               make_is_sheet(sheets, _wrappers.size())) {}

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

  auto mesh(const std::vector<int> &program) {
    if (program.empty()) {
      auto result = tf::make_csg_mesh(_graph);
      return make_numpy_array(std::move(result));
    }
    auto result = tf::make_csg_mesh(_graph, decode_expr(program));
    return make_numpy_array(std::move(result));
  }

  auto mesh_with_labels(const std::vector<int> &program) {
    auto pack = [](auto &&result, auto &&tag_labels, auto &&face_labels) {
      return nanobind::make_tuple(make_numpy_array(std::move(result)),
                                  make_numpy_array(std::move(tag_labels)),
                                  make_numpy_array(std::move(face_labels)));
    };
    if (program.empty()) {
      auto [result, tag_labels, face_labels] =
          tf::make_csg_mesh(_graph, tf::return_source_ids);
      return pack(std::move(result), std::move(tag_labels),
                  std::move(face_labels));
    }
    auto [result, tag_labels, face_labels] = tf::make_csg_mesh(
        _graph, decode_expr(program), tf::return_source_ids);
    return pack(std::move(result), std::move(tag_labels),
                std::move(face_labels));
  }

  auto mesh_with_index_map(const std::vector<int> &program) {
    if (program.empty())
      throw std::runtime_error(
          "the full-arrangement mesh has no index-map form; pass an "
          "expression");
    auto [result, imap] = tf::make_csg_mesh(_graph, decode_expr(program),
                                            tf::return_index_map);
    auto point_f = make_numpy_array(std::move(imap.point_f));
    return nanobind::make_tuple(
        make_numpy_array(std::move(result)),
        make_numpy_array(std::move(imap.point_tag_labels)),
        make_numpy_array(std::move(imap.point_labels)),
        make_numpy_array(std::move(imap.face_tag_labels)),
        make_numpy_array(std::move(imap.face_labels)),
        nanobind::make_tuple(point_f.first, point_f.second),
        imap.n_original_points, imap.n_original_faces, imap.n_tags,
        imap.n_output_points);
  }

  auto domains_with_index_map(const std::vector<int> &program, int config) {
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
    if (program.empty())
      return run();
    return run(decode_expr(program));
  }

  auto domains(const std::vector<int> &program, int config) {
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
    if (program.empty())
      return run();
    return run(decode_expr(program));
  }

  auto domains_with_labels(const std::vector<int> &program, int config) {
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
    if (program.empty())
      return run();
    return run(decode_expr(program));
  }

private:
  std::vector<wrapper_t> _wrappers;
  std::vector<form_t> _forms;
  graph_t _graph;
};

template <typename Index, typename RealT>
auto register_csg_graph(nanobind::module_ &m, const char *name) -> void {
  using G = csg_graph_wrapper<Index, RealT>;
  using W = mesh_wrapper<Index, RealT, 3, 3>;
  nanobind::class_<G>(m, name)
      .def(nanobind::init<std::vector<W>, std::vector<int>, int, double,
                          int>(),
           nanobind::arg("meshes"), nanobind::arg("sheets"),
           nanobind::arg("mode"), nanobind::arg("tolerance"),
           nanobind::arg("triangulation"))
      .def("created_points", &G::created_points)
      .def("intersection_curves", &G::intersection_curves)
      .def("mesh", &G::mesh, nanobind::arg("program"))
      .def("mesh_with_labels", &G::mesh_with_labels,
           nanobind::arg("program"))
      .def("mesh_with_index_map", &G::mesh_with_index_map,
           nanobind::arg("program"))
      .def("domains_with_index_map", &G::domains_with_index_map,
           nanobind::arg("program"), nanobind::arg("config"))
      .def("domains", &G::domains, nanobind::arg("program"),
           nanobind::arg("config"))
      .def("domains_with_labels", &G::domains_with_labels,
           nanobind::arg("program"), nanobind::arg("config"));
}

} // namespace tf::py
