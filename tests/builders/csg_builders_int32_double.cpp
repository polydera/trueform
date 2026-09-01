/**
 * @file csg_builders_int32_double.cpp
 * @brief The compiled classification and read tier for int32 indices, double
 * coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "arrangement_readers_impl.hpp"
#include "csg_builders_impl.hpp"
#include "csg_readers_impl.hpp"

#include <cstdint>

namespace tf::test {
namespace {
using tri = form_t<std::int32_t, double, 3>;
using tri_forms = forms_range_t<std::int32_t, double, 3>;
using tri_self_graph = self_csg_graph_t<tri>;
using tri_range_graph = range_csg_graph_t<tri_forms>;
using quad_forms = forms_range_t<std::int32_t, double, 4>;
using quad_range_graph = range_csg_graph_t<quad_forms>;
} // namespace

template tri_self_graph build_self_csg_graph<tri>(const tri &,
                                                  tf::arrangement_config);

template tri_range_graph
    build_range_csg_graph<tri_forms>(tri_forms,
                                     tf::range<const int *, tf::dynamic_size>,
                                     tf::arrangement_config);

template csg_domains_t<tri_self_graph>
csg_domains_of<tri_self_graph>(const tri_self_graph &);

template csg_domains_t<tri_self_graph>
csg_domains_of<tri_self_graph>(const tri_self_graph &, tf::domain_config);

template csg_domains_t<tri_self_graph>
csg_domains_of<tri_self_graph>(const tri_self_graph &,
                               const tf::csg::selection_t &);

template csg_domains_t<tri_self_graph>
csg_domains_of<tri_self_graph>(const tri_self_graph &,
                               const tf::csg::selection_t &, tf::domain_config);

template csg_mesh_t<tri_self_graph>
outer_shell_of<tri_self_graph>(const tri_self_graph &);

template arrangement_curves_t<tri_self_graph>
arrangement_curves_of<tri_self_graph>(const tri_self_graph &);

template csg_domains_t<tri_range_graph>
csg_domains_of<tri_range_graph>(const tri_range_graph &);

template csg_domains_t<tri_range_graph>
csg_domains_of<tri_range_graph>(const tri_range_graph &, tf::domain_config);

template csg_domains_t<tri_range_graph>
csg_domains_of<tri_range_graph>(const tri_range_graph &,
                                const tf::csg::selection_t &);

template csg_domains_t<tri_range_graph> csg_domains_of<tri_range_graph>(
    const tri_range_graph &, const tf::csg::selection_t &, tf::domain_config);

template csg_mesh_t<tri_range_graph>
outer_shell_of<tri_range_graph>(const tri_range_graph &);

template arrangement_curves_t<tri_range_graph>
arrangement_curves_of<tri_range_graph>(const tri_range_graph &);

template csg_mesh_t<tri_range_graph>
csg_mesh_of<tri_range_graph>(const tri_range_graph &);

template csg_mesh_t<tri_range_graph>
csg_mesh_of<tri_range_graph>(const tri_range_graph &,
                             const tf::csg::selection_t &);

template std::tuple<csg_mesh_t<tri_range_graph>, csg_labels_t<tri_range_graph>,
                    csg_labels_t<tri_range_graph>>
csg_mesh_with_source_ids_of<tri_range_graph>(const tri_range_graph &);

template std::tuple<csg_mesh_t<tri_range_graph>, csg_labels_t<tri_range_graph>,
                    csg_labels_t<tri_range_graph>>
csg_mesh_with_source_ids_of<tri_range_graph>(const tri_range_graph &,
                                             const tf::csg::selection_t &);

template std::tuple<csg_mesh_t<tri_range_graph>,
                    csg_index_map_t<tri_range_graph>>
csg_mesh_with_index_map_of<tri_range_graph>(const tri_range_graph &);

template std::tuple<csg_mesh_t<tri_range_graph>,
                    csg_index_map_t<tri_range_graph>>
csg_mesh_with_index_map_of<tri_range_graph>(const tri_range_graph &,
                                            const tf::csg::selection_t &);

template quad_range_graph
    build_range_csg_graph<quad_forms>(quad_forms,
                                      tf::range<const int *, tf::dynamic_size>,
                                      tf::arrangement_config);

template csg_domains_t<quad_range_graph>
csg_domains_of<quad_range_graph>(const quad_range_graph &);

} // namespace tf::test
