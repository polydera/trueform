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

/** @defgroup core Core Module
 *  Core geometric primitives, ranges, queries, and algorithms.
 */

/** @defgroup core_primitives Primitives
 *  @ingroup core
 *  Points, vectors, lines, rays, planes, segments, polygons, and bounding
 * volumes.
 */

/** @defgroup core_queries Queries
 *  @ingroup core
 *  Distance, intersection, classification, and ray casting operations.
 */

/** @defgroup core_ranges Ranges
 *  @ingroup core
 *  Primitive ranges and view adaptors for working with collections.
 */

/** @defgroup core_properties Properties
 *  @ingroup core
 *  Geometric properties: area, volume, normals, centroids, covariance.
 */

/** @defgroup core_buffers Buffers
 *  @ingroup core
 *  Data structures and owning buffers for geometric data.
 */

/** @defgroup core_policies Policies
 *  @ingroup core
 *  Policy decorators for adding metadata to primitives and ranges.
 */

/** @defgroup core_algorithms Algorithms
 *  @ingroup core
 *  Parallel algorithms for geometric computation.
 */

#include "./core/aabb.hpp"                      // IWYU pragma: export
#include "./core/aabb_from.hpp"                 // IWYU pragma: export
#include "./core/aabb_like.hpp"                 // IWYU pragma: export
#include "./core/aabb_metrics.hpp"              // IWYU pragma: export
#include "./core/aabb_union.hpp"                // IWYU pragma: export
#include "./core/algorithm.hpp"                 // IWYU pragma: export
#include "./core/allocate.hpp"                  // IWYU pragma: export
#include "./core/angle.hpp"                     // IWYU pragma: export
#include "./core/area.hpp"                      // IWYU pragma: export
#include "./core/array_hash.hpp"                // IWYU pragma: export
#include "./core/axis.hpp"                      // IWYU pragma: export
#include "./core/basis.hpp"                     // IWYU pragma: export
#include "./core/blocked_buffer.hpp"            // IWYU pragma: export
#include "./core/buffer.hpp"                    // IWYU pragma: export
#include "./core/centroid.hpp"                  // IWYU pragma: export
#include "./core/checked.hpp"                   // IWYU pragma: export
#include "./core/classify.hpp"                  // IWYU pragma: export
#include "./core/closest_metric_point.hpp"      // IWYU pragma: export
#include "./core/closest_metric_point_pair.hpp" // IWYU pragma: export
#include "./core/closest_point_on_triangle.hpp" // IWYU pragma: export
#include "./core/closest_point_parametric.hpp"  // IWYU pragma: export
#include "./core/concatenated_blocked_range_collections.hpp" // IWYU pragma: export
#include "./core/concatenated_blocked_ranges.hpp" // IWYU pragma: export
#include "./core/constants.hpp"                   // IWYU pragma: export
#include "./core/containment.hpp"                 // IWYU pragma: export
#include "./core/contains_coplanar_point.hpp"     // IWYU pragma: export
#include "./core/contains_point.hpp"              // IWYU pragma: export
#include "./core/contiguous_index_hash_map.hpp"   // IWYU pragma: export
#include "./core/coordinate_type.hpp"             // IWYU pragma: export
#include "./core/covariance_of.hpp"               // IWYU pragma: export
#include "./core/cross.hpp"                       // IWYU pragma: export
#include "./core/cross_covariance_of.hpp"         // IWYU pragma: export
#include "./core/curve.hpp"                       // IWYU pragma: export
#include "./core/curves.hpp"                      // IWYU pragma: export
#include "./core/curves_buffer.hpp"               // IWYU pragma: export
#include "./core/distance.hpp"                    // IWYU pragma: export
#include "./core/dot.hpp"                         // IWYU pragma: export
#include "./core/edges.hpp"                       // IWYU pragma: export
#include "./core/empty_aabb.hpp"                  // IWYU pragma: export
#include "./core/epsilon_inverse.hpp"             // IWYU pragma: export
#include "./core/faces.hpp"                       // IWYU pragma: export
#include "./core/frame.hpp"                       // IWYU pragma: export
#include "./core/frame_like.hpp"                  // IWYU pragma: export
#include "./core/frame_of.hpp"                    // IWYU pragma: export
#include "./core/frame_ptr.hpp"                   // IWYU pragma: export
#include "./core/hash_map.hpp"                    // IWYU pragma: export
#include "./core/hash_set.hpp"                    // IWYU pragma: export
#include "./core/index_hash_map.hpp"              // IWYU pragma: export
#include "./core/intersect_status.hpp"            // IWYU pragma: export
#include "./core/intersects.hpp"                  // IWYU pragma: export
#include "./core/interval.hpp"                    // IWYU pragma: export
#include "./core/inverted.hpp"                    // IWYU pragma: export
#include "./core/is_soup.hpp"                     // IWYU pragma: export
#include "./core/largest_axis.hpp"                // IWYU pragma: export
#include "./core/line.hpp"                        // IWYU pragma: export
#include "./core/line_like.hpp"                   // IWYU pragma: export
#include "./core/line_line_check.hpp"             // IWYU pragma: export
#include "./core/local_buffer.hpp"                // IWYU pragma: export
#include "./core/local_value.hpp"                 // IWYU pragma: export
#include "./core/local_vector.hpp"                // IWYU pragma: export
#include "./core/make_rotation.hpp"               // IWYU pragma: export
#include "./core/mean_edge_length.hpp"            // IWYU pragma: export
#include "./core/metric_point.hpp"                // IWYU pragma: export
#include "./core/metric_point_pair.hpp"           // IWYU pragma: export
#include "./core/minimal_maximal_distance.hpp"    // IWYU pragma: export
#include "./core/none.hpp"                        // IWYU pragma: export
#include "./core/normal.hpp"                      // IWYU pragma: export
#include "./core/normalize.hpp"                   // IWYU pragma: export
#include "./core/normalized.hpp"                  // IWYU pragma: export
#include "./core/obb.hpp"                         // IWYU pragma: export
#include "./core/obb_from.hpp"                    // IWYU pragma: export
#include "./core/obbrss.hpp"                      // IWYU pragma: export
#include "./core/obbrss_from.hpp"                 // IWYU pragma: export
#include "./core/offset_block_buffer.hpp"         // IWYU pragma: export
#include "./core/offset_block_vector.hpp"         // IWYU pragma: export
#include "./core/parallelogram_area.hpp"          // IWYU pragma: export
#include "./core/paths.hpp"                       // IWYU pragma: export
#include "./core/plane.hpp"                       // IWYU pragma: export
#include "./core/plane_like.hpp"                  // IWYU pragma: export
#include "./core/point.hpp"                       // IWYU pragma: export
#include "./core/point_like.hpp"                  // IWYU pragma: export
#include "./core/point_view.hpp"                  // IWYU pragma: export
#include "./core/points.hpp"                      // IWYU pragma: export
#include "./core/points_buffer.hpp"               // IWYU pragma: export
#include "./core/policy.hpp"                      // IWYU pragma: export
#include "./core/polygon.hpp"                     // IWYU pragma: export
#include "./core/polygons.hpp"                    // IWYU pragma: export
#include "./core/polygons_buffer.hpp"             // IWYU pragma: export
#include "./core/projector.hpp"                   // IWYU pragma: export
#include "./core/range.hpp"                       // IWYU pragma: export
#include "./core/ray.hpp"                         // IWYU pragma: export
#include "./core/ray_aabb_check.hpp"              // IWYU pragma: export
#include "./core/ray_cast.hpp"                    // IWYU pragma: export
#include "./core/ray_cast_info.hpp"               // IWYU pragma: export
#include "./core/ray_config.hpp"                  // IWYU pragma: export
#include "./core/ray_hit.hpp"                     // IWYU pragma: export
#include "./core/ray_hit_info.hpp"                // IWYU pragma: export
#include "./core/ray_like.hpp"                    // IWYU pragma: export
#include "./core/reallocate.hpp"                  // IWYU pragma: export
#include "./core/rss.hpp"                         // IWYU pragma: export
#include "./core/rss_from.hpp"                    // IWYU pragma: export
#include "./core/segment.hpp"                     // IWYU pragma: export
#include "./core/segments.hpp"                    // IWYU pragma: export
#include "./core/segments_buffer.hpp"             // IWYU pragma: export
#include "./core/sidedness.hpp"                   // IWYU pragma: export
#include "./core/signed_volume.hpp"               // IWYU pragma: export
#include "./core/small_vector.hpp"                // IWYU pragma: export
#include "./core/sphere.hpp"                      // IWYU pragma: export
#include "./core/sphere_like.hpp"                 // IWYU pragma: export
#include "./core/sqrt.hpp"                        // IWYU pragma: export
#include "./core/static_size.hpp"                 // IWYU pragma: export
#include "./core/stitch_index_maps.hpp"           // IWYU pragma: export
#include "./core/svd_of_symmetric.hpp"            // IWYU pragma: export
#include "./core/tick_tock.hpp"                   // IWYU pragma: export
#include "./core/transformation.hpp"              // IWYU pragma: export
#include "./core/transformation_like.hpp"         // IWYU pragma: export
#include "./core/transformation_view.hpp"         // IWYU pragma: export
#include "./core/transformed.hpp"                 // IWYU pragma: export
#include "./core/tuple.hpp"                       // IWYU pragma: export
#include "./core/tuple_hash.hpp"                  // IWYU pragma: export
#include "./core/unit_vector.hpp"                 // IWYU pragma: export
#include "./core/unit_vector_like.hpp"            // IWYU pragma: export
#include "./core/unit_vector_view.hpp"            // IWYU pragma: export
#include "./core/unit_vectors.hpp"                // IWYU pragma: export
#include "./core/unit_vectors_buffer.hpp"         // IWYU pragma: export
#include "./core/unsafe.hpp"                      // IWYU pragma: export
#include "./core/vector.hpp"                      // IWYU pragma: export
#include "./core/vector_like.hpp"                 // IWYU pragma: export
#include "./core/vector_view.hpp"                 // IWYU pragma: export
#include "./core/vectors.hpp"                     // IWYU pragma: export
#include "./core/vectors_buffer.hpp"              // IWYU pragma: export
#include "./core/view.hpp"                        // IWYU pragma: export
#include "./core/views.hpp"                       // IWYU pragma: export
#include "./core/wedge.hpp"                       // IWYU pragma: export
#include "./core/zero.hpp"                        // IWYU pragma: export
#include "./core/zip_apply.hpp"                   // IWYU pragma: export
#include "./core/zip_range.hpp"                   // IWYU pragma: export
