#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "trueform/trueform.hpp"

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Traits_2 = CGAL::Arr_segment_traits_2<Kernel>;
using Point_2 = Traits_2::Point_2;
using Segment_2 = Traits_2::X_monotone_curve_2;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;

auto run_tl(const std::vector<std::array<int, 2>> &edges,
            const std::vector<tf::point<double, 2>> &points) {

  tf::planar_arrangements<int, double> pa;
  pa.build(tf::make_segments(edges, points));
  return pa;
}

auto run_cgal(const std::vector<std::array<int, 2>> &edges,
              const std::vector<tf::point<double, 2>> &points) {
  std::map<tf::point<double, 2>, int> point_id_map;
  for (size_t i = 0; i < points.size(); ++i) {
    point_id_map[points[i]] = static_cast<int>(i);
  }

  Arrangement_2 arr;
  for (const auto &e : edges) {
    const Point_2 p1(points[e[0]][0], points[e[0]][1]);
    const Point_2 p2(points[e[1]][0], points[e[1]][1]);
    CGAL::insert(arr, Segment_2(p1, p2));
  }

  std::vector<std::vector<int>> faces;
  std::vector<std::vector<std::vector<int>>> holes_of_faces;

  for (auto fit = arr.faces_begin(); fit != arr.faces_end(); ++fit) {
    if (!fit->is_unbounded()) {
      faces.emplace_back();
      auto &face = faces.back();
      holes_of_faces.emplace_back();
      auto &holes = holes_of_faces.back();
      auto outer = fit->outer_ccb();
      auto curr = outer;
      do {
        Point_2 pt = curr->source()->point();
        auto it = point_id_map.find(tf::point<double, 2>{
            CGAL::to_double(pt.x()), CGAL::to_double(pt.y())});
        face.push_back(it->second);
        ++curr;
      } while (curr != outer);

      // Holes
      for (auto h_it = fit->inner_ccbs_begin(); h_it != fit->inner_ccbs_end();
           ++h_it) {
        holes.emplace_back();
        auto &hole = holes.back();
        auto circ = *h_it;
        auto curh = circ;
        do {
          Point_2 pt = curh->source()->point();
          auto it = point_id_map.find(tf::point<double, 2>{
              CGAL::to_double(pt.x()), CGAL::to_double(pt.y())});
          hole.push_back(it->second);
          ++curh;
        } while (curh != circ);
      }
    }
  }

  return std::make_pair(std::move(faces), std::move(holes_of_faces));
}

int dummy_out = 0;

void run_test(const std::vector<std::array<int, 2>> &edges,
              const std::vector<tf::point<double, 2>> &points, int n_iter,
              int duplicate_n) {
  std::vector<std::array<int, 2>> use_edges;
  std::vector<tf::point<double, 2>> use_points;
  int offset = points.size();
  for (int i = 0; i < duplicate_n; ++i) {
    for (auto edge : edges) {
      use_edges.push_back({edge[0] + i * offset, edge[1] + i * offset});
    }
    for (auto pt : points) {
      use_points.push_back(pt + tf::vector<double, 2>{2. * i, 2. * i});
    }
  }
  std::cout << "==== Test on " << use_edges.size() << " edges ===" << std::endl;
  float tf_time = std::numeric_limits<float>::max();
  float cgal_time = std::numeric_limits<float>::max();
  for (int i = 0; i < n_iter; ++i) {
    std::cout << "\rIteration: " << i << std::flush;
    tf::tick();
    auto out_tf = run_tl(use_edges, use_points);
    auto tf_tmp = tf::tock();
    tf_time = std::min(tf_time, tf_tmp);
    tf::tick();
    auto out_cgal = run_cgal(use_edges, use_points);
    auto cgal_tmp = tf::tock();
    cgal_time = std::min(cgal_time, cgal_tmp);

    std::cout << " :: tf: " << tf_tmp << " ms, cgal: " << cgal_tmp
              << " ms      " << std::flush;

    dummy_out += out_tf.faces().size() + out_tf.holes_for_faces().size();
    dummy_out += out_cgal.first.size() + out_cgal.second.size();
  }
  std::cout
      << "\r                                                                "
      << std::flush;
  std::cout << "\r     tf: " << tf_time << " ms" << std::endl;
  std::cout << "   cgal: " << cgal_time << " ms" << std::endl;
  std::cout << "speedup: " << (cgal_time / tf_time) << std::endl;
}

int main() {
  std::vector<int> base_loop{{0, 1, 2, 3, 13, 4, 5, 6}};
  std::vector<std::array<int, 2>> edges{{1, 7},   {7, 8},   {8, 6},   {1, 9},
                                        {9, 5},   {10, 11}, {11, 12}, {12, 10},
                                        {13, 14}, {13, 15}, {16, 17}, {17, 18}};
  std::vector<tf::point<double, 2>> points{
      {0.f, 0.f},
      {0.2f, 0.f},
      {1.f, 0.f},
      {1.f, 1.f},
      {0.f, 1.f},
      {0.f, 0.9f}, // 2
      {0.f, 0.8f},
      //
      {0.2f, 0.5f},
      {0.2f, 0.8f},
      //
      {0.3f, 0.9f},
      //
      {0.7f, 0.5f},
      {0.7f, 0.3f},
      {0.6f, 0.4f},
      //
      {0.8f, 1.f},
      {0.7f, 0.9f},
      {0.9f, 0.9f},
      //
      {0.1f, 0.9f},
      {0.2f, 0.9f},
      {0.25f, 0.9f},
  };

  int prev = base_loop.size() - 1;
  for (int i = 0; i < int(base_loop.size()); prev = i++)
    edges.push_back({base_loop[prev], base_loop[i]});


  std::cout << "---- Benchmarking ----" << std::endl;
  for (int duplicate_n : {10, 100, 250, 500, 1000, 2000})
    run_test(edges, points, 10, duplicate_n);
  return dummy_out;
}
