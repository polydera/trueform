#include "trueform/blocked_range.hpp"
#include "trueform/intersects.hpp"
#include "trueform/polygon.hpp"
#include "trueform/polygons.hpp"
#include "trueform/segment.hpp"
#include <iostream>
#include <ostream>
#include <vector>

template <class> class print_t;
int main() {
  std::vector<tf::point<float, 3>> pts{{{{-2, -2, 0}},
                                        {{2, -2, 0}},
                                        {{0, 2, 0}},
                                        {{-1, 0, -1}},
                                        {{1, 0, -1}},
                                        {{0, 0, 1}},
                                        {{-1, 10, -1}},
                                        {{1, 10, -1}},
                                        {{0, 10, 1}}}};
  std::vector<int> ids{{0, 1, 2, 3, 4, 5, 6, 7, 8}};
  auto polygons = tf::make_polygons(tf::make_blocked_range<3>(ids), pts);

  auto polygon0 = polygons[0];
  auto polygon1 = polygons[1];
  auto polygon2 = polygons[2];

  std::cout << "polygon0 and polygon1 intersect: "
            << tf::intersects(polygon0, polygon1) << std::endl;
  std::cout << "polygon1 and polygon2 intersect: "
            << tf::intersects(polygon1, polygon2) << std::endl;
  std::cout << "polygon2 and polygon0 intersect: "
            << tf::intersects(polygon2, polygon0) << std::endl;

  auto poly0_edge0 = tf::make_segment_between_points(polygon0[0], polygon0[1]);
  auto poly1_edge0 = tf::make_segment_between_points(polygon1[0], polygon1[1]);
  auto poly1_edge1 = tf::make_segment_between_points(polygon1[1], polygon1[2]);
  auto poly1_edge2 = tf::make_segment_between_points(polygon1[2], polygon1[1]);

  std::cout << "poly0_edge0 and poly1_edge1: "
            << tf::intersects(poly0_edge0, poly1_edge1) << std::endl;
  std::cout << "poly1_edge0 and polygon0: "
            << tf::intersects(poly1_edge0, polygon0) << std::endl;
  std::cout << "poly1_edge1 and polygon0: "
            << tf::intersects(poly1_edge1, polygon0) << std::endl;
  std::cout << "poly1_edge2 and polygon0: "
            << tf::intersects(poly1_edge2, polygon0) << std::endl;
}
