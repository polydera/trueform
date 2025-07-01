#include <iostream>
#include <vector>
#include <array>

#include "trueform/core.hpp" 

// Helper to print a point
template <typename Point>
void print_point(const Point& p) {
    std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]";
}

// Helper to print the sidedness enum for better readability
std::ostream& operator<<(std::ostream& os, tf::sidedness s) {
    switch (s) {
        case tf::sidedness::on_positive_side: os << "On Positive Side"; break;
        case tf::sidedness::on_negative_side: os << "On Negative Side"; break;
        case tf::sidedness::on_boundary:      os << "On Boundary"; break;
    }
    return os;
}

// Helper to print the containment enum for better readability
std::ostream& operator<<(std::ostream& os, tf::containment c) {
    switch (c) {
        case tf::containment::inside:  os << "Inside"; break;
        case tf::containment::outside: os << "Outside"; break;
        case tf::containment::on_boundary: os << "On Boundary"; break;
    }
    return os;
}


int main() {
    // --- Define the vertices for three triangles ---

    // Triangle 0: A flat triangle on the XY plane.
    tf::point<float, 3> t0_p0 = {0.0f, 0.0f, 0.0f};
    tf::point<float, 3> t0_p1 = {3.0f, 0.0f, 0.0f};
    tf::point<float, 3> t0_p2 = {1.5f, 3.0f, 0.0f};

    // Triangle 1: A vertical triangle designed to pierce through Triangle 0.
    tf::point<float, 3> t1_p0 = {1.5f, -1.0f, -1.0f};
    tf::point<float, 3> t1_p1 = {1.5f, -1.0f,  2.0f};
    tf::point<float, 3> t1_p2 = {1.5f,  2.0f,  0.5f};

    // Triangle 2: A triangle located far away, guaranteed not to intersect.
    tf::point<float, 3> t2_p0 = {10.0f, 10.0f, 10.0f};
    tf::point<float, 3> t2_p1 = {13.0f, 10.0f, 10.0f};
    tf::point<float, 3> t2_p2 = {11.5f, 13.0f, 10.0f};
    
    // A segment that is close to Triangle 0 but not touching.
    tf::point<float, 3> s0_p0 = {0.0f, -2.0f, 0.0f};
    tf::point<float, 3> s0_p1 = {3.0f, -2.0f, 0.0f};


    // --- Create trueform polygon primitives ---

    // For this example, we create owning polygons directly from the points.
    // In a larger application, these would typically be non-owning views
    // into a larger mesh data structure.
    auto triangle0 = tf::make_polygon(std::array{t0_p0, t0_p1, t0_p2});
    auto triangle1 = tf::make_polygon(std::array{t1_p0, t1_p1, t1_p2});
    auto triangle2 = tf::make_polygon(std::array{t2_p0, t2_p1, t2_p2});
    auto segment0 = tf::make_segment_between_points(s0_p0, s0_p1);

    // --- Perform and Print Intersection Tests ---

    std::cout << std::boolalpha;
    std::cout << "--- Triangle Intersection Tests ---" << std::endl;

    // Test Triangle 0 vs Triangle 1 (should intersect)
    bool intersects_01 = tf::intersects(triangle0, triangle1);
    std::cout << "Triangle 0 intersects Triangle 1? " << intersects_01 << " (Expected: true)" << std::endl;

    // Test Triangle 0 vs Triangle 2 (should not intersect)
    bool intersects_02 = tf::intersects(triangle0, triangle2);
    std::cout << "Triangle 0 intersects Triangle 2? " << intersects_02 << " (Expected: false)" << std::endl;

    // Test Triangle 1 vs Triangle 2 (should not intersect)
    bool intersects_12 = tf::intersects(triangle1, triangle2);
    std::cout << "Triangle 1 intersects Triangle 2? " << intersects_12 << " (Expected: false)" << std::endl;
    
    std::cout << "\n--- Closest Point Tests ---" << std::endl;

    // --- Perform and Print Closest Point Tests ---

    // Test between two intersecting triangles (distance should be 0)
    auto [dist2_01, pt_on_0, pt_on_1] = tf::closest_metric_point_pair(triangle0, triangle1);
    std::cout << "Closest points between T0 and T1 (intersecting):" << std::endl;
    std::cout << "  - Squared Distance: " << dist2_01 << " (Expected: 0)" << std::endl;
    std::cout << "  - Point on T0: "; print_point(pt_on_0); std::cout << std::endl;
    std::cout << "  - Point on T1: "; print_point(pt_on_1); std::cout << std::endl;

    // Test between a triangle and a non-intersecting segment
    auto [dist2_0s, pt_on_t0, pt_on_s0] = tf::closest_metric_point_pair(triangle0, segment0);
    std::cout << "\nClosest points between T0 and Segment0 (non-intersecting):" << std::endl;
    std::cout << "  - Squared Distance: " << dist2_0s << " (Expected: 4)" << std::endl;
    std::cout << "  - Point on T0: "; print_point(pt_on_t0); std::cout << " (Expected: [0, 0, 0] to [3, 0, 0])" << std::endl;
    std::cout << "  - Point on S0: "; print_point(pt_on_s0); std::cout << " (Expected: [0, -2, 0] to [3, -2, 0])" << std::endl;

    std::cout << "\n--- Distance and Classification Tests ---" << std::endl;
    
    // --- Perform and Print Distance and Classification Tests ---

    // Test distance between two non-intersecting triangles
    float dist_02 = tf::distance(triangle0, triangle2);
    std::cout << "Distance between T0 and T2: " << dist_02 << " (Expected: >0)" << std::endl;

    // Test point classification
    tf::point<float, 3> test_point_above = {1.5f, 1.0f, 5.0f}; // A point above the plane of T0
    tf::point<float, 3> test_point_inside = {1.5f, 1.0f, 0.0f}; // A point inside T0's projection

    auto plane_of_t0 = tf::make_plane(triangle0);

    // Classify against the infinite plane of the triangle
    auto sidedness_result = tf::classify(test_point_above, plane_of_t0);
    std::cout << "\nClassify point " << test_point_above[2] << " units above T0's plane: " << sidedness_result << std::endl;

    // Classify against the finite triangle itself
    auto containment_result = tf::classify(test_point_inside, triangle0);
    std::cout << "Classify point inside T0's boundary: " << containment_result << std::endl;

    std::cout << "\n--- 2D Sidedness Test ---" << std::endl;

    tf::point<float, 2> p_left = {1.0f, 2.0f};
    tf::point<float, 2> p_right = {2.0f, 1.0f};
    auto line_2d = tf::make_line_between_points(
        tf::point<float, 2>{0.0f, 0.0f}, 
        tf::point<float, 2>{5.0f, 5.0f}
    );

    auto sidedness_left = tf::classify(p_left, line_2d);
    std::cout << "Classify point [1, 2] against line from [0,0] to [5,5]: " << sidedness_left << " (Expected: On Positive Side)" << std::endl;

    auto sidedness_right = tf::classify(p_right, line_2d);
    std::cout << "Classify point [2, 1] against line from [0,0] to [5,5]: " << sidedness_right << " (Expected: On Negative Side)" << std::endl;

    std::cout << "\n--- Ray Casting Tests ---" << std::endl;

    // A ray that will hit triangle0
    auto hitting_ray = tf::make_ray(
        tf::point<float, 3>{1.5f, 1.0f, 5.0f}, 
        tf::vector<float, 3>{0.0f, 0.0f, -1.0f}
    );

    // A ray that will miss all triangles
    auto missing_ray = tf::make_ray(
        tf::point<float, 3>{10.0f, 0.0f, 5.0f}, 
        tf::vector<float, 3>{0.0f, 0.0f, -1.0f}
    );

    // Test tf::ray_cast (returns distance t)
    if (auto result = tf::ray_cast(hitting_ray, triangle0)) {
        auto [status, t] = result;
        std::cout << "ray_cast hit T0 at distance t = " << t << " (Expected: 5)" << std::endl;
    }

    // Test tf::ray_hit (returns distance t and hit point)
    if (auto result = tf::ray_hit(hitting_ray, triangle0)) {
        auto [status, t, hit_point] = result;
        std::cout << "ray_hit hit T0 at point: ";
        print_point(hit_point);
        std::cout << " (Expected: [1.5, 1, 0])" << std::endl;
    }

    // Test a ray that misses
    if (auto result = tf::ray_cast(missing_ray, triangle0)) {
        std::cout << "ERROR: Missing ray should not hit T0." << std::endl;
    } else {
        std::cout << "ray_cast correctly missed T0." << std::endl;
    }

    return 0;
}

