#include "../util/read_mesh.hpp"
#include "./nanoflann.hpp"
#include "trueform/spatial.hpp"
#include "trueform/core.hpp"
#include "trueform/random.hpp"
#include <filesystem>
#include <vector>

struct PointCloud {
  std::vector<tf::point<float, 3>> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    return pts[idx][dim];
  }

  // Optional: bounding box computation
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

using namespace nanoflann;

using KDTree = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, PointCloud>,
                                        PointCloud, 3 /* dimensionality */
                                        >;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input.obj>\n";
    return 1;
  }
  std::filesystem::path path{argv[1]};
  if (!path.has_extension() || path.extension() != ".obj") {
    std::cerr << "Need an .obj file." << std::endl;
    return 0;
  }

  std::cout << "Reading file: " << path.filename() << std::endl;
  auto [raw_points, raw_triangle_faces] = tf::examples::read_mesh(path.c_str());
  // create a point range from std::vector<float>
  std::cout << "  Contains " << (raw_points.size() / 3) << " points."
            << std::endl;
  std::cout << "----------------------------------" << std::endl;

  auto points = tf::make_points<3>(raw_points);
  PointCloud cloud;
  for (auto p : points)
    cloud.pts.push_back(p);

  tf::tick();
  KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();
  tf::tock("Built nanoflann tree in ");

  tf::tree<int, float, 3> tree;
  tf::tick();
  tree.build(points, tf::config_tree(4, 4));
  tf::tock("Built trueform tree  in ");
  std::cout << "----------------------------------" << std::endl;

  // so the compiler does not optimize away
  int sum_for_return = 0;
  auto aabb = tf::aabb_from(tf::make_polygon(points));
  auto center = aabb.center();
  auto diagonal = aabb.diagonal().length() / 2;

  auto run_for_n = [&](int n, int n_iters) {
    std::cout << "knn(" << n << ") average over " << n_iters << " samples:\n";
    float nanoflann_time = 0;
    float tf_time = 0;

    size_t ret_index[10];
    float out_dist_sqr[10];
    std::array<tf::nearest_neighbor<int, float, 3>, 10> buff;

    for (int i = 0; i < n_iters; ++i) {

      auto pt = center + tf::random(0.f, 3 * diagonal) *
                             tf::normalized(tf::random_vector<float, 3>());

      nanoflann::KNNResultSet<float> resultSet(n); // 1 = nearest neighbor
      resultSet.init(&ret_index[0], &out_dist_sqr[0]);
      tf::tick();
      index.findNeighbors(resultSet, &pt[0], nanoflann::SearchParameters());
      nanoflann_time += tf::tock();
      sum_for_return += ret_index[n - 1];
      auto knn = tf::make_nearest_neighbors(buff.data(), n);
      tf::tick();
      tf::neighbor_search(tf::make_form(tree, points), pt, knn);
      tf_time += tf::tock();
      sum_for_return += buff[n - 1].element;
    }
    std::cout << "  nanoflann: " << (1000 * nanoflann_time / n_iters) << " mcs"
              << std::endl;
    std::cout << "  trueform : " << (1000 * tf_time / n_iters) << " mcs"
              << std::endl;
    std::cout << "  speed-up : " << (nanoflann_time / tf_time) << std::endl;
    std::cout << "----------------------------------" << std::endl;
  };

  for (int i = 1; i <= 10; ++i) {
    run_for_n(i, 10000);
  }

  return sum_for_return;
}
