// Sharp-feature quality harness: extract an isosurface from an analytic SDF
// and score it against exact ground truth.
//   ./eval_sharp [n=128] [shape] [method] [stabilizer]
//   shape:  the names volume_fields.hpp answers to
//   method: fe   landed tf::isosurface (flying edges)
//           dc   dual contouring, refinement off
//           dcn  dual contouring, refinement on
//
// Metrics per run:
//   surf_mean / surf_max   |sdf| at area-weighted face centroids
//   nrm_smooth / nrm_edge   mean normal error (deg) far from / near creases
//   edge_cd mean / max      analytic sharp-edge samples -> mesh, in spacing
//   units dihedral err            measured across-crease angle vs the analytic
//   one
#include "volume_fields.hpp"

#include <trueform/volume/make_isosurface.hpp>

#include <trueform/trueform.hpp>
#include <trueform/volume/make_isosurface.hpp>
#include <trueform/volume/volume.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace {

template <typename Shape>
auto run(const Shape &shape, int n, const char *name, const char *method,
         double stabilizer) -> void {
  const auto vol = tf::test::sampled_volume(shape, n);
  const float spacing = float(vol.spacing()[0]);

  const bool use_dc = std::strcmp(method, "dc") == 0;
  const bool use_dcn = std::strcmp(method, "dcn") == 0;
  const auto t0 = std::chrono::steady_clock::now();
  tf::isosurface_config config;
  config.stabilizer = stabilizer;
  if (use_dc || use_dcn) {
    config.method = tf::isosurface_method::dual_contouring;
    config.refine = !use_dc;
  }
  const auto mesh = tf::make_isosurface(vol.volume(), 0.f, config);
  const auto t1 = std::chrono::steady_clock::now();
  const double extract_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  auto run_metrics = [&](auto polygons) {
    {
      double v_sum = 0, v_max = 0;
      auto pts = polygons.points();
      for (std::size_t i = 0; i < pts.size(); ++i) {
        const float sd = std::abs(shape.sdf({pts[i][0], pts[i][1], pts[i][2]}));
        v_sum += sd;
        v_max = std::max(v_max, double(sd));
      }
      std::printf("  vertex |sdf| / spacing: mean %.4f max %.4f (%zu verts)\n",
                  v_sum / double(pts.size()) / spacing, v_max / spacing,
                  std::size_t(pts.size()));
    }
    {
      std::map<std::pair<int, int>, int> edge_deg;
      for (const auto &poly : polygons) {
        const int m = int(poly.size());
        for (int i = 0; i < m; ++i) {
          const int a = int(poly.indices()[i]);
          const int b = int(poly.indices()[(i + 1) % m]);
          ++edge_deg[{std::min(a, b), std::max(a, b)}];
        }
      }
      std::size_t deg1 = 0, deg2 = 0, deg_other = 0;
      for (const auto &[k, d] : edge_deg)
        (d == 2 ? deg2 : (d == 1 ? deg1 : deg_other))++;
      std::printf("  edges: deg2 %zu, boundary %zu, non-manifold %zu -> %s\n",
                  deg2, deg1, deg_other,
                  deg_other == 0
                      ? (deg1 == 0 ? "CLOSED MANIFOLD" : "manifold w/ boundary")
                      : "NON-MANIFOLD");
    }

    const auto edges = shape.edges();
    const auto band = shape.edges();
    const float edge_band = 1.5f * spacing;
    auto seg_dist = [&](const tf::test::volume_field::vec3 &p) {
      float best = 1e30f;
      for (const auto &e : band) {
        const float ab[3] = {e[1][0] - e[0][0], e[1][1] - e[0][1],
                             e[1][2] - e[0][2]};
        const float ap[3] = {p[0] - e[0][0], p[1] - e[0][1], p[2] - e[0][2]};
        float t = (ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2]) /
                  (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
        t = std::min(std::max(t, 0.f), 1.f);
        const float dx = ap[0] - t * ab[0], dy = ap[1] - t * ab[1],
                    dz = ap[2] - t * ab[2];
        best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz));
      }
      return best;
    };

    double surf_sum = 0, surf_wsum = 0, surf_max = 0;
    double nrm_smooth_sum = 0, nrm_smooth_w = 0;
    double nrm_edge_sum = 0, nrm_edge_w = 0;
    for (const auto &poly : polygons) {
      const tf::test::volume_field::vec3 centroid = {
          (poly[0][0] + poly[1][0] + poly[2][0]) / 3.f,
          (poly[0][1] + poly[1][1] + poly[2][1]) / 3.f,
          (poly[0][2] + poly[1][2] + poly[2][2]) / 3.f};
      const float area = tf::area(poly);
      const float sd = std::abs(shape.sdf(centroid));
      surf_sum += double(sd) * area;
      surf_wsum += area;
      surf_max = std::max(surf_max, double(sd));

      const auto nrm = tf::make_normal(poly[0], poly[1], poly[2]);
      const auto g = tf::test::volume_field::field_gradient(shape, centroid);
      const float dot = std::abs(nrm[0] * g[0] + nrm[1] * g[1] + nrm[2] * g[2]);
      const float angle = std::acos(std::min(dot, 1.f)) * 180.f / 3.14159265f;
      if (!band.empty() && seg_dist(centroid) < edge_band) {
        nrm_edge_sum += double(angle) * area;
        nrm_edge_w += area;
      } else {
        nrm_smooth_sum += double(angle) * area;
        nrm_smooth_w += area;
      }
    }

    double ecd_sum = 0, ecd_max = 0;
    std::size_t ecd_n = 0;
    double dih_sum = 0, dih_max = 0;
    std::size_t dih_n = 0;
    if (!edges.empty()) {
      tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
      auto form = polygons | tf::tag(tree);
      const int samples_per_edge = 256;
      const float delta = 1.5f * spacing;
      for (const auto &e : edges)
        for (int k = 0; k < samples_per_edge; ++k) {
          const float t = (k + 0.5f) / samples_per_edge;
          const tf::test::volume_field::vec3 p = {
              e[0][0] + t * (e[1][0] - e[0][0]),
              e[0][1] + t * (e[1][1] - e[0][1]),
              e[0][2] + t * (e[1][2] - e[0][2])};
          const float d = tf::distance(form, tf::make_point(p[0], p[1], p[2]));
          ecd_sum += d;
          ecd_max = std::max(ecd_max, double(d));
          ++ecd_n;

          if (k % 16 != 0)
            continue;
          // step off the crease along its two faces; compare the angle the
          // mesh forms across the crease with the angle the field states
          const auto g = tf::test::volume_field::field_gradient(shape, p);
          tf::test::volume_field::vec3 tangent = {
              e[1][0] - e[0][0], e[1][1] - e[0][1], e[1][2] - e[0][2]};
          const float tl =
              std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1] +
                        tangent[2] * tangent[2]);
          for (auto &v : tangent)
            v /= tl;
          const tf::test::volume_field::vec3 side = {
              g[1] * tangent[2] - g[2] * tangent[1],
              g[2] * tangent[0] - g[0] * tangent[2],
              g[0] * tangent[1] - g[1] * tangent[0]};
          tf::test::volume_field::vec3 mesh_n[2], true_n[2];
          for (int s = 0; s < 2; ++s) {
            const float sgn = s == 0 ? 1.f : -1.f;
            const tf::test::volume_field::vec3 off = {
                p[0] + sgn * delta * side[0], p[1] + sgn * delta * side[1],
                p[2] + sgn * delta * side[2]};
            const auto [id, metric_pt] = tf::neighbor_search(
                form, tf::make_point(off[0], off[1], off[2]));
            static_cast<void>(metric_pt);
            const auto poly = polygons[id];
            const auto nrm = tf::make_normal(poly[0], poly[1], poly[2]);
            mesh_n[s] = {nrm[0], nrm[1], nrm[2]};
            true_n[s] = tf::test::volume_field::field_gradient(shape, off);
          }
          auto angle_between = [](const tf::test::volume_field::vec3 &a,
                                  const tf::test::volume_field::vec3 &b) {
            const float dot = std::min(
                std::max(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.f), 1.f);
            return std::acos(dot) * 180.f / 3.14159265f;
          };
          const float err = std::abs(angle_between(mesh_n[0], mesh_n[1]) -
                                     angle_between(true_n[0], true_n[1]));
          dih_sum += err;
          dih_max = std::max(dih_max, double(err));
          ++dih_n;
        }
    }

    // What a pass of the build costs is bench_isosurface's table, read off the
    // builder's own clock; this harness scores the surface.
    std::printf(
        "%-3s %-9s n=%-4d | surf %.4f / %.4f sp | nrm smooth %.2f edge "
        "%.2f deg | edge_cd %.3f / %.3f sp | dihedral err %.1f / %.1f "
        "deg | %.2f ms | %zu faces\n",
        method, name, n, surf_sum / surf_wsum / spacing, surf_max / spacing,
        nrm_smooth_w > 0 ? nrm_smooth_sum / nrm_smooth_w : 0.0,
        nrm_edge_w > 0 ? nrm_edge_sum / nrm_edge_w : 0.0,
        ecd_n ? ecd_sum / double(ecd_n) / spacing : 0.0, ecd_max / spacing,
        dih_n ? dih_sum / double(dih_n) : 0.0, dih_max, extract_ms,
        std::size_t(polygons.size()));
  };

  run_metrics(mesh.polygons());
}

} // namespace

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 128;
  const char *shape = argc > 2 ? argv[2] : "box";
  const char *method = argc > 3 ? argv[3] : "fe";
  const double stabilizer = argc > 4 ? std::atof(argv[4]) : 0.1;
  if (!tf::test::with_named_field(shape, [&](const auto &s) {
        run(s, n, shape, method, stabilizer);
      })) {
    std::fprintf(stderr, "unknown shape %s\n", shape);
    return 2;
  }
  return 0;
}
