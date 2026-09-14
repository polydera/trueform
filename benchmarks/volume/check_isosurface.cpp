// Topology gates for the manifold completion over the full fixture ladder,
// which is larger than a test wants to carry. The checks run on the emitted
// triangles; the completed polygon surface behind them is audited in
// tests/volume, where the builder states it on request.
//
//   ./check_isosurface [n=128] [fixture=all] [refine=1]
#include "volume_fields.hpp"

#include <trueform/volume/impl/flying_dc.hpp>
#include <trueform/volume/make_isosurface.hpp>

#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct report {
  bool manifold = false;
  bool closed = false;
  std::size_t boundary_edges = 0;
  std::size_t over_incident = 0;
  std::size_t pinched_vertices = 0;
  std::size_t duplicate_faces = 0;
  std::size_t degenerate_faces = 0;
  std::size_t unreferenced = 0;
  std::size_t nonfinite = 0;
  std::size_t unpaired_orientation = 0;
  double signed_volume = 0;
};

// A vertex whose incident corners form more than one cycle is pinched: two
// sheets meeting at one identity. Edge degree alone cannot see it, so the link
// of every vertex is walked.
struct corner_link {
  int v, a, b;
};

auto count_pinched(std::vector<corner_link> &links) -> std::size_t {
  std::sort(
      links.begin(), links.end(),
      [](const corner_link &p, const corner_link &q) { return p.v < q.v; });
  std::size_t pinched = 0;
  std::vector<int> nodes;
  std::vector<int> parent;
  for (std::size_t i = 0; i < links.size();) {
    std::size_t j = i;
    while (j < links.size() && links[j].v == links[i].v)
      ++j;
    nodes.clear();
    for (std::size_t k = i; k < j; ++k) {
      nodes.push_back(links[k].a);
      nodes.push_back(links[k].b);
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    parent.assign(nodes.size(), 0);
    for (std::size_t k = 0; k < parent.size(); ++k)
      parent[k] = int(k);
    const auto find = [&](int x) {
      while (parent[std::size_t(x)] != x)
        x = parent[std::size_t(x)] =
            parent[std::size_t(parent[std::size_t(x)])];
      return x;
    };
    const auto index_of = [&](int id) {
      return int(std::lower_bound(nodes.begin(), nodes.end(), id) -
                 nodes.begin());
    };
    for (std::size_t k = i; k < j; ++k) {
      const int ra = find(index_of(links[k].a));
      const int rb = find(index_of(links[k].b));
      if (ra != rb)
        parent[std::size_t(rb)] = ra;
    }
    std::size_t components = 0;
    for (std::size_t k = 0; k < parent.size(); ++k)
      components += find(int(k)) == int(k);
    pinched += components > 1;
    i = j;
  }
  return pinched;
}

auto fnv(const std::vector<int> &key) -> std::uint64_t {
  std::uint64_t h = 1469598103934665603ull;
  for (int v : key)
    for (int b = 0; b < 4; ++b) {
      h ^= std::uint64_t((v >> (b * 8)) & 0xff);
      h *= 1099511628211ull;
    }
  return h;
}

// `n_referenced` bounds the vertices this surface must use: the fan centres
// exist only in the triangulation, so the polygon view stops before them.
template <typename Polygons>
auto audit(const Polygons &polys, std::size_t n_points, const float *points,
           std::size_t n_referenced) -> report {
  report r;
  r.manifold = tf::is_manifold(polys);
  r.closed = tf::is_closed(polys);
  std::vector<std::pair<int, int>> edges;
  std::vector<std::pair<int, int>> directed;
  std::vector<std::uint64_t> face_hashes;
  std::vector<corner_link> links;
  std::vector<char> used(n_points, 0);
  std::vector<int> key;
  for (const auto &poly : polys) {
    const int m = int(poly.size());
    key.clear();
    bool degenerate = false;
    for (int i = 0; i < m; ++i) {
      const int a = int(poly.indices()[i]);
      const int b = int(poly.indices()[(i + 1) % m]);
      const int prev = int(poly.indices()[(i + m - 1) % m]);
      if (a == b)
        degenerate = true;
      used[std::size_t(a)] = 1;
      key.push_back(a);
      edges.push_back({std::min(a, b), std::max(a, b)});
      directed.push_back({a, b});
      links.push_back({a, prev, b});
    }
    std::sort(key.begin(), key.end());
    for (int i = 1; i < m; ++i)
      if (key[std::size_t(i)] == key[std::size_t(i - 1)])
        degenerate = true;
    r.degenerate_faces += degenerate;
    face_hashes.push_back(fnv(key));
  }
  std::sort(edges.begin(), edges.end());
  for (std::size_t i = 0; i < edges.size();) {
    std::size_t j = i;
    while (j < edges.size() && edges[j] == edges[i])
      ++j;
    r.boundary_edges += (j - i) == 1;
    r.over_incident += (j - i) > 2;
    i = j;
  }
  // a consistently oriented surface traverses every interior edge once each
  // way, so every directed edge must find its reverse
  {
    auto rev = directed;
    for (auto &d : rev)
      std::swap(d.first, d.second);
    std::sort(directed.begin(), directed.end());
    std::sort(rev.begin(), rev.end());
    std::size_t i = 0, j = 0;
    while (i < directed.size() && j < rev.size()) {
      if (directed[i] == rev[j]) {
        ++i;
        ++j;
      } else if (directed[i] < rev[j]) {
        ++r.unpaired_orientation;
        ++i;
      } else {
        ++j;
      }
    }
    r.unpaired_orientation += directed.size() - i;
  }
  std::sort(face_hashes.begin(), face_hashes.end());
  for (std::size_t i = 1; i < face_hashes.size(); ++i)
    r.duplicate_faces += face_hashes[i] == face_hashes[i - 1];
  for (std::size_t i = 0; i < n_points; ++i) {
    r.unreferenced += i < n_referenced && used[i] == 0;
    for (int d = 0; d < 3; ++d)
      r.nonfinite += !std::isfinite(points[i * 3 + std::size_t(d)]);
  }
  r.pinched_vertices = count_pinched(links);
  return r;
}

auto print(const char *what, const report &r) -> bool {
  const bool ok = r.manifold && r.over_incident == 0 &&
                  r.pinched_vertices == 0 && r.duplicate_faces == 0 &&
                  r.degenerate_faces == 0 && r.unreferenced == 0 &&
                  r.nonfinite == 0 &&
                  r.unpaired_orientation == r.boundary_edges &&
                  (!r.closed || r.signed_volume > 0);
  std::printf("    %-10s manifold=%d closed=%d boundary=%zu over=%zu pinch=%zu "
              "dup=%zu degen=%zu unref=%zu nonfinite=%zu unoriented=%zu "
              "volume=%.5f %s\n",
              what, int(r.manifold), int(r.closed), r.boundary_edges,
              r.over_incident, r.pinched_vertices, r.duplicate_faces,
              r.degenerate_faces, r.unreferenced, r.nonfinite,
              r.unpaired_orientation -
                  std::min(r.unpaired_orientation, r.boundary_edges),
              r.signed_volume, ok ? "" : "  <-- FAIL");
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 128;
  const std::string which = argc > 2 ? argv[2] : "all";
  const int refine = argc > 3 ? std::atoi(argv[3]) : 1;

  const char *all[] = {"box",     "wedge30",  "wedge45", "wedge60",
                       "wedge90", "wedge120", "gear",    "sphere",
                       "slab",    "gyroid",   "noise"};
  const bool closed_fixture[] = {true, true, true,  true,  true, true,
                                 true, true, false, false, false};
  int failures = 0;
  for (std::size_t fi = 0; fi < sizeof(all) / sizeof(all[0]); ++fi) {
    const char *name = all[fi];
    if (which != "all" && which != name)
      continue;
    const auto vol = tf::test::volume_by_name(name, n);
    tf::isosurface_config config;
    config.method = tf::isosurface_method::dual_contouring;
    config.refine = refine != 0;
    const auto mesh = tf::make_isosurface(vol.volume(), 0.f, config);
    const auto n_points = std::size_t(mesh.points().size());
    const float *pts = mesh.points_buffer().data_buffer().data();

    std::printf("  %-9s n=%-4d refine=%d | V=%td T=%td\n", name, n, refine,
                std::ptrdiff_t(n_points),
                std::ptrdiff_t(mesh.polygons().size()));
    const auto tri_report = audit(mesh.polygons(), n_points, pts, n_points);
    auto tri_r = tri_report;
    tri_r.signed_volume = double(tf::signed_volume(mesh.polygons()));
    bool ok = print("triangles", tri_r);
    if (closed_fixture[fi] && !(tri_r.closed && tri_r.manifold)) {
      std::printf("    contained fixture must be closed and manifold\n");
      ok = false;
    }
    failures += !ok;
  }
  std::printf("%s\n", failures ? "CHECK FAILED" : "checks clean");
  return failures ? 1 : 0;
}
