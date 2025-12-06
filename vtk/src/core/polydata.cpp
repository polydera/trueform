/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/polydata.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>

namespace tf::vtk {

vtkStandardNewMacro(polydata);

auto polydata::GetData(vtkInformationVector *v, int i) -> polydata * {
  auto *info = v->GetInformationObject(i);
  auto *obj = info->Get(vtkDataObject::DATA_OBJECT());
  if (auto *pd = polydata::SafeDownCast(obj)) {
    return pd;
  }
  // Replace with our type
  auto *pd = polydata::New();
  if (obj) {
    pd->ShallowCopy(obj);
  }
  info->Set(vtkDataObject::DATA_OBJECT(), pd);
  pd->Delete(); // info now owns it
  return pd;
}

polydata::polydata() = default;

void polydata::ShallowCopy(vtkDataObject *src) {
  vtkPolyData::ShallowCopy(src);

  if (auto *other = polydata::SafeDownCast(src)) {
    _is_triangles = other->_is_triangles;
    _poly_tree_mtime = other->_poly_tree_mtime;
    _fm_mtime = other->_fm_mtime;
    _mel_mtime = other->_mel_mtime;
    _fl_mtime = other->_fl_mtime;
    _vl_mtime = other->_vl_mtime;
    _edges_buffer_mtime = other->_edges_buffer_mtime;
    _segment_tree_mtime = other->_segment_tree_mtime;
    _point_tree_mtime = other->_point_tree_mtime;
    _poly_tree = other->_poly_tree;
    _fm = other->_fm;
    _mel = other->_mel;
    _fl = other->_fl;
    _vl = other->_vl;
    _edges_buffer = other->_edges_buffer;
    _segment_tree = other->_segment_tree;
    _point_tree = other->_point_tree;
  }
}

auto polydata::set_as_triangles(bool value) -> void { _is_triangles = value; }

auto polydata::is_triangles() const -> bool { return _is_triangles; }

auto polydata::points() -> points_t { return make_points(GetPoints()); }

auto polydata::polys() -> dynamic_polys_t { return make_polys(GetPolys()); }

auto polydata::triangles() -> polys_t<3> { return make_polys<3>(GetPolys()); }

auto polydata::paths() -> paths_t { return make_paths(GetLines()); }

auto polydata::polygons() -> dynamic_polygons_t {
  return tf::make_polygons(polys(), points());
}

auto polydata::triangle_polygons() -> polygons_t<3> {
  return tf::make_polygons(triangles(), points());
}

auto polydata::curves() -> curves_t {
  return tf::make_curves(paths(), points());
}

auto polydata::point_normals() -> normals_t {
  return make_point_normals(this);
}

auto polydata::cell_normals() -> normals_t {
  return make_cell_normals(this);
}

auto polydata::poly_tree() -> const tf::aabb_tree<vtkIdType, float, 3> & {
  build_poly_tree();
  return *_poly_tree;
}

auto polydata::face_membership() -> const tf::face_membership<vtkIdType> & {
  build_face_membership();
  return *_fm;
}

auto polydata::manifold_edge_link()
    -> const tf::manifold_edge_link<vtkIdType, tf::dynamic_size> & {
  build_manifold_edge_link();
  return *_mel;
}

auto polydata::face_link() -> const tf::face_link<vtkIdType> & {
  build_face_link();
  return *_fl;
}

auto polydata::vertex_link() -> const tf::vertex_link<vtkIdType> & {
  build_vertex_link();
  return *_vl;
}

auto polydata::edges_buffer() -> const tf::blocked_buffer<vtkIdType, 2> & {
  build_edges_buffer();
  return *_edges_buffer;
}

auto polydata::segment_tree() -> const tf::aabb_tree<vtkIdType, float, 3> & {
  build_segment_tree();
  return *_segment_tree;
}

auto polydata::point_tree() -> const tf::aabb_tree<vtkIdType, float, 3> & {
  build_point_tree();
  return *_point_tree;
}

auto polydata::build_poly_tree() -> void {
  auto points_mtime = GetPoints()->GetMTime();
  auto polys_mtime = GetPolys()->GetMTime();
  auto mtime = std::max(points_mtime, polys_mtime);
  if (!_poly_tree || _poly_tree_mtime < mtime) {
    if (!_poly_tree)
      _poly_tree = std::make_unique<tf::aabb_tree<vtkIdType, float, 3>>();
    _poly_tree->build(polygons(), tf::config_tree(4, 4));
    _poly_tree_mtime = mtime;
  }
}

auto polydata::build_face_membership() -> void {
  auto mtime = GetPolys()->GetMTime();
  if (!_fm || _fm_mtime < mtime) {
    if (!_fm)
      _fm = std::make_unique<tf::face_membership<vtkIdType>>();
    _fm->build(polygons());
    _fm_mtime = mtime;
  }
}

auto polydata::build_manifold_edge_link() -> void {
  auto mtime = GetPolys()->GetMTime();
  if (!_mel || _mel_mtime < mtime) {
    if (!_mel)
      _mel = std::make_unique<
          tf::manifold_edge_link<vtkIdType, tf::dynamic_size>>();
    _mel->build(polys(), face_membership());
    _mel_mtime = mtime;
  }
}

auto polydata::build_face_link() -> void {
  auto mtime = GetPolys()->GetMTime();
  if (!_fl || _fl_mtime < mtime) {
    if (!_fl)
      _fl = std::make_unique<tf::face_link<vtkIdType>>();
    _fl->build(tf::make_faces(polys()), face_membership());
    _fl_mtime = mtime;
  }
}

auto polydata::build_vertex_link() -> void {
  auto mtime = GetPolys()->GetMTime();
  if (!_vl || _vl_mtime < mtime) {
    if (!_vl)
      _vl = std::make_unique<tf::vertex_link<vtkIdType>>();
    _vl->build(tf::make_faces(polys()), face_membership());
    _vl_mtime = mtime;
  }
}

auto polydata::build_edges_buffer() -> void {
  auto mtime = GetLines()->GetMTime();
  if (!_edges_buffer || _edges_buffer_mtime < mtime) {
    if (!_edges_buffer)
      _edges_buffer = std::make_unique<tf::blocked_buffer<vtkIdType, 2>>();
    _edges_buffer->clear();
    auto p = paths();
    std::size_t n_edges = 0;
    for (auto path : p)
      if (path.size() > 1)
        n_edges += path.size() - 1;
    _edges_buffer->reserve(n_edges);
    for (auto path : p) {
      for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        _edges_buffer->emplace_back(path[i], path[i + 1]);
      }
    }
    _edges_buffer_mtime = mtime;
  }
}

auto polydata::build_segment_tree() -> void {
  auto points_mtime = GetPoints()->GetMTime();
  auto lines_mtime = GetLines()->GetMTime();
  auto mtime = std::max(points_mtime, lines_mtime);
  if (!_segment_tree || _segment_tree_mtime < mtime) {
    if (!_segment_tree)
      _segment_tree = std::make_unique<tf::aabb_tree<vtkIdType, float, 3>>();
    auto segs = tf::make_segments(tf::make_edges(edges_buffer()), points());
    _segment_tree->build(segs, tf::config_tree(4, 4));
    _segment_tree_mtime = mtime;
  }
}

auto polydata::build_point_tree() -> void {
  auto mtime = GetPoints()->GetMTime();
  if (!_point_tree || _point_tree_mtime < mtime) {
    if (!_point_tree)
      _point_tree = std::make_unique<tf::aabb_tree<vtkIdType, float, 3>>();
    _point_tree->build(points(), tf::config_tree(4, 4));
    _point_tree_mtime = mtime;
  }
}

} // namespace tf::vtk
