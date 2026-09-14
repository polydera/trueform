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
#include <trueform/vtk/cpp/mesh_source.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSOADataArrayTemplate.h>
#include <vtkTypeInt32Array.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace {

auto square() -> vtkSmartPointer<vtkPolyData> {
  vtkNew<vtkPoints> points;
  points->SetDataTypeToFloat();
  points->InsertNextPoint(0, 0, 0);
  points->InsertNextPoint(1, 0, 0);
  points->InsertNextPoint(1, 1, 0);
  points->InsertNextPoint(0, 1, 0);

  vtkNew<vtkCellArray> polys;
  const vtkIdType first[3] = {0, 1, 2};
  const vtkIdType second[3] = {0, 2, 3};
  polys->InsertNextCell(3, first);
  polys->InsertNextCell(3, second);

  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->SetPolys(polys);
  return polydata;
}

} // namespace

TEST_CASE("a mesh source reads VTK's own arrays", "[cpp][vtk][mesh-source]") {
  auto polydata = square();
  auto source = tf::vtk::make_mesh_source<float>(polydata);

  {
    const auto mesh = source.mesh();
    // nothing was copied: the view reads VTK's own arrays
    CHECK(&mesh.points()[0][0] ==
          tf::vtk::source::interleaved_points<float>(polydata));
    CHECK(mesh.number_of_faces() == 2);
    CHECK(mesh.faces()[0].size() == 3);
    CHECK(mesh.face_membership().size() == 4);
    CHECK(source.cache().face_membership_build_count() == 1);
    CHECK(mesh.tree().bv().max[0] == 1.0F);
  }

  // the same reading is answered from what the structure already holds
  static_cast<void>(source.mesh().face_membership());
  CHECK(source.cache().face_membership_build_count() == 1);

  // and a source that moved says so through its own modification time
  polydata->GetPoints()->SetPoint(3, 0, 2, 0);
  polydata->GetPoints()->Modified();
  const auto moved = source.mesh();
  CHECK(moved.tree().bv().max[1] == 2.0F);
  CHECK(source.cache().face_membership_build_count() == 1);
}

TEST_CASE("a mesh source reads at the index the matrix carries",
          "[cpp][vtk][mesh-source]") {
  // vtkIdType is whatever VTK was built with; on an LP64 target a 64-bit one
  // is `long long` while the matrix's own is `long`, and only the matrix's own
  // names a carrier this archive was built for. Same width, same signedness,
  // so VTK's arrays are read where they lie.
  using index_type = tf::vtk::mesh_source<float>::index_type;
  STATIC_REQUIRE(std::is_same_v<index_type, std::int32_t> ||
                 std::is_same_v<index_type, std::int64_t>);
  STATIC_REQUIRE(sizeof(index_type) == sizeof(vtkIdType));
  STATIC_REQUIRE(
      std::is_same_v<decltype(tf::vtk::make_mesh_source<float>(nullptr).mesh()),
                     tf::cpp::mesh<index_type, float, 3, tf::dynamic_size>>);

  auto source = tf::vtk::make_mesh_source<float>(square());
  const auto mesh = source.mesh();
  CHECK(mesh.faces()[0][2] == index_type{2});
}

TEST_CASE("a mesh read off a source outlives the caller's handle",
          "[cpp][vtk][mesh-source]") {
  auto held = tf::vtk::make_mesh_source<float>(square()).mesh();
  CHECK(held.form()[0][1][0] == 1.0F);
}

TEST_CASE("a polydata with no polys is the empty mesh",
          "[cpp][vtk][mesh-source][empty]") {
  vtkNew<vtkPoints> points;
  points->SetDataTypeToFloat();
  points->InsertNextPoint(0, 0, 0);
  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);

  auto source = tf::vtk::make_mesh_source<float>(polydata);
  const auto mesh = source.mesh();
  CHECK(mesh.number_of_faces() == 0);
  CHECK(mesh.number_of_points() == 1);
  CHECK(mesh.faces().size() == 0);
}

TEST_CASE("a mesh source refuses what it cannot read where it lies",
          "[cpp][vtk][mesh-source][validation]") {
  CHECK_THROWS_AS(tf::vtk::make_mesh_source<double>(square()),
                  std::invalid_argument);

  {
    auto polydata = square();
    vtkNew<vtkCellArray> strips;
    const vtkIdType strip[3] = {0, 1, 2};
    strips->InsertNextCell(3, strip);
    polydata->SetStrips(strips);
    CHECK_THROWS_AS(tf::vtk::make_mesh_source<float>(polydata),
                    std::invalid_argument);
  }

  {
    auto polydata = square();
    vtkNew<vtkCellArray> polys;
    const vtkIdType line[2] = {0, 1};
    polys->InsertNextCell(2, line);
    polydata->SetPolys(polys);
    CHECK_THROWS_AS(tf::vtk::make_mesh_source<float>(polydata),
                    std::invalid_argument);
  }

  {
    vtkNew<vtkSOADataArrayTemplate<float>> coordinates;
    coordinates->SetNumberOfComponents(3);
    coordinates->SetNumberOfTuples(3);
    vtkNew<vtkPoints> points;
    points->SetData(coordinates);
    auto polydata = square();
    polydata->SetPoints(points);
    CHECK_THROWS_AS(tf::vtk::make_mesh_source<float>(polydata),
                    std::invalid_argument);
  }

  // the fifth fact is the structure's: an id no point answers for
  {
    auto polydata = square();
    vtkNew<vtkCellArray> polys;
    const vtkIdType beyond[3] = {0, 1, 4};
    polys->InsertNextCell(3, beyond);
    polydata->SetPolys(polys);
    auto source = tf::vtk::make_mesh_source<float>(polydata);
    CHECK_THROWS_AS(source.mesh(), std::out_of_range);
  }
}

/// A source is LIVE: what it was when it was read is not what it is when it is
/// asked, so the facts a borrow depends on are asked again whenever it says
/// something changed.
TEST_CASE("a source restated after it was read is read again",
          "[cpp][vtk][mesh-source][validation]") {
  vtkNew<vtkPoints> points;
  points->SetDataTypeToFloat();
  points->InsertNextPoint(0, 0, 0);
  points->InsertNextPoint(1, 0, 0);
  points->InsertNextPoint(1, 1, 0);
  auto polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);

  auto source = tf::vtk::make_mesh_source<float>(polydata);
  CHECK(source.mesh().number_of_faces() == 0);

  vtkNew<vtkCellArray> narrow;
  narrow->Use32BitStorage();
  const vtkIdType face[3] = {0, 1, 2};
  narrow->InsertNextCell(3, face);
  polydata->SetPolys(narrow);
  if (sizeof(vtkIdType) == 8)
    CHECK_THROWS_AS(source.mesh(), std::invalid_argument);

  vtkNew<vtkCellArray> line;
  const vtkIdType two[2] = {0, 1};
  line->InsertNextCell(2, two);
  polydata->SetPolys(line);
  CHECK_THROWS_AS(source.mesh(), std::invalid_argument);
}

TEST_CASE("a source given strips after it was read refuses them",
          "[cpp][vtk][mesh-source][validation]") {
  auto polydata = square();
  auto source = tf::vtk::make_mesh_source<float>(polydata);
  CHECK(source.mesh().number_of_faces() == 2);

  vtkNew<vtkCellArray> strips;
  const vtkIdType strip[4] = {0, 1, 2, 3};
  strips->InsertNextCell(4, strip);
  polydata->SetStrips(strips);
  CHECK_THROWS_AS(source.mesh(), std::invalid_argument);

  // and a source that no longer states them reads again
  polydata->SetStrips(nullptr);
  CHECK(source.mesh().number_of_faces() == 2);
}
