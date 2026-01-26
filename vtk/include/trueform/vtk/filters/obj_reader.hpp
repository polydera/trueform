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
#include <trueform/vtk/core/polydata.hpp>
#include <vtkPolyDataAlgorithm.h>
#include <vtkSmartPointer.h>
#include <string>

namespace tf::vtk {

/// @brief Reads an OBJ file and outputs a tf::vtk::polydata.
///
/// Outputs a tf::vtk::polydata with cached acceleration structures,
/// ready for use in trueform VTK pipelines.
///
/// @note Only vertices and faces are read. Normals and texture coordinates
/// are not read.
///
/// @code
/// vtkNew<tf::vtk::obj_reader> reader;
/// reader->set_file_name("mesh.obj");
/// reader->Update();
/// // Output is tf::vtk::polydata with cached structures
/// @endcode
class obj_reader : public vtkPolyDataAlgorithm {
public:
  static auto New() -> obj_reader *;
  vtkTypeMacro(obj_reader, vtkPolyDataAlgorithm);

  /// @brief Set the path to the OBJ file to read.
  auto set_file_name(const std::string &file_name) -> void;
  /// @brief Get the current file path.
  auto file_name() const -> const std::string &;

protected:
  obj_reader();
  ~obj_reader() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  vtkSmartPointer<polydata> _polydata;
  std::string _file_name;

  obj_reader(const obj_reader &) = delete;
  auto operator=(const obj_reader &) -> void = delete;
};

} // namespace tf::vtk
