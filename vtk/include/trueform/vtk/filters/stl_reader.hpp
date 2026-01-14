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

/// @brief Reads an STL file and outputs a tf::vtk::polydata.
///
/// Outputs a tf::vtk::polydata with cached acceleration structures,
/// ready for use in trueform VTK pipelines.
///
/// @note Normals are not read from the file.
///
/// @code
/// vtkNew<tf::vtk::stl_reader> reader;
/// reader->set_file_name("mesh.stl");
/// reader->Update();
/// // Output is tf::vtk::polydata with cached structures
/// @endcode
class stl_reader : public vtkPolyDataAlgorithm {
public:
  static auto New() -> stl_reader *;
  vtkTypeMacro(stl_reader, vtkPolyDataAlgorithm);

  /// @brief Set the path to the STL file to read.
  auto set_file_name(const std::string &file_name) -> void;
  /// @brief Get the current file path.
  auto file_name() const -> const std::string &;

protected:
  stl_reader();
  ~stl_reader() override = default;

  auto RequestData(vtkInformation *, vtkInformationVector **,
                   vtkInformationVector *) -> int override;

private:
  vtkSmartPointer<polydata> _polydata;
  std::string _file_name;

  stl_reader(const stl_reader &) = delete;
  auto operator=(const stl_reader &) -> void = delete;
};

} // namespace tf::vtk
