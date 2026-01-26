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
#include <trueform/vtk/filters/obj_reader.hpp>
#include <trueform/vtk/functions/read_obj.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>

namespace tf::vtk {

vtkStandardNewMacro(obj_reader);

obj_reader::obj_reader() : _polydata(vtkSmartPointer<polydata>::New()) {
  SetNumberOfInputPorts(0);
  SetNumberOfOutputPorts(1);
}

auto obj_reader::set_file_name(const std::string &file_name) -> void {
  if (_file_name != file_name) {
    _file_name = file_name;
    Modified();
  }
}

auto obj_reader::file_name() const -> const std::string & { return _file_name; }

auto obj_reader::RequestData(vtkInformation *, vtkInformationVector **,
                             vtkInformationVector *) -> int {
  if (_file_name.empty()) {
    vtkErrorMacro("File name not set");
    return 0;
  }

  _polydata = read_obj(_file_name);
  SetOutput(_polydata);

  return 1;
}

} // namespace tf::vtk
