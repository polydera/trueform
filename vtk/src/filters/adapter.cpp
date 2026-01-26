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
#include <trueform/vtk/filters/adapter.hpp>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

namespace tf::vtk {

vtkStandardNewMacro(adapter);

adapter::adapter() : _polydata(vtkSmartPointer<polydata>::New()) {
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

auto adapter::FillInputPortInformation(int port, vtkInformation *info) -> int {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
  }
  return 0;
}

auto adapter::RequestData(vtkInformation *, vtkInformationVector **input,
                          vtkInformationVector *) -> int {
  auto *in = vtkPolyData::GetData(input[0]);

  // If input is already our polydata, pass through directly
  if (auto *tf_in = polydata::SafeDownCast(in)) {
    SetOutput(tf_in);
    return 1;
  }

  if (_input_ptr != in) {
    _polydata->ShallowCopy(in);
    _input_ptr = in;
  }

  SetOutput(_polydata);
  return 1;
}

} // namespace tf::vtk
