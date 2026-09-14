if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "primitive geometry link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

string(FIND "${link_map}" "primitive_geometry_float_2d.cpp.o" float_2d_position)
if(float_2d_position EQUAL -1)
  message(FATAL_ERROR
    "float 2D primitive geometry link omitted its explicit-instantiation shard")
endif()

foreach(unrelated_object IN ITEMS
    primitive_geometry_float_3d.cpp.o
    primitive_geometry_double_2d.cpp.o
    primitive_geometry_double_3d.cpp.o)
  string(FIND "${link_map}" "${unrelated_object}" unrelated_position)
  if(NOT unrelated_position EQUAL -1)
    message(FATAL_ERROR
      "float 2D primitive geometry link extracted unrelated object: ${unrelated_object}")
  endif()
endforeach()

message(STATUS
  "Float 2D primitive geometry link extracted only its requested geometry shard")
