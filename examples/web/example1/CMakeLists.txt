add_executable(web_example1 main.cpp)
target_link_libraries(web_example1 PRIVATE tf::trueform TBB::tbb)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # Produce an .html launcher so emrun can host the example directly.
  set_target_properties(web_example1 PROPERTIES SUFFIX ".html")
endif()
