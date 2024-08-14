cmake_minimum_required(VERSION 3.14)
project(biest_call LANGUAGES CXX)

# Minimum supported c++ standard of xtensor
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finding python and numpy
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 REQUIRED)

message(Python_LIBRARIES="${Python_LIBRARIES}")
message(Python_NumPy_INCLUDE_DIRS="${Python_NumPy_INCLUDE_DIRS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/xtensor/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/xtensor-python/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/xtl/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/SCTL/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/BIEST/include)
include_directories(${Python_NumPy_INCLUDE_DIRS})

# Source files
set(SOURCES src/biest_call_cpp/biest_call.cpp)

# Add OpenMP flags
if(OpenMP_CXX_FOUND)
    target_compile_options(biest_call PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(biest_call PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

pybind11_add_module(biest_call src/biest_call_cpp/biest_call.cpp)

# Link libraries
target_link_libraries(biest_call PRIVATE ${Python_LIBRARIES} pybind11::module)
target_compile_definitions(biest_call
                           PRIVATE VERSION_INFO=0.0.1)