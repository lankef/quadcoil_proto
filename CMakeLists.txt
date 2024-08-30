# cmake_minimum_required(VERSION 3.14)
# project(biest_call LANGUAGES CXX)

# # Minimum supported c++ standard of xtensor
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Finding python and numpy
# set(Python_FIND_STRATEGY LOCATION)
# # find_package(Python REQUIRED COMPONENTS Interpreter Development)
# find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# message(Python_LIBRARIES="${Python_LIBRARIES}")
# message(Python_NumPy_INCLUDE_DIRS="${Python_NumPy_INCLUDE_DIRS}")

# # Finding pybind11
# # Get the pybind11 CMake directory from the Python executable 
# # in the active conda environment
# execute_process(
#     COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
#     OUTPUT_VARIABLE pybind11_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# # Use the pybind11 directory
# list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
# find_package(pybind11 REQUIRED)
# find_package(xtensor REQUIRED)
# find_package(xtensor-python REQUIRED)

# # Include directories
# include_directories(${CMAKE_SOURCE_DIR}/thirdparty/xtensor/include)
# include_directories(${CMAKE_SOURCE_DIR}/thirdparty/xtensor-python/include)
# include_directories(${CMAKE_SOURCE_DIR}/thirdparty/xtl/include)
# include_directories(${CMAKE_SOURCE_DIR}/thirdparty/SCTL/include)
# include_directories(${CMAKE_SOURCE_DIR}/thirdparty/BIEST/include)
# include_directories(${Python_NumPy_INCLUDE_DIRS})

# # Source files
# set(SOURCES src/biest_call_cpp/biest_call.cpp)



# pybind11_add_module(biest_call src/biest_call_cpp/biest_call.cpp)

# # Link libraries
# target_link_libraries(biest_call PRIVATE ${Python_LIBRARIES} pybind11::module xtensor-python)
# target_include_directories(biest_call PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
# target_compile_definitions(biest_call
#                            PRIVATE VERSION_INFO=0.0.1)
cmake_minimum_required(VERSION 3.14)
project(biest_call LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and NumPy
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# Find pybind11 using Python
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

# Add xtensor and xtensor-python as subdirectories
# Has their own CMakeLists.txt
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtl/)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor/)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor-python/)
# Adding BIEST and SCTL as part of the current CMakeLists.txt
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/SCTL/include/)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/BIEST/include/)

# Find packages
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# Define your extension
# add_library(biest_call MODULE src/biest_call_cpp/biest_call.cpp)
pybind11_add_module(biest_call src/biest_call_cpp/biest_call.cpp)
target_link_libraries(biest_call PRIVATE OpenMP::OpenMP_CXX xtensor xtensor-python pybind11::module)
target_include_directories(biest_call PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
