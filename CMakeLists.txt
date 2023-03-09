cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

find_program (CLANG_TIDY_EXE NAMES "clang-tidy")
if (CLANG_TIDY_EXE)
  set (CMAKE_CXX_CLANG_TIDY clang-tidy)
endif ()

set (CMAKE_CXX_STANDARD 17)

project (panann)

include_directories (${PROJECT_SOURCE_DIR}/src)

set (LIB_SOURCES
  ${PROJECT_SOURCE_DIR}/src/ActivationFunction.cc
  ${PROJECT_SOURCE_DIR}/src/NeuralNetwork.cc
  ${PROJECT_SOURCE_DIR}/src/RandomWrapper.cc
  ${PROJECT_SOURCE_DIR}/src/RecurrentNeuralNetwork.cc
  ${PROJECT_SOURCE_DIR}/src/TrainingData.cc)
add_library (panann STATIC ${LIB_SOURCES})

set (TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/test.cc)
add_executable (panann_test ${TEST_SOURCES})
target_link_libraries (panann_test panann)

if (MSVC)
  # disable some benign warnings on MSVC
  add_compile_options ("/Wall;/wd4514;/wd4625;/wd4626;/wd5026;/wd5027;/wd5045;/wd4710;/wd4820;")
else ()
  add_compile_options ("-Wall;")
endif ()
