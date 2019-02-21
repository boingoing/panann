cmake_minimum_required (VERSION 2.6)
project (panann_test)

include_directories (${PROJECT_SOURCE_DIR}/src)

set (SOURCES ${PROJECT_SOURCE_DIR}/src/NeuralNetwork.cc
         ${PROJECT_SOURCE_DIR}/src/ActivationFunction.cc
         ${PROJECT_SOURCE_DIR}/src/TrainingData.cc
         ${PROJECT_SOURCE_DIR}/test/test.cc)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -Wall")

add_executable (panann_test ${SOURCES})
