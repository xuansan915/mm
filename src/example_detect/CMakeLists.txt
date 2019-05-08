cmake_minimum_required(VERSION 2.8)
project(detect)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH "/usr/local")
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

set(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O0 -pthread -Wall -g -ggdb  -fopenmp" )

# Requires OpenCV v3.0.0 or later
FIND_PACKAGE( OpenCV REQUIRED )
IF (${OpenCV_VERSION} VERSION_LESS 3.0.0)
    MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}. This requires atleast OpenCV v3.0.0")
ENDIF()


include_directories(
#   ${PROJECT_SOURCE_DIR}/../../include/tensorflow-pb
   /home/jerry/dlib-19.7
   /opt/OpenBLAS/include

)


link_directories(/opt/OpenBLAS/lib)
link_directories(/home/jerry/dlib-19.7/dlib/build)

add_definitions(-DUSE_SSE2_INSTRUCTIONS=ON -DUSE_SSE4_INSTRUCTIONS=ON -DUSE_AVX_INSTRUCTIONS=ON)
add_definitions(-DHAVE_OPENCV_IMGPROC=ON)

AUX_SOURCE_DIRECTORY(./ DIR_SRCS)
AUX_SOURCE_DIRECTORY(../mtcnn mtcnn)
AUX_SOURCE_DIRECTORY(../reasoningmachine reasonmachine)
AUX_SOURCE_DIRECTORY(../reasoningmachine/mvreasonmachine mvreasonmachine)
AUX_SOURCE_DIRECTORY(../eigenvalue eigenvalue)
AUX_SOURCE_DIRECTORY(../eigenvalue/mveigenvalue mveigenvalue)
AUX_SOURCE_DIRECTORY(../reconize reconize)

add_executable (detect  ${DIR_SRCS} ${mtcnn} ${reasonmachine} ${mvreasonmachine} ${mveigenvalue} ${eigenvalue} ${reconize})
#ADD_LIBRARY (reconize SHARED ${DIR_SRCS} ${mtcnn} ${reasonmachine} ${mvreasonmachine} ${mveigenvalue} ${eigenvalue})
target_link_libraries( detect ${OpenCV_LIBS} openblas mvnc)
