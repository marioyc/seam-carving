CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_MODULE_PATH $ENV{OPENCV_DIR})

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(seam-carving)

ADD_EXECUTABLE(seam-carving seam-carving-image.cpp)
ADD_EXECUTABLE(video-seams seam-carving-video.cpp)

TARGET_LINK_LIBRARIES(seam-carving ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(video-seams ${OpenCV_LIBS})
