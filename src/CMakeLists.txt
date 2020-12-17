project(solveT)



# Eigen
include_directories("/opt/homebrew/Cellar/eigen/3.3.8_1/include")
set(EIGEN3_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/3.3.8_1/include/")
include_directories("/opt/homebrew/Cellar/eigen/3.3.8_1/include/eigen3")

SET(OpenCV_DIR "/Users/jin/Q_Mac/third_party/opencv-3.4.12/build")
find_package( OpenCV 3.4 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${OpenCV_LIB_DIR})
MESSAGE( STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS} )
MESSAGE( STATUS "OpenCV_LIB_DIR: " ${OpenCV_LIB_DIR} )
MESSAGE( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS} )



SET(Sophus_DIR "/Users/jin/Q_Mac/third_party/Sophus/build")
find_package( Sophus REQUIRED )
message( STATUS "Sophus_INCLUDE_DIR: " ${Sophus_INCLUDE_DIR})
message( STATUS "Sophus_LIBRARY: " ${Sophus_LIBRARY})
include_directories( ${Sophus_INCLUDE_DIR} )



SET(CSparse_include_DIR "/Users/jin/Q_Mac/third_party/SuiteSparse/include/")
SET(CSparse_LIB "/Users/jin/Q_Mac/third_party/SuiteSparse/lib/")
include_directories( ${CSparse_include_DIR} )
link_directories( ${CSparse_LIB} )

# SET( Ceres_DIR  "/Users/jin/Q_Mac/third_party/ceres-solver/builds")

# SET(CERES_INCLUDE_DIRS "/Users/jin/Q_Mac/third_party/ceres-solver/include/")
# SET(CERES_LIBRARIES "/Users/jin/Q_Mac/third_party/ceres-solver/builds/lib/")

find_package( Ceres REQUIRED )
MESSAGE( STATUS "CERES_INCLUDE_DIRS: " ${CERES_INCLUDE_DIRS} ) # include eigen3 as well
MESSAGE( STATUS "CERES_LIBRARIES: " ${CERES_LIBRARIES} )
include_directories( ${CERES_INCLUDE_DIRS} )

# /usr/local/Cellar/glog/0.4.0
SET(glog_include_DIR "/opt/homebrew/Cellar/glog/0.4.0/include/")
SET(glog_LIB "/opt/homebrew/Cellar/glog/0.4.0/lib/")
find_package( glog )
include_directories( ${glog_include_DIR})
link_directories( ${glog_LIB} )

# "/usr/local/Cellar/gflags/2.2.2/include"
SET(gflags_include_DIR "/opt/homebrew/Cellar/gflags/2.2.2/include/")
SET(gflags_LIB "/opt/homebrew/Cellar/gflags/2.2.2/lib/")
find_package( gflags )
include_directories( ${gflags_include_DIR})
link_directories( ${gflags_LIB} )



ADD_EXECUTABLE( objectBA objectBA.cpp) # always test cmake function first
target_link_libraries(objectBA  ${CERES_LIBRARIES} ${Sophus_LIBRARY}) 


ADD_EXECUTABLE( objectBAmv objectBAmv.cpp) # always test cmake function first
target_link_libraries(objectBAmv  ${CERES_LIBRARIES} ${Sophus_LIBRARY}) 
