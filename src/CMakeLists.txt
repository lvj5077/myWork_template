project(myVOsum)

ADD_EXECUTABLE( main main.cpp) # always test cmake function first

INCLUDE_DIRECTORIES ("/usr/local/include/eigen3/")

SET(PCL_DIR "/usr/local/Cellar/pcl/1.9.1_4/share/pcl-1.9")
FIND_PACKAGE( PCL 1.9 REQUIRED COMPONENTS common io visualization filters )
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )
link_directories( /usr/local/Cellar/flann/1.9.1_7/lib )
MESSAGE( STATUS "PCL_INCLUDE_DIRS: " ${PCL_INCLUDE_DIRS} )
MESSAGE( STATUS "PCL_LIBRARY_DIRS: " ${PCL_LIBRARY_DIRS} )


# SET(OpenCV_DIR "/Users/lingqiujin/Third_Party_Packages/opencv-2.4.13.6/build")
# find_package( OpenCV 2.4 REQUIRED )
SET(OpenCV_DIR "/usr/local/Cellar/opencv@3/3.4.5_2/share/OpenCV")
find_package( OpenCV 3.4 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${OpenCV_LIB_DIR})

include_directories( /usr/local/Cellar/opencv@3/3.4.5_2/include )
set(SIFT_LIBS "/usr/local/Cellar/opencv@3/3.4.5_2/lib/libopencv_xfeatures2d.dylib" )

MESSAGE( STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS} )
MESSAGE( STATUS "OpenCV_LIB_DIR: " ${OpenCV_LIB_DIR} )
MESSAGE( STATUS "OpenCV_LIBS: " ${OpenCV_LIBS} )



SET(G2O_DIR "/Users/jin/Third_Party_Packages/g2o/")
SET(G2O_LIB "/Users/jin/Third_Party_Packages/g2o/lib/")
include_directories( ${G2O_DIR} )
link_directories( ${G2O_LIB} )

LIST( APPEND CMAKE_MODULE_PATH "/Users/jin/Third_Party_Packages/g2o/cmake_modules" )
SET( G2O_ROOT "/Users/jin/Third_Party_Packages/g2o" )
FIND_PACKAGE( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIR} )
message(STATUS "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
MESSAGE( STATUS "G2O_INCLUDE_DIRS " ${G2O_INCLUDE_DIR} )
message(STATUS "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

SET(CSparse_include_DIR "/usr/local/Cellar/suite-sparse/5.3.0_1/include/")
SET(CSparse_LIB "/usr/local/Cellar/suite-sparse/5.3.0_1/lib/")
include_directories( ${CSparse_include_DIR} )
link_directories( ${CSparse_LIB} )


ADD_LIBRARY( slamBase slamBase.cpp )
TARGET_LINK_LIBRARIES( slamBase
   ${OpenCV_LIBS}
   ${SIFT_LIBS} # if use brew version, then not very neccessary
)

ADD_LIBRARY( myG2Oedge myG2Oedge.cpp )
TARGET_LINK_LIBRARIES( myG2Oedge
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_solver_csparse g2o_types_slam3d
   ${CSparse_include_LIB} 
)


ADD_LIBRARY( pose_estimation pose_estimation.cpp )
TARGET_LINK_LIBRARIES( pose_estimation
   ${OpenCV_LIBS}
   myG2Oedge
)

ADD_EXECUTABLE( testT testT.cpp )
TARGET_LINK_LIBRARIES( testT
	slamBase
   pose_estimation
   ${OpenCV_LIBS}
)
