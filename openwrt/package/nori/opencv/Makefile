#
# Copyright (C) 2013-2014 wrtnode.com
# Copyright (C) 2015-2016 OpenWrt.org
#
# This is free software, licensed under the GNU General Public License v2.
# See /LICENSE for more information.
#

include $(TOPDIR)/rules.mk

PKG_NAME:=opencv
PKG_VERSION:=4.9.0
PKG_RELEASE:=1

PKG_SOURCE_PROTO:=git
PKG_SOURCE_URL:=https://github.com/opencv/opencv.git
PKG_SOURCE_VERSION:=$(PKG_VERSION)
PKG_MIRROR_HASH:=60ae3d790adb89b1951e9b928d369fd33655bc98540fa18a7afc31f59f63a668

PKG_LICENSE:=Apache-2.0
PKG_LICENSE_FILES:=LICENSE

PKG_BUILD_PARALLEL:=1

QUILT:=1

include $(INCLUDE_DIR)/package.mk

CMAKE_BINARY_SUBDIR:=build
CMAKE_INSTALL:=1

CMAKE_OPTIONS += \
	-DWITH_GTK=OFF \
	-DWITH_TIFF=OFF \
	-DWITH_QT=OFF \
	-DWITH_GSTREAMER=OFF

CMAKE_OPTIONS += \
	-DBUILD_IPP_IW:BOOL=OFF \
	-DBUILD_ITT:BOOL=OFF \
	-DBUILD_JAVA:BOOL=OFF \
	-DBUILD_PERF_TESTS:BOOL=OFF \
	-DBUILD_TESTS:BOOL=OFF

# opencv modules
CMAKE_OPTIONS += \
	-DBUILD_opencv_apps:BOOL=OFF \
	-DBUILD_opencv_calib3d:BOOL=OFF \
	-DBUILD_opencv_features2d:BOOL=OFF \
	-DBUILD_opencv_flann:BOOL=OFF \
	-DBUILD_opencv_gapi:BOOL=OFF \
	-DBUILD_opencv_java_bindings_gen:BOOL=OFF \
	-DBUILD_opencv_js_bindings_gener:BOOL=OFF \
	-DBUILD_opencv_objc_bindings_gen:BOOL=OFF \
	-DBUILD_opencv_objdetect:BOOL=OFF \
	-DBUILD_opencv_photo:BOOL=OFF \
	-DBUILD_opencv_python3:BOOL=OFF \
	-DBUILD_opencv_python_bindings_g:BOOL=OFF \
	-DBUILD_opencv_python_tests:BOOL=OFF \
	-DBUILD_opencv_stitching:BOOL=OFF \
	-DBUILD_opencv_ts:BOOL=OFF \
	-DBUILD_opencv_video:BOOL=OFF \
	-DBUILD_opencv_highgui:BOOL=OFF \
	-DBUILD_opencv_videoio:BOOL=ON \
	-DBUILD_opencv_dnn:BOOL=ON

ifeq ($(CONFIG_ENABLE_OPENCV_PYTHON_SUPPORT),y)
CMAKE_OPTIONS += -DPYTHON3_INCLUDE_PATH=$(STAGING_DIR)/usr/include/python3.9 \
				-DPYTHON3_LIBRARIES=$(STAGING_DIR)/usr/lib/libpython3.9.so \
				-DPYTHON3_NUMPY_INCLUDE_DIRS=$(TARGET_ROOTFS_DIR)/pypi/numpy-1.20.1/ipkg-install/usr/lib/python3.9/site-packages/numpy/core/include \
				-DBUILD_opencv_python3:BOOL=ON \
				-DBUILD_opencv_python_bindings_g:BOOL=ON 
endif

ifeq ($(CONFIG_ENABLE_SUNXI_VIN_ISP),y)
CMAKE_OPTIONS += -DENABLE_SUNXI_VIN_ISP=ON
else
CMAKE_OPTIONS += -DENABLE_SUNXI_VIN_ISP=OFF
endif

include $(INCLUDE_DIR)/cmake.mk

define Package/$(PKG_NAME)/config
	source "$(SOURCE)/Config.in"
endef

define Package/$(PKG_NAME)
  SECTION:=libs
  CATEGORY:=AlgorithmsLibraries
  TITLE:=OpenCV
  DEPENDS:=+libpthread +librt +libatomic +libstdcpp +zlib +libjpeg +libpng +libwebp +ENABLE_SUNXI_VIN_ISP:libAWIspApi +ENABLE_OPENCV_PYTHON_SUPPORT:python3-numpy
  URL:=https://opencv.org/
endef

define Package/$(PKG_NAME)/description
  OpenCV is the world's biggest computer vision library.
  It's open source, contains over 2500 algorithms and
  is operated by the non-profit Open Source Vision Foundation.
endef

define Package/$(PKG_NAME)/install
	$(INSTALL_DIR) $(1)
	$(CP) $(PKG_INSTALL_DIR)/* $(1)/
endef

$(eval $(call BuildPackage,opencv))