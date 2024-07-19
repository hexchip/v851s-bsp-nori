include $(TOPDIR)/rules.mk

PKG_NAME:=yolov8
PKG_VERSION:=1.0
PKG_RELEASE:=1

include $(INCLUDE_DIR)/package.mk

define Package/$(PKG_NAME)
  SECTION:=Application
  CATEGORY:=Application
  TITLE:=yolov8
  DEPENDS:=+viplite-driver +opencv +libstdcpp
endef

define Package/$(PKG_NAME)/install
	$(INSTALL_DIR) $(1)/usr/bin/
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/$(PKG_NAME) $(1)/usr/bin/
endef

$(eval $(call BuildPackage,$(PKG_NAME)))