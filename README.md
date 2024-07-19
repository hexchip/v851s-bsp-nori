# v851s-bsp-nori 使用指南

![孙老师的yolo小相机视频封面](https://i2.hdslb.com/bfs/archive/d6b2e0424d4c7081aa58c6cff940c24e08301e53.jpg@320w_200h_1c_!web-space-index-myvideo.webp)

演示视频链接如下：  
> https://www.bilibili.com/video/BV18m421G7nS

## 前置条件

- 需要全志开源社区提供的Tina Linux v5.0 SDK。  
  请参考下列链接：
  > https://v853.docs.aw-ol.com/study/study_3getsdktoc/

- 已经搭建完编译环境。  
  搭建编译环境请参考下列链接：
  > https://v853.docs.aw-ol.com/study/study_2ubuntu/

## 应用BSP至SDK

假设`${SDK_PATH}`为 **Tina Linux v5.0 SDK** 所在路径

### 将本仓库所有文件覆盖拷贝至SDK目录

```bash
cp -rf ./* ${SDK_PATH}
```

### 应用内核补丁

```
cd ${SDK_PATH}/kernel/linux-4.9
git apply ${SDK_PATH}/patch/kernel/*.patch
```

## 直接从Docker开始

使用容器进行开发方便快捷，并且避免了污染本地环境。

我们提供了三种版本的容器镜像：

- **仅编译环境**  
    `docker pull cld1994/tina-sdk:5.0`  
    请将sdk挂载至容器。  
    例如将sdk挂载至容器中的`/workspace`目录：  
    `docker run -it -v ${SDK_PATH}:/workspace cld1994/tina-sdk:5.0`  
    **注意：windows用户请将sdk放在wsl内再进行挂载，否则影响编译速度**
- **编译环境+官方开源SDK**  
    `docker pull cld1994/tina-sdk:5.0-v853`  
    即完成了**前置条件**
- **编译环境+官方开源SDK+BSP**  
    `docker pull cld1994/tina-sdk:5.0-nori`  
    即完成了**应用BSP至SDK**