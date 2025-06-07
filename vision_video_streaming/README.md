# 树莓派视频流使用说明

本文档用于介绍树莓派视频流的使用方法。

## 目录

- [树莓派视频流使用说明](#树莓派视频流使用说明)
  - [目录](#目录)
  - [1. 网络连接要求](#1-网络连接要求)
  - [2. 获取树莓派IP地址](#2-获取树莓派ip地址)
    - [方法1：树莓派上查看（推荐）](#方法1树莓派上查看推荐)
    - [方法2：主机上扫描网络](#方法2主机上扫描网络)
  - [3. 访问视频流](#3-访问视频流)
  - [4. 相机参数说明](#4-相机参数说明)
  - [5. 服务管理命令](#5-服务管理命令)
  - [6. 常见问题解决](#6-常见问题解决)
    - [无法看到视频？](#无法看到视频)
    - [服务无法启动？](#服务无法启动)

## 1. 网络连接要求

- **必须**确保树莓派和观看视频的主机连接到**同一个Wi-Fi网络**
- 比赛现场使用手机热点时，树莓派和主机都需连接该热点

## 2. 获取树莓派IP地址

### 方法1：树莓派上查看（推荐）

1. 打开树莓派终端
2. 输入命令：

   ```bash
   hostname -I
   ```

3. 记录输出的IP地址（如 `192.168.1.100`）

### 方法2：主机上扫描网络

1. Windows系统：

   - 下载安装[Advanced IP Scanner](https://www.advanced-ip-scanner.com/)
   - 扫描网络，查找设备名为 `rescue`的设备
2. Linux系统：

   ```bash
   sudo apt install arp-scan
   sudo arp-scan --localnet
   ```

## 3. 访问视频流

在电脑浏览器中输入：

```txt
http://[树莓派IP地址]:8080/?action=stream
```

将 `[树莓派IP地址]`替换为实际IP，例如：

```txt
http://192.168.1.100:8080/?action=stream
```

## 4. 相机参数说明

- **当前设置**：

  - 分辨率：640×480
  - 帧率：30fps
- **查看支持参数**：
  在树莓派终端运行：

  ```bash
  v4l2-ctl --device=/dev/video0 --list-formats-ext
  ```

- **修改参数**（如需调整）：

  1. 编辑服务配置：

     ```bash
     sudo nano /etc/systemd/system/camera.service
     ```

  2. 修改 `-r`后的分辨率（如1280x720）和 `-f`后的帧率值
  3. 保存后重启服务：

     ```bash
     sudo systemctl restart camera
     ```

## 5. 服务管理命令

| 操作     | 命令                              |
| -------- | --------------------------------- |
| 启动服务 | `sudo systemctl start camera`   |
| 停止服务 | `sudo systemctl stop camera`    |
| 重启服务 | `sudo systemctl restart camera` |
| 查看状态 | `systemctl status camera`       |
| 开机自启 | `sudo systemctl enable camera`  |

## 6. 常见问题解决

### 无法看到视频？

1. 检查树莓派电源指示灯（绿灯常亮表示正常运行）
2. 确认树莓派和主机在同一网络
3. 在树莓派本地测试：

   ```bash
   curl http://localhost:8080/?action=stream
   ```

4. 重启服务：

   ```bash
   sudo systemctl restart camera
   ```

### 服务无法启动？

1. 检查相机连接：

   ```bash
   ls /dev/video*
   ```

2. 查看错误日志：

   ```bash
   journalctl -u camera -n 50 --no-pager
   ```
