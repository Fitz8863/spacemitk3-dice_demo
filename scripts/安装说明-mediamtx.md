# mediamtx 安装说明

Dice Arena 的视频流媒体服务（骰子识别实时画面经它推到网页）。
独立于主包安装，一次安装、开机自启，升级时只需替换本包重装。

## 安装

```bash
tar -xf mediamtx-bundle-*.tar
cd mediamtx-bundle
./install.sh
```

安装内容：

- 二进制与配置落位 `~/projects/mediamtx/`；
- 注册 systemd **用户级**服务（无需 root，随桌面登录自启）。

## 管理

```bash
systemctl --user status mediamtx     # 状态
systemctl --user restart mediamtx    # 重启
systemctl --user stop mediamtx       # 停止
journalctl --user -u mediamtx -f     # 日志
```

服务地址（仅本机使用）：

- WebRTC / HTTP：`http://127.0.0.1:8889`
- RTSP：`rtsp://127.0.0.1:8554`

## 重装 / 升级

```bash
systemctl --user stop mediamtx
rm -rf ~/projects/mediamtx
# 再执行上面的安装步骤
```
