# 拼好卡 · StageBridge

**把生成流程拆开，让现有显卡各司其职。**

[English](README.md) | **简体中文**

StageBridge（中文名“拼好卡”）把文本／视觉编码器 TE 做成独立服务：一台 GPU 机器负责编码，另一台 ComfyUI worker 负责 DiT、采样、VAE 解码和保存。v0.1.0 提供服务端、四个自定义节点、API 工作流、脱敏实测数据和配图。

这是一个**已经跑出图片和视频的实验性组件发布**。它按阶段分工，不把几张卡的显存拼成一个大池；一个 worker 仍要承担自己的采样负载。跨机器自动任务分发尚未接入。

![拼好卡按阶段分工：输入、远程编码、生成worker、输出文件](assets/architecture.zh.png)

## 已实现的能力

- Qwen Image 2.1 文生图和图像编辑的远程编码；参考图的 VAE latent 仍在 worker 本地生成。
- MiniMax H3 文本／首尾帧条件编码。完整视频实测覆盖首帧 `fl2va`；任意音频、视频参考模式尚未接入。
- safetensors 无损传输 float32 条件张量，校验形状、附加字段和权重指纹；版本不符返回 409。
- 有界队列、断连取消、节点内连接复用、可选令牌认证；不会把不确定的 POST 失败随意重发。
- Image 与 H3 按请求加载并互斥切换，空闲自动卸载；本次配置为 **10 分钟**，已做真实等待验收。
- Image 可选本地回退；公开样例默认关闭，避免小显存 worker 意外加载本地 TE。H3 没有自动本地 TE 回退。

## 四张笔记本显卡，都实际生成了 H3 视频

每行一个完整样本：**864×480、124 帧、24 fps、10 步，输出约 5.17 秒**。这里全部是 Laptop GPU，不能套用桌面卡规格。

| 视频 worker | 显存 | TE 所在机器 | ComfyUI 执行时间 |
|---|---:|---|---:|
| RTX 5090 Laptop | 24 GiB | RTX 5080 Laptop | 187.570 秒 |
| RTX 5080 Laptop | 16 GiB | RTX 5090 Laptop | 233.113 秒 |
| RTX 4080 Laptop | 12 GiB | RTX 5080 Laptop | 243.159 秒 |
| RTX 3080 Ti Laptop | 16 GiB | RTX 5080 Laptop | 261.838 秒 |

计时取 ComfyUI 的开始和成功事件，包含 TE、采样与保存。**TE 机器、缓存节点和冷热状态不同，这不是显卡算力排行榜。** 四台是先后测试，尚未测四 worker 并发吞吐。4080 完整样例的最低采样空闲系统内存约 1.54 GiB，也说明“能跑”不等于有充足并发余量。

![四次完整H3样例耗时及比较边界](assets/workflow-times.zh.png)

## 不贴视频，直接看实际抽帧

每行分别是保存视频的第 0、62、123 帧。四个源 MP4 都实际解码出 124 帧；图中显示输出样例，不代表不同机器画质等价。源视频 SHA 和抽帧位置见[拼图来源](assets/montage-provenance.json)。

![四台worker生成视频的首中末帧拼图](assets/h3-four-workers-montage.png)

## 快速开始

先准备带对应 Qwen Image 2.1／MiniMax H3 实现的兼容 ComfyUI，以及有权使用的模型文件。本包不包含模型、ComfyUI 或 CUDA 内核；特定量化格式还依赖宿主运行时。先阅读[环境和安装说明](docs/SETUP.md)。

```powershell
# 使用现有 ComfyUI 的 Python；不另装一套 Torch 覆盖宿主。
python -m pip install -r requirements-server.txt
python scripts/check_runtime.py --comfy-root C:/AI/ComfyUI_windows_portable/ComfyUI
Copy-Item config.example.yaml config.local.yaml
# 编辑 ComfyUI / 权重路径；删除不需要的 profile。
python run_server.py --config config.local.yaml
```

默认只监听本机。跨机器使用时，将配置 `host` 改为 `0.0.0.0`，按自己网络环境放行选定端口，再把节点 `server_url` 改成 TE 机器的局域网地址。无需 SSH 隧道。服务端令牌用 `TE_AUTH_TOKEN`，worker 用 `REMOTE_TE_API_TOKEN`，不要写进工作流或 Git。

在 worker 安装节点并重启 ComfyUI：

```powershell
python scripts/install_nodes.py --comfy-root C:/AI/ComfyUI_windows_portable/ComfyUI
```

安装器拒绝覆盖已有目录。也可以将节点 ZIP 解压到 `ComfyUI/custom_nodes`。如已安装旧 `ComfyUI-RemoteTE`，先按自己的停机流程备份并移出旧副本，避免与 `ComfyUI-StageBridge` 重复注册。

`workflows/` 内是 **API prompt graph，不是拖入界面的 UI 工作流**。提交前设置服务地址、核对 `/health` 的指纹、选好模型，并把 `example.png` 换成 worker 已上传的图片。原参考图和视频未分发，不能据此逐字节重现历史结果。

## 验证与边界

```powershell
python scripts/inspect_results.py
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

第一条只用标准库重算历史计时；测试使用 mock 编码器和本机 HTTP，不加载权重。公开数据、打包改动和验收解释见 [results](results/h3-runs.json)、[EVIDENCE](docs/EVIDENCE.md)、[PACKAGING](docs/PACKAGING.md)。

Image 同一 5090 上的 HTTP／原生编码，已测 7 例逐比特相等；这证明被测传输链路无额外误差，不代表跨 GPU 原生编码全等，也不代表 H3 已完成同等级数值对照。历史跨 5090／3080Ti 的最大绝对差 0.09015 **不是 9% 错误**，相对 RMS 差约 0.00425%；旧 0.02 最大差门槛确实未过，公开保留。[详细解释](docs/EVIDENCE.md)

拆开 TE 的价值是资源分工和部署选择，不承诺每条任务更快。[另一个 CPU–GPU TE 实验](https://github.com/JingWang-Star996/cpu-te-bench)已经观察到合理预留显存后 GPU 编码明显加速，历史约 70 秒不能作为普遍基准。该实验只测 TE，不能与这里完整视频时间混用。

当前服务仍要求 CUDA；纯 CPU TE、跨机器自动分发、其他系统／后端、高并发和长视频尚未验收。本项目原始代码与文档采用 MIT；ComfyUI、模型及依赖遵循各自许可证。[第三方说明](THIRD_PARTY_NOTICES.md)
