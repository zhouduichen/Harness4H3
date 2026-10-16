# Wan2.1-1.3B 远程真实 RSI 记录（2026-10-16）

本实验严格在远程 `ai4mbse-remote` 执行，本机没有下载 Wan 权重，也没有运行真实 Wan pipeline。

## 远程执行边界

- Code staging：`/home/intern/huangjiahao/Harness4H3-rsi/work/codex-algorithm-discovery-rsi-254959e`
- Wan checkpoint：`/home/intern/hf_cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b`
- Output root：`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-gpu1-20261016`
- Device：`cuda:1`；GPU0 保留给远程 vLLM controller
- 评测输入：256×256、81 frames、16 denoising steps、1 development case、1 seed

远端已验证：Diffusers `0.40.0`、PyTorch `2.10.0+cu128`、4 张 L40；Wan transformer 30 层、FFN 8960。远端 `nvidia-smi` 存在 NVML driver/library mismatch，但 GPU1 的 PyTorch CUDA 分配和完整 Wan pipeline 加载/生成均成功，因此 GPU1 被固定为本实验设备。

## 首轮真实结果

首轮命令：

```bash
cd /home/intern/huangjiahao/Harness4H3-rsi/work/codex-algorithm-discovery-rsi-254959e
HF_HUB_OFFLINE=1 /home/intern/miniconda3/envs/comfy/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-remote-rsi-20261016.yaml \
  --rounds 1 --json
```

首轮 A0001 从真实 profiling 选择删除 `block.17`，结构诊断通过并 reload 为 29 层；随后 independent Wan video evaluator 生成基线和候选视频，`offline_simulation=false`，视觉质量 gate 通过，视频帧数和像素均有效。

| 指标 | Baseline | Candidate | 结论 |
|---|---:|---:|---|
| CLIP-temporal quality | 0.615681 | 0.618910 | aggregate 略升 |
| semantic | 0.594424 | 0.602816 | 上升 |
| temporal | 0.955881 | 0.926122 | 下降，触发 quality floor |
| generation wall time | 7.611 s | 7.266 s | -4.54% |
| pipeline logical size | 24.311 GB | 24.225 GB | -0.36% |
| transformer parameters | 1,418,996,800 | 1,372,556,096 | -3.27% |
| peak memory | 18.773 GB | 19.279 GB | 变差 |

因此 A0001 按 promotion policy **rejected**：真实执行与质量/硬件数据齐全，但 temporal 质量门槛和效率改善门槛均未满足。首轮证据位于：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-gpu1-20261016/rounds/R0001/`

## 后续 RSI

首轮完成后，剩余 9 轮已在远程 staging 后台启动，使用同一 output root、固定 validation manifest 和 GPU1；远程日志为：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-gpu1-20261016/remote-rsi-remaining9.log`

本次远程真实配置暂未开启 TICA：远程主机未发现可用的真实 Wan calibration tensor manifest，因此不能伪造 TICA 证据。TICA 仍需单独准备远程 calibration/validation/test 缓存后再执行。
