# Wan2.1-T2V-1.3B 5 秒真实训练与蒸馏设计

## 背景

仓库已经接入 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 W0 预训练生成与评估路径，但现有正式训练 worker 仍然面向 H3 latent 空间的自定义 `VideoLatentDiT`。因此不能仅把 30 秒配置改成 5 秒，也不能把 Wan 权重强行加载到旧 Student 图中。

本阶段把实验目标收敛为：从官方 Wan2.1-T2V-1.3B 预训练权重 `W0000` 出发，在 Wan 原生视频 latent 空间进行真实训练，并使用真实 MiniMax-H3 Teacher 做蒸馏，输出单个约 5 秒视频。

## 目标

1. 真实加载 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，禁止随机初始化 fallback。
2. 使用 Wan 原生 VAE、文本编码器、scheduler 和 Transformer 主干进行训练。
3. 使用真实 MiniMax-H3 Teacher 生成 Teacher 视频目标，并通过 Wan VAE 编码到 Wan latent 空间后进行 Student 蒸馏。
4. 训练过程必须产生真实 Student 参数更新，记录 Teacher forward、checkpoint lineage、损失和 `offline_simulation=false`。
5. 训练和评估均采用单个 Wan 原生短视频：81 帧、16 FPS、256×256，时长约 5.06 秒。
6. 保留现有 H3 Student 和 30 秒实验配置，不修改其 checkpoint lineage。

## 非目标

- 不实现 30 秒 overlap latent continuation。
- 不将训练目标强行改成 H3 latent 空间。
- 不在本阶段做 24 FPS 插值、实机部署或多架构搜索。
- 不把 W0 预训练 baseline 误标记为已经完成蒸馏训练。

## 方案比较

### 方案 A：Wan 原生 latent + H3 视频重编码蒸馏（采用）

Wan Student 保留自己的 VAE、文本条件和 latent 几何。真实 H3 Teacher 根据同一 prompt 生成 5 秒视频，Teacher 视频解码后通过 Wan VAE 编码，得到 Student 可消费的 clean latent；训练时使用 Wan 的噪声调度构造 noisy latent，并对 Student 的 denoising/velocity 预测施加标准训练损失和 Teacher 蒸馏损失。

该方案最大化保留 Wan 的预训练先验，且不假设 H3 latent 与 Wan latent 可直接互换。代价是需要实现 H3 视频导出/解码到 Wan VAE 的边界，以及 Wan 专用的训练 worker。

### 方案 B：训练 H3 latent 到 Wan latent 的直接适配器

增加一个跨 VAE/latent 的可训练适配器，直接将 H3 latent 映射为 Wan latent。该方案可能减少视频编解码，但适配器会引入新的未验证参数，难以区分 Wan 主干收益和映射误差，本阶段不采用。

### 方案 C：只用 Teacher 生成视频做 Wan 监督微调

把 H3 Teacher 视频当作伪数据，执行普通 Wan denoising 微调，不显式计算 Teacher prediction/trajectory 蒸馏项。实现较简单，但不能证明真正的 Teacher→Student 蒸馏闭环，本阶段不采用。

## 训练架构

### 根模型与冻结边界

- 根 checkpoint：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，lineage `W0000`。
- 训练参数：Wan Transformer 主干；默认冻结 Wan VAE 和文本编码器。
- 推理参数：使用 Wan 原生 scheduler/采样路径，生成 81 帧。
- 旧 H3 `VideoLatentDiT` 不参与本实验的 Student forward。

### Batch 数据流

```text
prompt + seed
    ├── real H3 Teacher forward
    │      └── H3 video latent → H3 VAE decode → RGB video
    └── Wan text encoder(prompt)
           + Wan VAE encode(RGB video)
           + noise/timestep
                 └── Wan Student Transformer
                         ├── standard denoising/velocity loss
                         └── Teacher target loss
```

Teacher 参数始终冻结；Student loss 反向传播只更新 Wan Transformer。Teacher 与 Student 必须使用独立 GPU 角色，不能通过随机目标、固定张量或 offline simulation 替代真实 forward。

### 训练阶段

1. **W1a：Wan 真实训练 smoke**
   - 加载本地 Wan 权重。
   - 完成一个真实 batch 和一个 optimizer step。
   - 验证 Student 参数 checksum 发生变化。
   - 验证真实 H3 Teacher forward 次数大于 0。

2. **W1b：Wan 5 秒蒸馏训练**
   - 使用固定 evaluation manifest 的 prompt/seed。
   - 在 Wan latent 空间执行标准训练损失与 Teacher 蒸馏损失。
   - 保存带有父 checkpoint digest、Teacher digest、model ID 和训练统计的 Student checkpoint。

3. **W1c：5 秒生成评估**
   - 从 W1b checkpoint 重新加载 Wan pipeline/Transformer。
   - 生成约 5 秒视频，验证 81 帧、16 FPS、256×256、可解码、无黑帧和无全帧重复。
   - 记录 semantic、quality、temporal 和 Teacher/Student lineage 元数据。

## 配置与边界

新增 Wan 5 秒训练配置，明确包含：

- `model_id: Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Wan 本地 pretrained snapshot
- `lineage_root: W0000`
- `num_frames: 81`
- `native_fps: 16.0`
- `width: 256`、`height: 256`
- 真实 H3 Teacher checkpoint
- Wan 训练 worker 与结果目录
- 训练方法、learning rate、standard loss/蒸馏 loss 权重
- `offline_simulation: false` 的结果断言

本阶段不复用 `student-campaign.long-video-30s.yaml` 的 36 latent frames、7 窗口和 6 帧 overlap 字段，避免把 30 秒 H3 Student 实验误当成 Wan 训练。

## 错误处理

- Wan 权重、Diffusers、CUDA 或 Wan VAE 缺失时直接失败，不能随机初始化。
- H3 Teacher 无法加载、未发生 forward 或 Teacher/Student GPU 重叠时直接失败。
- Wan latent shape、帧数、分辨率或 dtype 不匹配时直接失败。
- 训练结果若参数未变化、`offline_simulation=true` 或缺少 checkpoint digest，则不得标记成功。
- 不覆盖已有 H3/30 秒输出目录；Wan 5 秒实验使用独立目录和独立 lineage。

## 测试与验收

### 静态测试

- Wan 5 秒配置可解析，字段无未知项。
- dry-run 不下载模型、不初始化随机模型。
- 旧 H3 Student 配置仍通过现有 contract tests。

### 真实 smoke 测试

- 真实 Wan checkpoint 加载成功。
- 真实 H3 Teacher 至少完成一次 forward。
- Wan Student 至少完成一次 optimizer step，参数 checksum 改变。
- checkpoint 可重新加载并恢复 lineage 元数据。

### 生成验收

- 输出 81 帧、16 FPS、256×256，时长约 5.06 秒。
- 视频可解码，无黑帧、截断、全帧重复。
- 结果包含 `model_id`、`pretrained=true`、`lineage_root`、Teacher digest、Student digest 和 `offline_simulation=false`。

