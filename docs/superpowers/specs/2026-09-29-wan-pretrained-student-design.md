# Wan2.1 预训练 Student 接入设计

## 背景

当前 Student 是 H3 latent 空间上的自定义 `VideoLatentDiT`，约 1.12B
参数但随机初始化。现有 F1→F2→F3 结果可以证明训练/评估闭环运行，不能
证明一个 1–2B 模型具备可用的视频生成先验；当前语义分数偏低和输出伪影
也说明继续在随机初始化模型上追加短步数实验不是合适的主线。

本阶段采用官方发布的 Wan2.1-T2V-1.3B-Diffusers 作为新的 Student 根模型。
H3 仍保留为 Teacher，但不再强迫 Student 与 H3 共享 latent、VAE 或文本
条件张量。这样才能真正继承开源基模的时空和文本-视频先验。

## 目标与非目标

目标是建立可复现的预训练基模入口，并为后续 H3 蒸馏和 30 秒连续生成提供
稳定边界：

1. 从 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 加载真实预训练权重。
2. 支持 256×256 的短视频基线和固定 seed、采样步数、文本 prompt。
3. 记录新 lineage 根 `W0000`，明确它不是旧 H3 Student checkpoint 的续训。
4. 为 native Wan 帧率、输出帧率和重叠窗口参数留下显式配置。
5. 没有安装 Diffusers 或没有下载模型时，dry-run 和单元测试仍可执行，
   不能静默退回随机模型。

本次不把 H3→Wan 蒸馏和 30 秒训练伪装成已经完成；它们是接入成功后的
后续阶段。

## 方案比较

### 方案 A：保留 Wan 原生 VAE/latent/text encoder（采用）

Student 使用 Wan 原生 pipeline 和权重，H3 作为独立 Teacher。后续蒸馏
通过视频/latent 对齐适配器或 LoRA 进入 Wan，而不是修改 Wan 的预训练
输入空间。优点是最大化复用预训练先验，缺点是需要新增 Teacher-Student
对齐层和蒸馏损失。

### 方案 B：强制 Wan 映射到 H3 latent

增加 H3 latent↔Wan latent 的双向适配器。这样可以复用当前 H3 cache 和
长视频组合器，但适配器本身需要大量训练，Wan 原生 VAE 的收益会被削弱，
且旧 Student 的 checkpoint 仍然不能直接继承。

### 方案 C：把 Wan 权重部分移植到当前 DiT

当前 DiT 的 patch、条件投影、位置编码和 block 结构与 Wan 不同，部分
加载会产生大量未初始化参数，无法满足“不要从头随机训练”的约束，因此
不采用。

## 分阶段实施

### 阶段 W0：预训练基线

- 新增 `WanStudentSpec`、配置文件和独立 CLI。
- 通过 `from_pretrained` 加载官方 Diffusers checkpoint。
- 生成固定 prompt/seed 的 256×256 短片并记录模型 ID、采样参数、帧数、
  原生帧率和 lineage。
- 任何依赖缺失或模型下载失败都 fail-closed；不创建随机初始化替代物。

### 阶段 W1：H3 蒸馏适配

- 保持 Wan 主干预训练权重作为 `W0000`。
- 优先训练 LoRA/轻量条件适配器，再评估是否需要解冻 Wan 的时空 block。
- Teacher 目标使用 H3 的真实视频/latent rollout；单独记录 prompt 对齐、
  内容保真和时序一致性。

### 阶段 W2：连续 30 秒

- 在 Wan 的 native latent 空间实现重叠窗口 continuation。
- 先在 native FPS 下证明窗口边界连续，再做 24 FPS 输出转换和 720 帧
  合同验证。
- 只有 W1/W2 都通过后，才把结果与旧 H3 Student 的 0.696 总分进行科学
  比较。

## 验收条件

W0 必须满足：

- 代码路径明确加载 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`；
- 元数据包含 `lineage_root=W0000`、`pretrained=true`、模型 ID 和 seed；
- 没有 Diffusers、CUDA 或模型文件时，错误可读且不会 fallback 到随机模型；
- 单元测试不依赖模型下载；
- 现有 H3 Student 文件和未提交实验改动不被覆盖。

## 参考

- [Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)
- [Wan2.1 Diffusers 模型](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
