# Wan2.1 预训练 Student 接入实施计划

> 设计对应：`docs/superpowers/specs/2026-09-29-wan-pretrained-student-design.md`

## 1. 建立独立的 Wan Student 边界

- 新增 `harness4h3/student/wan.py`。
- 定义严格校验的 `WanStudentSpec`，默认模型为
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。
- 提供 lazy `from_pretrained` 加载、固定 seed 推理、输出帧校验和元数据。
- 加载失败时抛出明确异常，禁止随机初始化 fallback。

## 2. 加入配置和独立 smoke CLI

- 新增 Wan 配置 YAML。
- 新增 `tools/wan_student_smoke.py`，支持 `--dry-run`、prompt、seed、
  device、output 和显式模型 ID。
- 保持与现有 H3 campaign 配置隔离，避免误把 Wan 权重当成 H3 Student
  checkpoint 载入。

## 3. 补充依赖和测试

- 增加可选 `wan` 依赖组；默认测试安装不强制下载大模型。
- 测试配置校验、帧数/分辨率合同、lineage 元数据和缺少 Diffusers 时的
  fail-closed 行为。
- 运行相关单元测试、compileall 和 dry-run。

## 4. 后续阶段（本次不伪造完成）

- 接入 H3 Teacher→Wan 的视频/latent 蒸馏适配器。
- 将重叠窗口 continuation 迁移到 Wan native latent 空间。
- 在真实预训练基线通过后再进行 30 秒、24 FPS 的训练与评估。
