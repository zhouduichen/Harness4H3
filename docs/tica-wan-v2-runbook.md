# TICA-Prune v2 Wan2.1-1.3B 实验准备

这份说明只描述独立实验的准备和预检，不会修改历史四组对照实验，也不在本地启动 GPU。

## 数据准备

在目标 GPU 主机准备三份 TICA manifest：

- Calibration：只用于多时间步敏感度和 FFN 激活贡献收集；
- Validation：用于父子算法比较和 AlgorithmAgent 改进判断；
- Test：只用于最终独立视频质量和硬件评测，不能出现在搜索反馈中。

三份 manifest 的 `sample_id` 必须不重复，并且应固定 prompt、latent、conditioning、timestep 和来源哈希。程序启动时会再次检查划分泄漏。

## 启动前预检

将 [evolutionary-compression.wan-tica-v2.example.yaml](/Users/huangjiahao/MinMax-H3/worktrees/Harness4H3-tica-prune-v2/configs/evolutionary-compression.wan-tica-v2.example.yaml) 复制为目标主机配置，替换模型、Wan 配置、控制器、CLIP、三份 TICA manifest 和 held-out 视频评价清单路径，并把 `source_revision` 改为实际提交号。

先执行配置解析和协议预检，确认四组输出目录彼此独立、16 个 case/seed 单元和每 case 三次延迟测量一致。预检不加载 Wan、不启动控制器，也不写历史实验目录。

确认预检通过后，再由现有四臂协议生成器生成独立输出目录。每一轮应保留 `tica-splits.json`、`tica-profile.json`、`tica-search.json`、算法源码哈希、候选 plan digest、独立评测结果和晋升决策。

## 运行边界

真实 Evolution 应使用 Validation；只有算法、模型 checkpoint、源码版本和评测协议冻结后，才运行独立的 `final-evaluate` Test。任何控制器失败、manifest 不一致、Test split 被 Evolution 读取或计划 digest 不一致，都应停止该轮并记录失败，不能把失败轮当作算法结果。FinalEvaluation 只写独立报告，不更新 Memory、SearchPolicy、模型晋升或 Agent 输入。

## 指标与预算口径

- `parameter_count` 和 `transformer_model_size_*` 只统计压缩目标 Transformer；`pipeline_model_size_*` 统计完整 Pipeline 的逻辑唯一文件大小，并按 inode 去重，因此硬链接、软链接和共享的文本编码器/VAE 不会被重复计数。报告同时保留 `fixed_component_size_*`，不能把固定组件的变化算成 Transformer 收益。
- 延迟是固定生成条件下的去噪加 VAE 解码时间；峰值显存只接受实际 `generation_device` 对应 CUDA 设备的正数监控值，缺失或异常时保留为 `null`，不会填零。
- TICA 搜索报告分别记录 `fast_evals`、`full_evals`、`interaction_evals`、每层实际深度、预算耗尽原因、共享 Calibration profiling 成本和总成本。启用四个 baseline 时，它们共享同一个搜索预算，属于候选种子对照，不是四次独立算法实验。

在 GPU 主机运行前，先执行 `tools/preflight_wan_tica.py`。它只检查 checkpoint 文件布局、三份 manifest 的来源哈希和划分泄漏，不加载 Wan；`runtime_probe` 仍必须在目标主机对实际加载的 Transformer 验证 FFN 路径、Hook 输出维度和全局通道对齐。
