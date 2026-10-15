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

真实实验应使用固定 checkpoint、固定 GPU、固定 dtype、固定随机种子和固定 Evaluation Recipe。任何控制器失败、manifest 不一致、Test split 被读取或计划 digest 不一致，都应停止该轮并记录失败，不能把失败轮当作算法结果。
