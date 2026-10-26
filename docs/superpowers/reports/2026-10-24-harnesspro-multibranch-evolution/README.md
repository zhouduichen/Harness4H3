# HarnessPro Pareto 机制在 Harness4H3 的应用与实测

日期：2026-10-24
实验设备：远程 GPU 3；目标模型：Wan2.1-T2V-1.3B
完整机器可读记录见本目录的 campaign、event、accounting reconciliation、comparison 和 held-out JSON 文件。

## 论文与开源实现

论文 [HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-Offs](https://arxiv.org/abs/2610.26770) 的开源仓库是 [pzs19/HarnessPro](https://github.com/pzs19/HarnessPro)（MIT）。对 Harness4H3 最有用的不是照搬论文任务，而是把搜索拆成可恢复的候选谱系，并从质量—成本 Pareto 前沿上选取不同偏好的起点；用 Lite、Medium、Heavy 三组权重代表低成本、折中和高质量偏好；对生成、验证、训练和评测花费建立真实账本，让被拒绝和失败的尝试也计入搜索成本。

## 在 Harness4H3 的落点

- [selection.py](../../../../harness4h3/evolution/selection.py) 按质量和部署推理延迟归一化后，计算加权平方距离，最多选出三个不重复的代表。Lite／Medium／Heavy 权重为 `(1,2)`、`(1,1)`、`(2,1)`。只接受有效、非离线且达到固定质量门槛的候选。部署成本使用评测记录里的 `inference_wall_time_s_mean`。
- [campaign.py](../../../../harness4h3/evolution/campaign.py)、[loop.py](../../../../harness4h3/evolution/loop.py) 和 [cli.py](../../../../harness4h3/cli.py) 记录分支、父子候选、失败反馈及阶段耗时，支持从已有候选继续搜索。
- [compare_evolution_campaigns.py](../../../../tools/compare_evolution_campaigns.py) 对账 campaign event 与外部实测墙钟时间，并在共同开销上按完整轮次截断，重建该开销下的 Pareto 前沿。
- 运行说明见 [evolution README](../../../../harness4h3/evolution/README.md)。

这里把论文的 cost 目标落成两种明确口径：候选部署成本是每段视频推理延迟；搜索预算成本是生成、profiling、压缩训练、校验、评测及失败尝试所花的实际墙钟时间。两者不能混为同一个指标。

## 实验协议

比较单 Active 搜索与两个种子的 Pareto 多分支搜索。两组使用相同 Wan checkpoint、Controller、算子空间、GPU、质量门槛和开发集评测协议；每组最多 20 轮，名义墙钟预算为 900 秒。搜索前以 4 个开发提示、固定 seed `20261008`、81 帧、16 steps、256×256、bfloat16 重新测量原有候选。质量门槛是初始基线开发集分数的 98%，即 `0.62993658`。

开发集种子前沿为 M0001 和 M0003。按代表选择规则，Lite 选 M0003，Medium 选 M0001；当前合格前沿只有两个候选，因此没有 Heavy 第三个种子。单 Active 从其指定基线分支继续。held-out 使用另一组 4 个提示，完整 manifest SHA256 为 `ddb9a5d0e88f7a1fba0289182b7919a9eb64bdbb6f2b12695b6530e696709c74`；只在两组搜索冻结后评测。

开发集评测 recipe SHA256 为 `14d6128eae4231a61593caafe45f22f7128c73fd68b2ecde18dc77b9bca47442`，held-out recipe SHA256 为 `54e8ade57dd2c69ec4ef717e50dff1f6d207614514ab4fe51ac6e3a84bf606d7`。共同协议 SHA256 为 `1d417e9af0f3189aa9b8820cebea87e715d40a0757feb36cae31ed83b543e8a6`。

## 搜索成本与共同预算结果

日志的 `branch-events.jsonl` 阶段账本在早期重复候选失败的终止路径上漏记了一部分 profile/generation 和其他实际等待时间。为避免把这段墙钟时间当作免费，本次给每个 round 追加了 `accounting-reconciliation.jsonl`：按远端 round 目录创建与决策写入时间恢复实测 round wall time，再与已记录阶段对账。比较和下表使用对账后的总墙钟时间。漏记部分单列为 `unattributed_seconds`，不伪装成已知阶段。实验没有发生 retry，故 retry 为 0。

共同搜索开销上限取两组完整实际花费的较小值：755.550 秒。因为只纳入完整搜索轮次，多分支在上限内实际花费 740.444 秒，剩余 15.106 秒不足以纳入下一整轮（共同上限与实际花费相差约 2.0%）。

| 搜索方式 | 完整运行实际墙钟 | 共同开销内轮数 | 共同开销内实花 | 首个合格候选 | 首次合格累计开销 | 共同开销下前沿 |
|---|---:|---:|---:|---|---:|---|
| 单 Active | 755.550 秒 | 20 | 755.550 秒 | M0005 | 573.920 秒 | M0005 |
| Pareto 双分支 | 870.141 秒 | 11 | 740.444 秒 | 无 | — | M0001、M0003 |

Pareto 搜索完整运行实际花费 870.141 秒，仍在 900 秒名义上限内，但由于 20 轮上限而非预算耗尽结束。共同开销内，单 Active 的未晋级/拒绝工作消耗 569.242 秒；多分支所有 740.444 秒都没有产出合格候选。两组完整运行的未晋级/拒绝开销分别为 569.242 秒和 870.141 秒。多分支两个最早真实变异轮次都被拒绝，后续重复算法候选失败；单 Active 先拒绝 M0004，再从 M0003 分支晋级 M0005。

单 Active 完整成本分项：评测 441.670 秒、父模型加载 95.106 秒、训练/压缩 62.361 秒、校验 16.417 秒、profile 18.018 秒、sandbox 0.553 秒、generation 0.012 秒、retry 0 秒、未归类 121.413 秒。多分支：评测 474.749 秒、父模型加载 122.444 秒、训练/压缩 64.687 秒、校验 18.062 秒、profile 19.134 秒、sandbox 0.341 秒、generation 0.023 秒、retry 0 秒、未归类 170.700 秒。阶段分类和原始 round 记录可在各自的 `*-campaign.json`、`*-events.jsonl` 与 `*-accounting-reconciliation.jsonl` 中审计。

## 候选质量与 held-out 复测

质量是 Harness4H3 的 CLIPTemporal 聚合分数；延迟是缓存 prompt embedding 后的 denoising + VAE decode 平均时间。下表是 held-out 上的独立评测，不参与搜索或选择。

| 候选 | 来源/角色 | 开发集质量 | 开发集延迟 | Held-out 质量 | Held-out 延迟 | Held-out 模型体积 | Held-out 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|
| M0000 | 初始基线 | 0.642792 | 7.933 秒 | 0.647291 | 7.944 秒 | 13.487 GB | 19.358 GB |
| M0001 | Medium 种子 | 0.648000 | 7.634 秒 | 0.654198 | 7.932 秒 | 13.400 GB | 19.275 GB |
| M0003 | Lite 种子 | 0.644379 | 6.400 秒 | 0.641422 | 6.767 秒 | 12.718 GB | 18.654 GB |
| M0005 | 单 Active 晋级 | 0.667228 | 5.214 秒 | **0.663795** | **5.417 秒** | **12.025 GB** | **17.930 GB** |

held-out 基线质量为 0.647291，98% 对应门槛约 0.634346。M0001、M0003、M0005 都过门槛。M0005 相对 M0000 的 held-out CLIPTemporal 分数提高约 2.55%，平均延迟降低约 31.8%，模型体积降低约 10.8%，峰值显存降低约 7.4%；相对 M0001 和 M0003，M0005 的 held-out 质量更高且延迟更低。因此，本次实测的正向结果来自单 Active 搜索，多分支策略在相同开销下没有找到改进候选。

## 结论与适用边界

HarnessPro 的 Pareto 代表选择、按分支延续谱系、失败也计成本和冻结后 held-out 验证，已接入 Harness4H3 的演化路径。本轮结果不支持“多分支在此设置下更有效”：小样本、多分支消耗主要落在重复候选失败，且在共同预算内没有新晋级者；反而单 Active 找到一个在四条 held-out 提示上同时改善代理质量和部署指标的候选。建议保留多分支能力用于更宽的 Controller/算子搜索空间，但不要据本轮结果默认启用它作为更省预算的策略。

结论强度受以下限制：held-out 只有 4 个提示和一个 seed；质量指标是 CLIPTemporal 代理分，不等价于人类偏好或完整 VBench；推理成本只测 GPU 上 denoising+decode 墙钟时间，不含 prompt encoding，也不是 GPU-active 时间或服务商 token 账单；匹配开销按整轮截断，两个策略实际支出相差约 2%；阶段耗时中的一部分只能审计到 round 总墙钟，标记为未归类。本轮是一个可复核的工程对照实验，不是对多分支方法的一般性否定。

## 验证状态

本地语法检查通过；演化选择、协议、晋级和 campaign 比较相关的 34 项单元测试通过。当前环境无法运行完整 Wan 集成测试（本机没有 PyTorch/CUDA 模型运行环境）；远端实验本身完成了实际 Wan 生成、压缩、评测和 held-out 复测。比较器产物和逐轮对账记录均保存在本目录。
