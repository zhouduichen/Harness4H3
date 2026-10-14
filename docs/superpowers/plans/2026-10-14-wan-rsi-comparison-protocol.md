# Wan 四组实验协议预检与运行清单实施计划

## 目标

实现纯 CPU 的四组 Wan 对照实验协议预检，为后续 GPU 实验生成不可歧义的运行清单，并让对照统计在缺少或不一致清单时明确拒绝“可比”结论。

## 任务

1. **实现协议校验模块**
   - 新增 `harness4h3/evolution/protocol.py`。
   - 读取四份 YAML 配置，校验公共模型、评价、预算和硬件设置。
   - 校验四组控制变量、输出目录隔离、`8 × 2 = 16` 评价单元和三次延迟测量。
   - 生成稳定协议指纹及每组 `run-manifest.json` 内容。
   - 为配置缺失、公共条件漂移、控制变量错误、输出目录冲突增加单元测试。

2. **增加命令行预检工具**
   - 新增 `tools/validate_evolution_protocol.py`。
   - 支持重复 `--run ARM=CONFIG`、预算参数、source revision、JSON 输出和写入运行清单。
   - 先完整校验四组，再原子写入清单，避免部分写入造成伪协议。
   - 增加有效配置、失败配置和清单冲突测试。

3. **让对照统计检查运行清单**
   - 修改 `tools/compare_evolution_runs.py`，读取每组 `run-manifest.json`。
   - 在缺失清单、指纹不一致、臂名称重复或四组不完整时加入明确警告，并将 `protocol_comparable` 设为否。
   - 保留既有 Active/候选/拒绝语义和历史数据，不改写旧实验。
   - 增加四组协议一致和协议不一致回归测试。

4. **更新实验说明并验证**
   - 更新 `docs/evolutionary-compression.md` 和 `docs/experiment-schema.md`，写清协议预检、清单字段和比较边界。
   - 运行新测试、现有对照测试、CPU 全量测试、Python 编译和 `git diff --check`。
   - 只提交本任务涉及文件并推送 `codex/rsi-compression-loop`；不运行完整 GPU 实验。

## 自检

- [ ] 不使用远程候选结果填补最终 Active 指标。
- [ ] 旧实验目录没有被覆盖或追认。
- [ ] 当前本机无法提供真实四组 Wan 运行数据时，报告仍明确标记证据缺口。
