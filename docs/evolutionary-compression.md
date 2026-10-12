# EvoCompress 操作说明

EvoCompress 在 Harness4H3 中增加一个独立的 DiT 结构化压缩闭环：

```text
Wan checkpoint
    ↓
profile
    ↓
AlgorithmAgent 生成受限算法
    ↓
Sandbox 校验并执行
    ↓
Block/FFN Channel Pruning
    ↓
EvaluationRecord / archive / ExperienceStore
    ↓
下一轮算法
```

## CPU 契约验证

不需要 CUDA 或真实 Wan 权重时，可以使用 fake Wan DiT：

```bash
.venv/bin/python -m harness4h3 profile-wan \
  --config configs/evolutionary-compression.fake.yaml \
  --output var/evolutionary-compression-fake/profile.json --json

.venv/bin/python -m harness4h3 compress-wan \
  --config configs/evolutionary-compression.fake.yaml \
  --algorithm configs/algorithm-prune-least-sensitive.py --json

.venv/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake.yaml --rounds 2 --json
```

每轮会写入 `parent/`、`rounds/Rxxxx/`、`algorithms/candidates/`、
`models/` 和 `experiences.jsonl`。fake 结果显式带有
`offline_simulation=true`，不能作为真实 Wan 质量或硬件结论。

## 真实 Wan smoke

编辑 `configs/evolutionary-compression.wan.example.yaml` 中的可信本地
checkpoint、Wan 生成配置、评测 manifest 和本地 CLIP 路径，在安装 `.[wan]`
依赖并具备 CUDA 的机器上运行：

```bash
python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan.example.yaml \
  --rounds 1 --json
```

真实 Diffusers checkpoint 必须配置 `evaluator.backend: wan_video`；CLI
会拒绝用 structural proxy 给真实权重打分。真实 evaluator 使用固定的
prompt/seed 生成视频，用本地 CLIP-temporal backend 评估语义、时间一致性
和运动，并把生成耗时、视频有效性和质量证据写入
`output_root/real-evaluation/`。当前 1.3B smoke 支持把 Transformer 放到
GPU，T5 encoder 和 VAE 留在 CPU。

`search-policy.json` 会跨次运行保存并恢复；实测可行且有改进的轮次提高
exploit 比例，失败轮次降低 exploit 比例。真实 adapter 对未知 Wan 模块布局
fail closed，不会把未知结构当作可安全裁剪的 Transformer block。

不要让两个命令共享同一个 `output_root` 并发运行；每个实验使用独立的
output root，避免 parent checkpoint 和 round artifacts 互相覆盖。

## 设计边界

- 算法源码只能返回 `operator` 和 `args`，不能指定路径、设备、命令或
  evaluator；
- sandbox 禁止 shell、网络、文件访问和未 allowlist 的 import；
- 结构修改必须经过 forward shape、finite parameter 和 save/reload 校验；
- 失败算法同样写入 ExperienceStore，供后续 SkillHub 避免重复失败；
- 现有 H3 operator、Wan student training 和 remote campaign 接口保持不变。
