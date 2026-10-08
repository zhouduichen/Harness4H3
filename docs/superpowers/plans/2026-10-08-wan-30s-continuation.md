# Wan2.1 1.3B 30 秒连续生成实施记录

## 目标

从 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 `W0000` 预训练权重出发，保留 Wan 原生 VAE、文本条件、scheduler 和 Transformer，在真实 MiniMax-H3 Teacher 在线蒸馏后，用 Wan-native latent overlap 生成约 30 秒视频。Thor/Orin、手机实时运行、LLM 自动改代码和多架构搜索不在范围内。

## 原生窗口几何

Wan 81 帧窗口在 temporal scale 4 下对应 21 个 latent frames。当前连续生成契约为：

- 8 个 81-frame 窗口；
- 相邻窗口重叠 7 个 latent frames，即 25 个 decoded frames；
- 组合长度 `81 + 7 * (81 - 25) = 473` 个原生 16 FPS 帧；
- 最终 temporal resample 到 720 帧、24 FPS、256×256，时长 30.0 秒。

每个后续窗口在每个去噪 step 将其 latent 前缀按进度锚定到上一窗口的 clean latent 尾部；解码后只对 overlap 做 crossfade，再追加新 suffix。这样不是独立短片的简单拼接。

## 首轮验收

真实 Wan/H3 训练 smoke 输出：

- Student：`/data/models/harness4h3-v27-code-long-30s/wan-student-5s-w1d/student.safetensors`
- `optimizer_steps=1`、`teacher_forward_count=4`、`offline_simulation=false`；
- Student SHA256：`b555da24bb11cfdf4bac03901635fa4eb71c2bfbda38c444f616b1331b14eb3d`；
- Teacher SHA256：`907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182`；
- W0000 parent digest：`c06eb89a9ac4b1c72f68936c859a778e22421f2938ad92bf7dff73ed3716aac9`；
- 825/825 个匹配 Wan Transformer tensors 发生变化，checkpoint 严格恢复通过。

2-step continuation smoke 输出：

- `/data/models/harness4h3-v27-code-long-30s/wan-long-w1-smoke/result.json`；
- 8 窗口、latent prefix 最大误差均为 0；
- 720 帧、24 FPS、256×256、30.0 秒；
- 黑帧、非有限像素、全帧重复、截断均未发现；
- 该 smoke 只证明连续生成和稳定性链路，不作为最终 semantic 质量结果。

正式 30-step batch 输出：

- `/data/models/harness4h3-v27-code-long-30s/wan-long-w2-30steps/result.json`；
- 2 prompts × 2 seeds，四组均为 720 帧、24 FPS、256×256、30.0 秒；
- 四组均 `continuation_verified=true`，latent prefix 最大误差为 0，黑帧/非有限像素/全帧重复/截断均为 0；
- CLIP temporal backend（12 个全片采样帧）semantic 分数为 `0.5916 / 0.6053 / 0.6097 / 0.6081`，均值 `0.6037`，aggregate 均值 `0.6896`。

## 尚未宣称完成的验收

该 semantic 均值接近既有长视频 floor，但仍低于 Teacher 约 `0.636`，且最低 case 为 `0.5916`；因此当前记录证明了 Wan-native 30 秒生成与稳定性闭环，但不宣称语义目标已经完成。下一轮应继续优化 prompt 对齐/内容保真，并用相同的多 prompt、多 seed 评估与 Teacher 对照。只有结构稳定性、continuation、`offline_simulation=false`、lineage 和 semantic floor 同时通过，才可把本目标标记为完成。

## 多 prompt child

为避免单一飞机 caption 主导训练，W2 child 从同一 W0000 parent 轮换四个真实 H3 cache caption（纸飞机、橙猫、玻璃球、黄色火车），完成 4 个真实 optimizer steps 和 16 次 Teacher forward：

- checkpoint：`/data/models/harness4h3-v27-code-long-30s/wan-student-5s-w2-multiprompt/student.safetensors`；
- Student SHA256：`9fd669b9b84c3f574e3b2d66c9c41743ee844f33137cde3709c95cbf07645840`；
- loss history：`0.1357 → 0.1586 → 0.0765 → 0.0698`；
- 825/825 个匹配 Transformer tensors 改变，严格恢复 `0 missing / 0 unexpected`，parent digest 仍为 W0000 的 `c06eb89a...71aac9`。

W3 使用该 child 的相同 2 prompts × 2 seeds 30 秒复评：

- `/data/models/harness4h3-v27-code-long-30s/wan-long-w3-multiprompt/result.json`；
- 四组仍全部满足 720 帧、24 FPS、256×256、30.0 秒、continuation prefix 最大误差 0、无黑帧/截断/全帧重复；
- semantic：`0.5933 / 0.6128 / 0.6093 / 0.6079`，均值 `0.6058`；相对 W0000 baseline `0.6042` 提升约 `+0.0016`，相对 W1d `0.6037` 提升约 `+0.0022`；
- 该提升方向正确但幅度很小，最低 case 仍低于 Teacher 约 `0.636`，所以目标继续保持 active，下一轮需要更多 prompt-diverse training/semantic calibration，而不是宣称已达到最终质量。
