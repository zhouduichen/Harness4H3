# Verifier-Grounded Campaign Control Plane Design

## Goal

为 Harness4H3 建立一个可复用的 Campaign Control Plane，使 StudentCampaign 与现有 H3 OptimizationLoop 在同一组不可变目标、验证基线、候选身份、审查隔离、事件追踪和接受语义下运行。

本设计是完整 Verifier-Grounded Autonomous Training / Distillation Loop 的第一个独立子项目。它先建立可信的控制边界和可审计状态流，不假设某一种训练算法已经真实可用，也不把服务器 proxy 指标误标为端侧证据。

系统保持以下职责分离：

~~~text
Controller π       提出诊断、Parent 选择和候选实验假设
Advocate            说明实验为何值得运行
Critical Agent U    寻找反例、风险和缺失证据
Modifier m          只执行受限的结构化 proposal revision
Verifier ρ          产生确定性验证证据和指标
Gate                依据 hard constraints 与 Pareto 规则作决定
Archive             保存 Pareto、Novelty、Failure 三类历史
Experience          保存预测 delta 与实际 delta
~~~

其中 π != U != ρ 由 campaign identity 约束检查，而不是仅靠 prompt 声明。

## Scope and non-goals

### In scope

- 固定并哈希 TargetProfile、Verifier Bank、dataset、evaluation recipe、版本和 capability snapshot。
- 为每轮候选提供统一的 campaign_id / round_id / experiment_id / candidate_id / parent_candidate_id 身份链。
- 支持每轮 3–5 个结构化候选，并在训练前完成 deterministic validation。
- 提供有限轮次的 Advocate/Critical/Revision/Final Critical 审查管线。
- 提供 V0–V4 deterministic verification boundary 与 V5–V6 LLM advisory boundary。
- 提供 structured failure attribution、feasibility gate、Pareto comparison 和 promotable/target_satisfied 分离。
- 提供 append-only Decision Trace、崩溃恢复、幂等重放和 integrity failure 语义。
- 通过适配器接入现有 StudentCampaign，并为旧 H3 OptimizationLoop 留出相同接口。

### Out of scope for this sub-project

- 不在本设计中重新实现 velocity distillation、Progressive Distillation、DMD2、Recovery/Fine-tuning 或 Quantization backend。
- 不在本设计中声称已经完成真实 TargetDevice export/compile/runtime/benchmark。
- 不允许 Controller 生成 Python、shell、远程命令或新的 evaluator 代码。
- 不引入 UI、分布式工作流引擎或新的数据库；第一版使用现有文件式 stores 和 append-only JSONL。
- 不把现有 557 个通过的测试视为完整目标完成证据；它们只是回归基线。

## Current repository evidence

截至 2026-09-21，当前 nested repository Harness4H3 的全量测试基线为：

~~~text
557 passed, 2 skipped
~~~

已有基础设施包括：

- harness4h3/target/profile.py：冻结的 TargetProfile 和 objective 解析，但尚未作为 campaign-wide immutable verification base 使用。
- harness4h3/controller/loop.py：单实验 H3 loop、ModelStore、Pareto 和 ExperimentStore 集成。
- harness4h3/student/campaign.py：单 proposal/round 的 Student loop、resume 和基础事件文件。
- harness4h3/student/compiler.py：Student DSL 的 deterministic compile manifest。
- harness4h3/student/metrics.py 与 student/evaluator.py：基础指标和 generation validity 评估。
- harness4h3/memory 与 harness4h3/archive：已有 experience、trajectory、model/system archive 的局部实现。

主要缺口是：Student 与旧 H3 loop 的 identity/base 语义不统一；Student 每轮只接受一个 proposal；reviewer 不是 proposal pre-training gate；事件没有统一的 base digest 与 candidate lineage；promotable 与 campaign target_satisfied 的语义尚未由共享控制平面强制。

## Design options

### Option A — Shared campaign control plane (selected)

新增小型 harness4h3/campaign domain package，定义 CampaignBase、candidate batch、review、failure、gate 和 trace contracts；StudentCampaign 与 OptimizationLoop 通过 adapter 接入。

优点：职责集中、不会复制两套 trust boundary，后续真实训练 capability、target device verifier 和 multi-fidelity executor 都能复用。缺点：需要在旧 loop 与 Student loop 之间定义清晰的适配边界。

### Option B — Rewrite StudentCampaign only

把所有新语义直接放进 StudentCampaign。优点是第一条 Student 路径改动较少；缺点是旧 H3 loop 会保留另一套 target hash、failure code、acceptance 和 event semantics，无法满足统一 Verification Base。

### Option C — Introduce a workflow engine

用通用 event-sourced workflow engine 编排所有阶段。表达能力最强，但会增加部署和恢复面；当前项目已有 JSONL stores、remote supervisor 和 resume 机制，因此暂不采用。

## Architecture

~~~text
Immutable CampaignBase
        |
        +--> Experience/Archive context
        |
        +--> Controller π --> ProposalBatch(K=3..5)
        |                         |
        |                         +--> V0-V1 deterministic validation
        |                         |
        |                         +--> Advocate --> Critical U
        |                                         |
        |                                         +--> Revision --> Final Critical
        |
        +--> Modification Gate --> FidelityExecutor
                                      |
                                      +--> V3 Generation Verifier
                                      +--> V4 Objective/Hardware Verifier
                                      |
                                      +--> Feasibility Gate
                                      +--> Pareto/Novelty/Failure Archive
                                      +--> Experience delta update
                                      +--> next round or stop
~~~

新增 focused modules：

~~~text
harness4h3/campaign/base.py          immutable base and identity
harness4h3/campaign/events.py        append-only decision trace
harness4h3/campaign/capabilities.py real capability snapshot and action space
harness4h3/campaign/proposals.py     batch/candidate contracts and validation
harness4h3/campaign/reviews.py       advocate/critical/revision contracts
harness4h3/campaign/failures.py      deterministic failure attribution
harness4h3/campaign/gates.py         feasibility, Pareto and stop decisions
harness4h3/campaign/adapters.py      Student/H3 adapter protocols
harness4h3/campaign/__init__.py      public exports
~~~

现有 student、controller、memory 和 archive 继续拥有各自领域逻辑；control plane 不承载模型构造、SSH、训练算法或视频编码细节。

## Immutable CampaignBase

CampaignBase 是 frozen value object，创建时从 trusted configuration 和已注册 Verifier 构造。所有 Controller、Advocate、Critical、Trainer、Evaluator 只能读取其 digest 和 payload，不能通过 proposal 或 review 返回值修改。

Canonical payload 至少包含：

~~~json
{
  "schema_version": 1,
  "campaign_id": "camp_20260921_0001",
  "target_profile": {},
  "target_profile_hash": "sha256:target-profile-digest",
  "verifier_bank": {},
  "verifier_bank_hash": "sha256:verifier-bank-digest",
  "dataset_manifest_hash": "sha256:dataset-manifest-digest",
  "evaluation_recipe_hash": "sha256:evaluation-recipe-digest",
  "controller_identity": {"provider": "openai-compatible", "model": "controller-v1", "version": "2026-09-21"},
  "critic_identity": {"provider": "ollama", "model": "critic-v1", "version": "2026-09-21"},
  "evaluator_identity": {"version": "verifier-bank-v1"},
  "prompt_version": "campaign-prompt-v1",
  "capability_snapshot": {}
}
~~~

要求：

1. digest 使用禁止 NaN 的 canonical JSON；任何 field 变化都会产生新的 base digest。
2. 四个 base hash 同时写入每一个 experiment/candidate record。
3. 读取事件时，事件 base digest 必须与当前 campaign base 一致；不一致只能标记 integrity_failure，不能继续 append 到旧 campaign。
4. 三种 identity 至少包含 provider/model/version，且三组 identity 必须两两不同。
5. hard constraints 来源只能是 base；LLM payload 不得携带新的 constraint definition 或替换 verifier recipe。

## Candidate and ProposalBatch contracts

所有 executor 都必须转换到统一 candidate envelope：

~~~text
CandidateEnvelope
  candidate_id: str
  parent_candidate_id: str | null
  generation: int
  experiment_id: str
  proposal_digest: str
  mutation_fields: tuple[str, ...]
  architecture: mapping
  training_recipe: mapping
  deployment_recipe: mapping
  provenance: mapping
  predicted_metric_delta: mapping
~~~

每轮 Controller 返回：

~~~text
ProposalBatch
  batch_id
  round_id
  diagnosis
  parent_selection_evidence_ids
  candidates: tuple[CandidateEnvelope, ...]
~~~

规则：

- 候选数量默认 3–5，Controller 不得扩大 trusted campaign config 的上限。
- 每个 candidate 必须有合法 parent；初始 parent 可为 teacher-derived root candidate。
- mutation_fields 必须来自注册字段集合，不允许任意 repository path。
- Candidate digest 包含 parent ID、base digest、proposal payload 和 capability snapshot。
- 同一 batch、历史 candidate、parent relation 和 experiment ID 不能冲突。
- V0–V1 validation 失败的 candidate 只记录 failure event，不进入 review 或 executor；V2 training validity 失败的 candidate 不得进入 promotion。

## Review pipeline

Review 是训练前的 bounded advisory stage，不是最终 evaluator。

Advocate 必须输出：

~~~text
bottleneck
changed_fields
expected_metric_delta
supporting_evidence_ids
falsification_experiment
resource_assumptions
~~~

Critical 必须输出：

~~~text
objections
objection_categories
missing_evidence_ids
proxy_gaming_risks
target_device_risks
required_revisions
~~~

允许的 objection category 包括：unsupported_assumption、proxy_gaming、goal_drift、credit_assignment_error、repeated_failed_design、evaluator_blind_spot、resource_mismatch、architecture_algorithm_incompatibility、target_device_mismatch。

Revision 只能改变注册 proposal fields，并必须保留原始 candidate digest、objection IDs 和 resolved objection mapping。它不得改变 CampaignBase、hard constraints、verifier versions 或 evaluator recipe。

max_review_rounds 是 trusted campaign config。达到上限后 unresolved hard objection 会拒绝 candidate 并进入 Failure Archive，不继续 debate。最终接受/拒绝仍由实验和 Gate 决定。

## Verification hierarchy

| Level | 责任 | 允许的实现 |
|---|---|---|
| V0 | schema/security/provenance | deterministic code |
| V1 | architecture/shape/parameter | compiler and fixed checks |
| V2 | training validity | trusted worker result and checkpoint verifier |
| V3 | generation validity | fixed media/video verifier |
| V4 | objective metrics/hardware evidence | fixed verifier bank and target runtime |
| V5 | research diagnosis | LLM advisory with evidence IDs |
| V6 | research direction selection | Controller proposal constrained by V0–V4 |

V5/V6 文本不得伪装成 V0–V4 测量证据。每条 metric evidence 至少包含 metric_name、metric_version、input_reference、value、confidence_or_validity、evidence_source 和 device_profile_id。

服务器 GPU measurement 必须标记为 proxy/development evidence；只有通过 TargetDeviceProfile 的 export/compile/runtime/benchmark 才能标记 edge evidence。

## State machine and event trace

每个 campaign round 使用以下状态：

~~~text
BASE_FROZEN -> DIAGNOSED -> PARENTS_SELECTED -> PROPOSED
-> VALIDATED -> REVIEWED -> MODIFICATION_GATED -> FIDELITY_RUNNING
-> VERIFIED -> GATED -> ARCHIVED -> EXPERIENCED
-> NEXT_ROUND | STOPPED
~~~

每次合法状态变化追加一个 DecisionEvent：

~~~text
event_id
sequence
event_type
campaign_id
round_id
experiment_id
candidate_id
parent_candidate_id
actor
base_digest
payload_digest
evidence_ids
created_at
~~~

必需 event types：

~~~text
campaign.created
proposal.generated
proposal.validated
critic.completed
proposal.revised
training.started
training.metric
training.completed
evaluation.started
evaluation.completed
gate.decided
archive.updated
parent.selected
campaign.replanned
campaign.stopped
~~~

事件 payload 保存摘要和 URI/digest，不写完整 Chain-of-Thought。必须能够从 candidate ID 查询 proposal、parent、mutation、review objections、training result、evaluation evidence、gate decision 和 next-round rationale。

## Failure attribution and recovery

FailureAttributor 的优先级是：

~~~text
deterministic failure > hard experimental failure > semantic optimization opportunity
~~~

输出：

~~~text
FailureReport
  stage
  category
  responsible_variables
  evidence_ids
  deterministic_fix
  confidence
  prohibited_changes
~~~

schema、shape、missing checkpoint、hash mismatch、decode failure 和 unsupported capability 由程序归因并可给出固定修复；只有无法由程序确认的 research diagnosis 才进入 LLM context。

恢复规则：

- proposal validation failure：不启动 worker，记录 candidate-level failure，允许 batch 中其它 candidate 继续。
- review failure：candidate reject，保留 objection evidence。
- training/evaluation failure：保留 parent 为 active，child 进入 Failure Archive；不得把失败 child 当作下一轮默认 parent。
- integrity failure：立即停止当前 campaign，禁止自动 replan；需要新 campaign base。
- worker timeout 或 SSH disconnect：先检查 trusted result bundle 和 lock，再决定重试或归因；不能重复训练同一 experiment ID。
- promotable 只表示 candidate 可进入 archive 或成为 parent；campaign 只有在 target constraints、required evidence、目标指标和最小轮次均满足时才返回 target_satisfied。

## Acceptance and archive semantics

Gate 是两阶段。

### Stage 1 — Feasibility

所有 hard constraints 必须通过，包括适用时的：

~~~text
1B <= parameter_count <= 2B
video_decodable == true
peak_memory <= target
latency <= target
model_size <= target
quality >= quality_floor
~~~

缺失或无效 required evidence 等价于 infeasible；scalar reward 不能覆盖 hard violation。

### Stage 2 — Pareto and novelty

只有 feasible candidate 才比较 quality、latency、memory、energy、size、temporal consistency 等 soft objectives。Candidate 可因 Pareto 优势、novel architecture、独特 temporal behavior 或失败信息价值进入相应 archive。Rejected checkpoint 可以删除，但 proposal、mutation、review、failure、evaluation 和 attribution 必须长期保留。

Campaign stop reasons 仅允许：

~~~text
target_satisfied
budget_exhausted
no_progress
safety_or_integrity_failure
~~~

产生可解码视频只能是 promotable，不能单独作为 target_satisfied。

## Integration strategy

### StudentCampaign adapter

现有 StudentProposal/StudentCompiler 继续负责 Student DSL schema 和 compile manifest；adapter 负责：

- 将 proposal 包装为 ProposalBatch，后续支持多个 proposal。
- 将 compile report 映射为 V0–V1 evidence。
- 将 TrainingResult 映射为 V2 evidence。
- 将 StudentEvaluation 和 MetricVerifierBank 映射为 V3–V4 evidence。
- 用统一 candidate/parent IDs 写 Decision Trace 和 Experience。
- 让 accepted candidate 进入 archive，但不因 accepted 自动结束 campaign。

### Legacy H3 OptimizationLoop adapter

现有 OperatorRegistry、TargetProfile、CompositeEvaluator、ModelStore、ParetoArchive 和 ExperimentStore 保持内部实现；adapter 将 operator plan 转换到同一 candidate envelope 和 GateDecision。旧 loop 的 critical_regression 只能作为一个 typed hard-failure evidence，不能继续承担全部质量语义。

### Backward compatibility

旧 session 不具备完整 CampaignBase 时只能以 legacy mode 读取或迁移，不能与新 campaign 的 metrics 直接比较。新 campaign 的所有 events 必须包含 base digest；没有 digest 的旧事件不能伪装成新 trace。

## Testing and acceptance evidence

新增测试分四层：

1. **Contract tests**：canonical hash、unknown fields、identity separation、base immutability、candidate digest 和 event schema。
2. **Deterministic control tests**：batch size、duplicate candidate、capability filtering、V0–V1 failure 不启动 worker、V2 failure 不晋升、hard gate 优先级、promotable/target_satisfied 分离。
3. **Recovery tests**：partial event、duplicate sequence、worker disconnect、resume、rollback、integrity mismatch 和 idempotent replay。
4. **Scripted integration**：至少两个 parent 分支、3 个 candidate、一次 Critical objection、一次 rejected child、一次 accepted/promotable child，并验证第二轮 context 含 failure attribution、predicted-vs-actual delta 和完整 evidence IDs。

验收命令：

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit tests/integration
.venv/bin/python -m compileall -q harness4h3
~~~

实现完成前，原有全量回归必须仍保持至少当前基线：557 passed, 2 skipped；新增测试不能依赖 CUDA、真实 SSH、远程 LLM 或真实 H3 checkpoint。

## Rollout order

实现计划按以下顺序拆分：

1. CampaignBase、canonical identity 和 event trace。
2. Capability snapshot、ProposalBatch 和 deterministic validation。
3. Review contracts、bounded review orchestration 和 identity separation。
4. FailureAttributor、Feasibility/Pareto gate 和 stop semantics。
5. StudentCampaign adapter、scripted integration 和 resume migration。
6. Legacy H3 adapter 与全量回归。

完成本设计后，后续独立子项目再实现真实训练 capability routing、Parent/Child model archive、quality verifier bank、target device profile、multi-fidelity executor 和 Experience delta learning；这些能力必须依赖本控制平面的 contracts，而不是各自再定义一套接受语义。
