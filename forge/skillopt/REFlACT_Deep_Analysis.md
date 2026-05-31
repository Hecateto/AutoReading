# SkillOpt (ReflACT) 深度精读报告

> **一句话带走**：SkillOpt 将 LLM agent 的 skill document（Markdown 提示词）当作"模型权重"，用一套严格类比深度学习训练循环的 6 阶段 pipeline（Rollout → Reflect → Aggregate → Select → Update → Gate）对其进行迭代优化——不碰模型参数，只优化自然语言。

---

## 一、背景与痛点：Why Now?

### 领域卡点

当前 agent harness/prompt engineering 的主流做法存在一个结构性矛盾：

- **手工调 prompt 是手工作坊式劳动**——依赖人类直觉，无法系统化迭代，无法规模化；
- **自动 prompt 优化方法（如 DSPy、APE）通常做的是"单点搜索"**——在一个离散的空间里搜索最优 prompt 模板，缺乏多轮迭代的"训练感"；
- **agent skill document 不是一行 prompt，而是一个复杂的多章节 Markdown 文档**——局部修改可能引发全局连锁反应，单纯搜索无法处理这种结构性依赖。

### SkillOpt 的 unique angle

SkillOpt 的核心洞察：**优化自然语言 skill document 的过程，在结构上与训练神经网络完全同构**。这个类比不是修辞，而是架构设计的底层逻辑——每一个 DL 概念都能精确映射到一个实现组件。

| DL 概念 | SkillOpt 对应 | 功能定位 |
|---|---|---|
| 模型权重 | Skill document (Markdown) | 被优化的对象 |
| 前向传播 | Rollout | Target model 用当前 skill 执行任务 |
| 损失函数 | Task evaluator | 评估任务执行质量 |
| 反向传播 | Reflect | Optimizer 分析失败轨迹 → 生成 edit patches |
| 梯度 | Edit patches | 对 skill 的提议修改 |
| 梯度聚合 | Patch aggregation | 合并相似 edits |
| 梯度裁剪 | Edit selection | 限制每步最大 edits 数 |
| 学习率 | `learning_rate` | 每步最大 edits 数量 |
| LR scheduler | `lr_scheduler` | 衰减策略：cosine/linear/constant |
| SGD step | Skill update | 将选中的 patches 应用到 skill |
| 验证集 | Selection split | Gate 在接受前检验改进 |
| 早停 | Gate patience | 拒绝不改进的更新 |
| 动量 | Slow update | epoch 级纵向比较与指导注入 |
| 元学习 | Meta skill | 跨 epoch optimizer 策略记忆 |
| 批大小 | `batch_size` | 每次 rollout 的任务数 |
| 数据并行 | `analyst_workers` | 并行 reflection 工作线程 |

---

## 二、核心机制：The Turn

SkillOpt 的核心架构是一个 **6 阶段 step-level pipeline + 2 阶段 epoch-level 机制**，通过 `ReflACTTrainer` 类编排。

### 2.1 Step-Level Pipeline：6 阶段循环

```
for epoch in epochs:
    for step in steps:
        ① Rollout   → Target model 用当前 skill 执行任务
        ② Reflect   → Optimizer 分析轨迹，生成 edit patches
        ③ Aggregate → 层次化合并 patches
        ④ Select    → 排序 + 裁剪 edits（学习率控制）
        ⑤ Update    → 应用 edits 到 skill document
        ⑥ Gate      → 验证候选 skill，接受/拒绝
```

源码入口：`skillopt/engine/trainer.py:ReflACTTrainer.train()`

#### ① Rollout（前向传播）

- Target model（执行任务的 LLM）以当前 skill document 为 system prompt，对一批训练任务执行推理/交互
- 每个任务产生一条轨迹（trajectory）和一个评分（hard=0/1, soft=0.0~1.0）
- 支持梯度累积（`accumulation` 参数）：多批 rollout 的结果合并后再进入 Reflect

关键设计决策：
- **Target 与 Optimizer 解耦**：Target model 和 Optimizer model 可以是不同的模型/部署。Target 是"学生"（执行任务），Optimizer 是"老师"（分析失败）。两者可以不同规模、不同后端。
- **`use_eval_feedback=True`**：rollout 时打开评估反馈，让失败轨迹携带 `fail_reason` 信息

#### ② Reflect（反向传播 / 梯度计算）

这是整个 pipeline 的核心创新——**minibatch trajectory analysis**。

源码：`skillopt/gradient/reflect.py`

**关键设计：Minibatch 分析 vs 逐条分析**

类比 DL 中的 minibatch SGD vs per-sample SGD：
- 不是对每条失败轨迹独立分析，而是将 M 条轨迹分组为一个 minibatch
- 一个 optimizer 调用同时分析 M 条轨迹，识别**跨样本的共性失败模式**
- 这比逐条分析更能发现系统性问题，也更高效

**两路分离：Failure Analyst + Success Analyst**

```python
# 失败分析：识别共性失败模式 → 生成修复 patches
run_error_analyst_minibatch(skill, items, prediction_dir, edit_budget, ...)

# 成功分析：提取成功策略 → 生成强化 patches
run_success_analyst_minibatch(skill, items, prediction_dir, edit_budget, ...)
```

**Prompt 层次结构（两级优先级）**：
1. 环境特定 prompt（如 `skillopt/envs/searchqa/prompts/analyst_error.md`）
2. 通用默认 prompt（`skillopt/prompts/analyst_error.md`）

**Step Buffer Context**：Reflect 阶段会注入"本 epoch 内之前步骤"的摘要信息（失败模式 + 被拒绝的 edits），让 optimizer 避免重复提议无效修改。

**Meta Skill Context**：如果开启了 meta skill，还会注入跨 epoch 的 optimizer 策略记忆。

#### ③ Aggregate（梯度聚合）

源码：`skillopt/gradient/aggregate.py`

**层次化合并**：多个 minibatch patches 通过树状 LLM 调用层次合并

```
minibatch_0 ─┐              ┐
minibatch_1 ─┤→ merge → P_0 ├─┐
minibatch_2 ─┤              │ │→ final merge → merged_patch
minibatch_3 ─┘              ┘
```

**Failure-first 原则**：
1. 先独立合并 failure patches（并行）
2. 再独立合并 success patches（并行）
3. 最终合并：failure 组高优先级，success 组低优先级

如果最终合并失败，fallback 为简单拼接：`failure_edits + success_edits`。

#### ④ Select（梯度裁剪 / 学习率控制）

源码：`skillopt/optimizer/clip.py`

**三种学习率控制模式**（`lr_control_mode`）：

| 模式 | 行为 | 类比 |
|---|---|---|
| `fixed` | 按 scheduler 输出的 edit_budget 裁剪 | 标准梯度裁剪 |
| `autonomous` | LLM 自主决定应用几个 edits | 自适应学习率 |
| `none` | 不裁剪，全量应用 | 无裁剪 |

**Fixed 模式下的 Scheduler**（`skillopt/optimizer/scheduler.py`）：

| Scheduler | 行为 |
|---|---|
| `constant` | 固定 edit_budget |
| `linear` | 线性衰减 |
| `cosine` | 余弦退火 |
| `autonomous` | 无限制（999） |

**Autonomous 模式**（`skillopt/optimizer/lr_autonomous.py`）：
- 向 optimizer 展示当前 step 的证据（rollout 分数、候选 edits 列表）
- 让 optimizer 自主决定应用几个 edits
- 返回的整数会被 clamp 到可用 edits 数量内
- 这是一个**纯决策调用**，不涉及排序

**Rank & Select**：
- 当候选 edits 数量超过 budget，调用 optimizer LLM 进行重要性排序
- 选择 top-L 个最重要的 edits
- 排序失败时 fallback 为简单截断

#### ⑤ Update（参数更新）

源码：`skillopt/optimizer/skill.py`

**三种 skill 更新模式**（`skill_update_mode`）：

| 模式 | 行为 | 产出 |
|---|---|---|
| `patch` | 逐条应用 edit 操作到 skill | 修改后的 skill |
| `rewrite_from_suggestions` | Optimizer 根据 suggestions 重写整个 skill | 完整新 skill |
| `full_rewrite_minibatch` | 每个 minibatch 直接产出完整 skill 候选 | 多个 skill 候选 |

**Patch 模式下的 4 种 edit 操作**：

```python
EditOp = Literal["append", "insert_after", "replace", "delete"]
```

- `append`：在 skill 末尾追加（如果存在 SLOW_UPDATE 区域，则在其之前追加）
- `insert_after`：在指定文本后插入
- `replace`：替换指定文本
- `delete`：删除指定文本

**受保护区域机制**：`<!-- SLOW_UPDATE_START -->` ... `<!-- SLOW_UPDATE_END -->` 之间的内容不能被 step-level edits 修改。这是一个关键的安全防护。

#### ⑥ Gate（验证 / 早停）

源码：`skillopt/evaluation/gate.py`

**纯决策函数**——不产生副作用，只返回决策：

```python
GateAction = Literal["accept_new_best", "accept", "reject"]
```

- `accept_new_best`：候选分数 > 当前最佳分数
- `accept`：候选分数 > 当前分数（但未超过最佳）
- `reject`：候选分数 ≤ 当前分数

**注意**：Gate 是强制开启的（`use_gate=false` 会导致配置加载报错）。这是一个设计决策——没有任何 skill 更新可以绕过验证。

### 2.2 Epoch-Level 机制

#### Slow Update（动量 / EMA）

源码：`skillopt/optimizer/slow_update.py`

**动机**：step-level edits 可能在追求当前 batch 的改进时，遗忘了之前 epoch 已经学会的能力——类比 catastrophic forgetting。

**机制**：
1. Epoch 1：在 skill 末尾注入空的 `SLOW_UPDATE` 占位符
2. Epoch 2+：对同一组样本分别用上一 epoch 和当前 epoch 的 skill rollout，构建**纵向对比**（longitudinal comparison）
3. 将样本分为四类：improved（错→对）、regressed（对→错）、persistent_fail（错→错）、stable_success（对→对）
4. Optimizer 分析纵向对比，生成**战略指导**注入受保护区域

**Force-Accept 策略**：Slow update 的内容**无条件注入**到 current_skill 和 best_skill 中——不受 step-level Gate 检验。这是刻意的设计：epoch 级纵向指导不应被 step 级分数裁剪。

**纵向对比对策略**（`longitudinal_pair_policy`）：
- `mixed`：保留所有四类
- `changed`：只保留 improved + regressed（变化样本）
- `unchanged`：只保留 persistent_fail + stable_success（稳定样本）

#### Meta Skill（元学习 / Optimizer 端记忆）

源码：`skillopt/optimizer/meta_skill.py`

**与 Slow Update 的关键区别**：
- Slow Update 修改 skill document（target 端可见）
- Meta Skill 修改 optimizer 的上下文（target 端不可见）

**机制**：
- 在每个 epoch 结束时，optimizer 反思相邻 epoch 的 skill 变化和纵向对比
- 产出一个紧凑的策略记忆，在后续所有 step-level 阶段（Reflect, Aggregate, Select）注入为额外上下文
- 内容面向**未来 optimizer 调用**，而非 target agent

**注入方式**：通过 `format_meta_skill_context()` 格式化为 `## Optimizer Meta Skill` 区块，拼接到 analyst/merge/ranking 的 user prompt 中。

---

## 三、架构总览

### 3.1 代码结构

```
skillopt/
├── engine/trainer.py          # 核心训练循环 (ReflACTTrainer)
├── config.py                  # YAML 配置加载与继承
├── types.py                   # 标准化 I/O 类型定义
├── gradient/
│   ├── reflect.py             # Minibatch 轨迹分析引擎
│   └── aggregate.py           # 层次化 patch 合并
├── optimizer/
│   ├── skill.py               # Edit 应用逻辑
│   ├── clip.py                # LLM 驱动的 edit 排序与选择
│   ├── scheduler.py           # LR scheduler (constant/linear/cosine)
│   ├── lr_autonomous.py       # 自主学习率决策
│   ├── rewrite.py             # 完整 skill 重写
│   ├── slow_update.py         # Epoch 级纵向对比与指导注入
│   ├── meta_skill.py          # Optimizer 端跨 epoch 策略记忆
│   └── update_modes.py        # 更新模式切换辅助
├── evaluation/
│   └── gate.py                # 验证门（accept/reject 决策）
├── model/
│   ├── router.py              # 运行时后端路由
│   ├── azure_openai.py        # Azure OpenAI 后端
│   ├── claude_backend.py      # Claude 后端
│   ├── codex_backend.py       # Codex 执行后端
│   ├── qwen_backend.py        # Qwen 本地模型后端
│   └── common.py              # 后端公共工具
├── envs/
│   ├── base.py                # EnvAdapter 抽象接口
│   ├── searchqa/              # SearchQA 适配器
│   ├── alfworld/              # ALFWorld 适配器
│   ├── docvqa/                # DocVQA 适配器
│   ├── livemathematicianbench/# 数学 benchmark 适配器
│   ├── spreadsheetbench/      # 电子表格代码生成适配器
│   └── officeqa/              # Office 工具增强 QA 适配器
├── datasets/
│   └── base.py                # BaseDataLoader / SplitDataLoader
├── prompts/                   # 所有 LLM prompt 模板 (.md)
└── utils/                     # 工具函数
```

### 3.2 核心抽象接口

#### EnvAdapter（环境适配器）

```python
class EnvAdapter(ABC):
    def setup(self, cfg)                    # 一次性初始化
    def get_dataloader(self) -> DataLoader  # 返回数据加载器
    def build_train_env(self, ...)           # 构建训练环境
    def build_eval_env(self, ...)            # 构建评估环境
    def rollout(self, env, skill, ...)       # 执行 rollout → List[RolloutResult]
    def reflect(self, results, skill, ...)   # 分析轨迹 → List[RawPatch]
    def get_task_types(self) -> List[str]    # 返回任务类型列表
```

这个接口使得 ReflACT pipeline **环境无关**——所有环境特定逻辑被封装在 adapter 中。添加新 benchmark 只需实现 `EnvAdapter` 子类。

#### DataLoader（数据加载器）

```python
class BaseDataLoader(ABC):
    def plan_train_epoch(...) -> List[BatchSpec]   # 规划一个 epoch 的批次
    def build_train_batch(...) -> BatchSpec         # 构建训练批次
    def build_eval_batch(...) -> BatchSpec          # 构建评估批次

class SplitDataLoader(BaseDataLoader):
    # 支持 split_dir 模式和 ratio 分割模式
    # 自动管理 train/val/test 分割
```

#### BatchSpec（批次规格）

```python
@dataclass
class BatchSpec:
    phase: str           # "train" or "eval"
    split: str           # "train", "val", "test"
    seed: int            # 确定性种子
    batch_size: int      # 请求的批大小
    payload: object      # 环境特定的批次数据
    metadata: dict       # 可选元数据
```

### 3.3 模型后端架构

```
router.py（统一入口）
    ├── chat_optimizer()          # Optimizer 调用
    ├── chat_target()             # Target 调用
    ├── chat_with_deployment()    # 指定部署调用
    └── *_messages() 变体         # 消息格式 + tools 支持
        │
        ├── azure_openai.py       # Azure OpenAI（支持三种 auth 模式）
        ├── claude_backend.py     # Claude SDK
        ├── codex_backend.py      # Codex 代码执行
        └── qwen_backend.py       # 本地 vLLM
```

**关键设计：Optimizer 与 Target 的独立配置**

```yaml
model:
  optimizer: gpt-5.5           # 分析轨迹的模型
  target: gpt-5.5              # 执行任务的模型
  optimizer_backend: openai_chat
  target_backend: openai_chat
```

两套模型可以完全独立——不同的 endpoint、不同的 auth、不同的部署名称。

---

## 四、配置体系

### 4.1 结构化 YAML 配置

SkillOpt 使用基于继承的结构化 YAML 配置系统：

```yaml
# configs/searchqa/default.yaml
_base_: ../_base_/default.yaml    # 继承基础配置

train:
  batch_size: 40                   # 覆盖基础值

env:
  name: searchqa
  skill_init: skillopt/envs/searchqa/skills/initial.md
```

配置加载流程：
1. 递归解析 `_base_` 继承链
2. 深度合并（`_deep_merge`）：子配置覆盖父配置
3. 扁平化（`flatten_config`）：将结构化 section 转为 trainer 需要的 flat dict
4. CLI 覆盖（`apply_overrides`）：`--cfg-options optimizer.learning_rate=8`

### 4.2 关键超参数

| 参数 | 默认值 | 含义 | 类比 |
|---|---|---|---|
| `train.num_epochs` | 4 | 训练 epoch 数 | epoch |
| `train.batch_size` | 40 | 每次 rollout 的任务数 | batch size |
| `train.accumulation` | 1 | 梯度累积步数 | gradient accumulation |
| `gradient.minibatch_size` | 8 | 轨迹分析的 minibatch 大小 | — |
| `optimizer.learning_rate` | 4 | 每步最大 edits 数 | learning rate |
| `optimizer.lr_scheduler` | cosine | LR 调度策略 | LR scheduler |
| `optimizer.lr_control_mode` | fixed | LR 控制模式 | — |
| `optimizer.skill_update_mode` | patch | skill 更新模式 | — |
| `optimizer.use_slow_update` | true | 是否启用 slow update | momentum |
| `optimizer.use_meta_skill` | true | 是否启用 meta skill | meta-learning |
| `optimizer.longitudinal_pair_policy` | mixed | 纵向对比对筛选策略 | — |

### 4.3 实验经验

来自项目文档的 DL 直觉迁移规则：

**有效迁移**：
- Cosine schedule > constant（与 DL 一致）
- 中等 LR (4-16) > 过高/过低
- Slow update 有帮助（纵向对比防止 catastrophic forgetting）
- Meta skill 记忆改善 reflection 质量

**不完全迁移**：
- 更大 batch size ≠ 更好（API 成本递增，边际收益递减）
- 更多 epochs ≠ 更好（skill 收敛比神经网络快，2-4 epochs 通常足够）

---

## 五、Prompt 工程设计

SkillOpt 的 prompt 体系是其核心竞争力的隐性组成部分。所有 prompt 以 `.md` 文件存储在 `skillopt/prompts/` 目录下。

### 5.1 Prompt 文件结构

```
prompts/
├── analyst_error.md              # 失败分析（patch 模式）
├── analyst_error_rewrite.md      # 失败分析（rewrite 模式）
├── analyst_error_full_rewrite.md # 失败分析（full_rewrite_minibatch 模式）
├── analyst_success.md            # 成功分析（patch 模式）
├── analyst_success_rewrite.md    # 成功分析（rewrite 模式）
├── analyst_success_full_rewrite.md
├── merge_failure.md              # 失败 patches 合并
├── merge_failure_rewrite.md
├── merge_failure_full_rewrite.md
├── merge_success.md              # 成功 patches 合并
├── merge_success_rewrite.md
├── merge_success_full_rewrite.md
├── merge_final.md                # 最终合并（failure + success）
├── merge_final_rewrite.md
├── merge_final_full_rewrite.md
├── ranking.md                    # Edit 排序
├── ranking_rewrite.md
├── lr_autonomous.md              # 自主学习率决策
├── slow_update.md                # Slow update 指导生成
├── meta_skill.md                 # Meta skill 策略记忆生成
└── rewrite_skill.md              # 完整 skill 重写
```

### 5.2 关键 Prompt 设计洞察

**Analyst Error Prompt**（`analyst_error.md`）的核心指令：
- 识别**跨轨迹的共性失败模式**，非个别边缘情况
- Edits 必须**可泛化**，禁止硬编码任务特定值
- 只填补 skill 中的**空白**，不重复现有内容
- 明确告知受保护区域（SLOW_UPDATE markers），禁止修改
- 严格的 JSON 输出格式，包含 `failure_summary` 元信息

**Slow Update Prompt**（`slow_update.md`）的独特角色定位：
> "Your role is different from the per-step analyst. The per-step analyst sees individual trajectories and proposes local patches. YOU see how the skill has evolved across an entire epoch..."

- 要求**反思上一轮指导**的有效性
- 指导面向 target model（直接指令式）
- 优先级：防止回归 > 修复持续失败 > 强化成功模式

**Meta Skill Prompt**（`meta_skill.md`）面向 optimizer 而非 target：
> "Your job is not to solve tasks directly and not to write target-facing skill rules. Your job is to write a compact OPTIMIZER-SIDE memory..."

- 捕获哪种 edits 有帮助、哪种有害
- 什么样的抽象层级最适合这个环境
- 什么回归风险需要防护

---

## 六、训练循环的完整数据流

```
                    ┌─────────────────────────────────────┐
                    │          Skill Document (S₀)         │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  ① ROLLOUT: Target 执行任务           │
                    │  inputs: skill + train_batch          │
                    │  outputs: List[RolloutResult]         │
                    │  {id, hard, soft, n_turns,            │
                    │   fail_reason, task_type, ...}        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  ② REFLECT: Minibatch 分析            │
                    │  分离 failure / success 轨迹           │
                    │  按 minibatch_size 分组               │
                    │  并行调用 Optimizer                    │
                    │  → List[RawPatch]                     │
                    │  {patch: {edits/revise_suggestions/   │
                    │   skill_candidates},                  │
                    │   source_type, batch_size,            │
                    │   failure_summary}                    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  ③ AGGREGATE: 层次化合并               │
                    │  failure patches → 层次合并 → P_fail   │
                    │  success patches → 层次合并 → P_succ   │
                    │  P_fail + P_succ → final merge        │
                    │  → merged_patch                       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  ④ SELECT: 排序 + 裁剪                 │
                    │  if lr_control_mode == "autonomous":  │
                    │      LLM 自主决定 edit_budget           │
                    │  else:                                │
                    │      scheduler.step() → edit_budget    │
                    │  if edits > budget:                    │
                    │      LLM 排序 → 保留 top-L             │
                    │  → ranked_patch                       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  ⑤ UPDATE: 应用修改                    │
                    │  patch mode: 逐条 apply_edit           │
                    │  rewrite mode: LLM 重写整个 skill      │
                    │  full_rewrite: 选择最佳候选             │
                    │  → candidate_skill                    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  ⑥ GATE: 验证决策                      │
                    │  在 selection split 上 rollout          │
                    │  比较 cand_hard vs current_score       │
                    │  → accept_new_best / accept / reject   │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  更新 current_skill / best_skill       │
                    │  记录 step_buffer（失败模式+被拒 edits）│
                    │  持久化: skill_v{step}.md, history.json │
                    └─────────────────────────────────────┘

                    ────── Epoch 边界 ──────

                    ┌─────────────────────────────────────┐
                    │  SLOW UPDATE                          │
                    │  同一样本 × 两个 epoch skill → rollout  │
                    │  → 纵向对比（improved/regressed/        │
                    │    persistent_fail/stable_success）    │
                    │  → 战略指导 → 注入受保护区域            │
                    │  （force-accept，不受 Gate 检验）        │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │  META SKILL                           │
                    │  相邻 epoch skill + 纵向对比             │
                    │  → Optimizer 策略记忆                   │
                    │  （不修改 skill，只修改 optimizer 上下文）│
                    └─────────────────────────────────────┘
```

---

## 七、设计洞察与 Agent Harness 启示

### 7.1 DL 类比的价值不仅是隐喻

SkillOpt 最深刻的设计洞察在于：**DL 训练循环的每一个组件都可以在 prompt/skill 优化中找到精确对应，且这些对应不是表面类比，而是功能等价**。

这意味着：
- 已有的 DL 超参数调优经验可以直接迁移
- 理论分析工具（收敛性、稳定性）可能有跨域迁移的可能
- 工程基础设施（checkpoint、resume、分布式训练）的迁移路径清晰

### 7.2 Minibatch Analysis 是核心创新

传统的 prompt 优化方法（包括 DSPy）通常是：
- 对每个样本独立优化
- 或在全局层面做搜索

SkillOpt 的 minibatch analysis 开辟了第三条路：**在中间粒度上做跨样本的共性分析**。这比逐条分析更能发现系统性问题，又比全局搜索更高效和可控。

### 7.3 双层优化架构

Slow Update + Meta Skill 构成了一个**双层优化**结构：

```
外层 (epoch-level): Slow Update 优化 target 的 skill
                     Meta Skill 优化 optimizer 的策略
内层 (step-level):  Reflect → Aggregate → Select → Update
```

这类似于 DL 中的：
- 内层：标准梯度下降
- 外层（Slow Update）：EMA / 动量
- 外层（Meta Skill）：learning-to-learn / hypergradient

### 7.4 受保护区域机制

SLOW_UPDATE 区域的受保护设计是一个精巧的工程决策：
- Step-level edits 不能修改它 → 防止短期优化破坏长期策略
- 只有 epoch-level slow update 可以覆写它 → 确保纵向指导的权威性
- Force-accept 策略 → epoch 级指导不应被 step 级分数裁剪

这解决了 agent skill 迭代中一个常见问题：**局部优化与全局稳定性的冲突**。

### 7.5 Step Buffer 作为短期记忆

Step buffer（`_format_step_buffer`）实现了 epoch 内的短期记忆：
- 记录每步的失败模式和被拒绝的 edits
- 在后续步骤的 Reflect 中注入
- 让 optimizer 避免重复提议无效修改

这解决了一个实际问题：**LLM optimizer 倾向于重复提议相同的 edits**。

---

## 八、What Remains（余味）

1. **Gate 的纯粹性 vs 温度控制**：当前 Gate 是硬阈值（cand_hard > current_score），没有任何温度/容忍度机制。这在早期训练时可能过于激进——许多有潜力的 edits 因微小分数波动被拒绝。是否有设计"软 gate"或"patience buffer"的空间？

2. **Edit 的语义一致性验证**：当前 system 只检查 edit 的字符串匹配（target 是否存在于 skill 中），不检查语义一致性。多个 edits 可能产生矛盾——replace A→B 和 replace A→C 谁优先？

3. **Skill document 的信息容量上限**：随着训练推进，skill document 持续增长（append 为主，delete 较少）。是否存在一个最优 skill 长度？过长的 skill 是否会稀释 target model 的注意力？

4. **Optimizer 与 Target 的能力不对等**：当 optimizer 的分析能力弱于 target 的执行能力时，reflect 阶段可能无法识别真正的失败根因。这个系统隐含了一个假设：optimizer ≥ target。

5. **跨 benchmark 迁移**：README 提到了 "Transfer learning = Seed skill / cross-benchmark init"，但代码中未见系统性的迁移学习支持。这是一个有前景的方向。

---

## 九、Critical Audit（批判性审计）

**问题**：Slow Update 的 force-accept 机制绕过了 Gate 验证，这意味着一个可能降低 selection 分数的纵向指导会被无条件注入。当 slow update 的分析与 step-level 的实际 rollout 结果冲突时（例如 slow update 建议加强某策略，但该策略在当前 step 的 rollout 中已被验证为有害），系统没有任何纠错机制。

**锚点**：`trainer.py:1580-1597`，slow update 内容被 force-inject 到 current_skill 和 best_skill，并更新 sel_cache 使其 hash 对应当前分数。这意味着后续 step 如果试图回滚这个 slow update 的效果，需要通过 replace/delete edit 修改受保护区域外的内容来间接抵消——但受保护区域本身的内容无法被 step-level edits 修改。

**实质性威胁**：在 epoch 间存在 non-stationarity 的环境中（如 curriculum learning、数据分布漂移），上一 epoch 的纵向分析可能对当前 epoch 有害。Force-accept 机制缺乏退出路径，可能导致 skill document 中积累过时的战略指导。

---

## 十、Brainstorm Q&A

**Q1：如果 optimizer 和 target 是同一个模型（self-play），系统行为会有什么质变？**

Self-play 意味着 reflect 阶段的失败归因可能存在自洽性偏差——optimizer 分析的是"自己犯了什么错"，这可能导致 edits 倾向于自我辩护而非真正改进。但也可能带来一种自省能力：模型对自己失败模式的内部表征更丰富。这是一个值得实验的方向。

**Q2：Minibatch size M 的最优值是否与任务类型相关？**

从代码看 M=8 是固定默认值。但高多样性任务（如 OfficeQA 涵盖多种 office 工具）可能需要更大的 M 才能发现跨类型共性模式；而同质化任务（如纯算术）可能小 M 就够了。M 的选择实际上控制了 optimizer 的"感受野"——类比 CNN 的 kernel size。

**Q3：为什么不把 Gate 设计成 soft accept（如 EMA 更新 current_score），而是硬阈值？**

硬阈值的好处是训练轨迹完全确定性，有利于 resume 和复现。但代价是放弃了"部分有用 edits"的增量收益——一个让分数从 0.3 提升到 0.299 的候选会被拒绝。一个可能的改进：对 accept 的 edits 做 partial application（只应用其中几个），而不是全有或全无。

**Q4：Meta skill 的信息是否可能被滥用——optimizer 过度依赖历史策略而忽略当前证据？**

Meta skill prompt 中有一句："Prefer it when the current evidence is ambiguous, but do not force it if the current trajectories clearly contradict it." 这是一个软约束，完全依赖 LLM 的遵循能力。如果 meta skill 积累了过时的环境知识，它可能系统性地误导后续 optimizer 调用。一个缓解方案：给 meta skill 加入衰减机制（如只保留最近 N 个 epoch 的记忆）。

**Q5：Full rewrite 模式下，多个 skill 候选的选择标准是什么？**

从 `trainer.py:1174-1204`，full_rewrite_minibatch 模式从 merged patch 的 `skill_candidates` 列表中选取第一个包含非空 `new_skill` 字段的候选。这意味着选择逻辑是**顺序依赖**的——完全取决于 aggregate/merge 阶段的输出顺序。这是一个相对弱的选择标准，可能遗漏更好的候选。
