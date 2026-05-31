# Polar Q&A 深度辨析

> *Polar: Agentic RL on Any Harness at Scale* (2605.24220v1) 精读与讨论

---

**Q1：【定义理解】传统做法如何将 harness 接入 RL 框架？给一个具体例子。**

以训练 Codex CLI agent 修 GitHub issue 为例，传统做法需要把 Codex 内部逻辑重写为框架（如 SkyRL-Agent）的 Gymnasium 风格接口：

```python
class CodexEnv(gym.Env):
    def reset(self, task):
        # 自己构造 Codex 风格的 system prompt（需逆向工程源码）
        self.messages = [{"role": "system", "content": "You are Codex..."}]
        return self._encode_obs(self.messages)

    def step(self, action):
        # 把 LLM 输出的文本翻译成 Codex 期望的 tool call JSON
        tool_call = self._parse_action_to_tool_call(action)
        result = self._execute_tool(tool_call)
        self.messages = self._codex_append_result(self.messages, result)
        return obs, reward, done, {}
```

三个核心困难：

1. **适配成本**：每个 harness 的 tool call 格式、上下文管理、停止条件都不同，需要逐一重写
2. **复现失真**：手写的 Environment 几乎不可能 100% 还原 harness 原生行为，训练出的策略在真实 harness 中可能失效
3. **闭源不可行**：Claude Code、Codex CLI 等闭源 harness，源码都看不到，无从适配

传统做法是"把 agent 拆开塞进 RL 框架"。

---

**Q2：【定义理解】`_parse_action_to_tool_call()` 在 RL 数据构造中的角色是什么？其他上下文如何用于构造 RL 数据？**

它是 RL 框架和 harness 之间的**双向协议转换器**，贯穿 episode 每一步，而非仅做一次提取。数据流：

```
RL 框架: obs_1 (token序列) → LLM 生成文本 "我需要看源码"
    ↓ _parse_action_to_tool_call() 翻译
  tool_call = {"name": "bash", "arguments": {"command": "cat groupby.py"}}
    ↓ 执行工具
  tool_result = "class GroupBy: ..."
    ↓ _codex_append_result() 拼回上下文
  obs_2 (包含历史+工具返回) → LLM 继续生成
```

每步产生的训练数据：

| 上下文部分 | 角色 | 是否可训练 |
|---|---|---|
| System prompt | 模型生成的条件（prompt_ids 的一部分） | 否（loss_mask=0） |
| User message（任务描述） | 条件 | 否 |
| LLM 生成的 assistant 文本 | 策略的输出（response_ids） | **是（loss_mask=1）** |
| 工具返回结果 | 环境反馈（下一轮 prompt_ids 的一部分） | 否 |

核心原则：**只有模型自己生成的 token 才是策略行为，只有行为才应该被训练。** 其他一切定义了"在什么条件下模型做了什么决定"，但不参与梯度更新。传统做法的翻译层每步都可能引入偏差，且偏差逐轮累积——模型在你手写的 Environment 中训练，但部署在真实 harness 中，两者上下文不同导致策略迁移失效。Polar 消除了翻译层，模型看到的始终是 harness 自己拼的上下文。

---

**Q3：【定义理解】上下文压缩等 harness 复杂策略，在传统 RL-Harness 中如何体现？**

上下文压缩（Context Compaction）指对话超长时，harness 将早期历史总结/截断/替换，腾出空间给后续交互。真实 Claude Code 的压缩行为：保留 system prompt → 用 `## Previous Progress\n{summary}` 替换历史 → 追加 continue 消息 → 摘要由内部 LLM 生成。

传统做法的三种尝试：

| 尝试 | 做法 | 问题 |
|---|---|---|
| 不实现压缩 | 对话超长时直接截断或报错 | 模型从未见过压缩后对话格式，部署到真实 harness 遇到压缩就不知如何处理 |
| 自己实现压缩 | 保留首尾 / LLM 摘要 | 摘要格式≠真实格式、内容≠真实内容、插入位置≠真实位置，每条"≠"都是训练-部署偏差 |
| 逆向工程源码 | 复现真实压缩策略 | 成本极高，harness 更新后需重做 |

更广泛的复杂策略同理：子代理调度（如何建模？单条/多条 trace？reward 怎么分配？）、prompt 重写（动态修改 prompt 如何复现？）、错误恢复（harness 的容错逻辑如何复现？）。

Polar 不受影响的原因：它不关心压缩做了什么，只看 token 前缀关系。压缩发生后前缀关系断裂，prefix merging 自动安全分链；未断裂则忠实合并。harness 的复杂策略无论怎么变，Polar 都能正确处理。

---

**Q4：【定义理解】Gateway 只拦截模型 API 请求，harness 内部不依赖模型的操作（如 hook 执行的函数）是否进入训练数据？**

分两种情况：

**情况 A：操作结果最终被喂回模型** → 间接存在于训练数据中。如 harness 调 bash 执行 pytest，输出拼进下一轮 prompt → Polar 代理捕获到该请求 → 测试输出成为 prompt_ids 的一部分，但 loss_mask=0（是环境反馈，不是模型行为）。

**情况 B：操作结果不喂回模型** → 完全不存在于训练数据中。如内部 hook 做日志记录、权限检查但不进入 prompt → Polar 和模型都不知道它发生过。

**是否有必要加入？** 没有。RL 训练的本质是在给定环境条件下优化策略行为。环境条件（含 hook 结果）是外生的、不可控的，定义了决策场景但不是决策本身。LLM 参数只能影响自己生成的 token，训练信号应且只应作用于这些 token。

微妙风险：如果 hook 改变了模型可见的上下文（过滤输出、注入指令），Polar 记录的是被修改后的上下文——这是**正确的**，因为部署时 hook 同样存在，模型应适应真实条件。Polar 不需要知道 hook 做了什么，但忠实记录了 hook 产生的效果。

---

**Q5：【定义理解】两种 trace 重建策略的详细示例？**

设定：3 轮主 agent + 1 次上下文压缩 + 1 个子代理的 session，代理捕获 5 次模型调用：

```
C1: prompt=[system, user:"修bug"]                    response="我需要看源码"
C2: prompt=[system, user, assistant, tool_result]     response="找到bug，编辑"  ← C1前缀延续
C3: prompt=[system, COMPACTED, user_continue]         response="运行测试"      ← 压缩后，前缀断裂
C4: prompt=[system_sub, user:"分析测试输出"]            response="失败原因是..."  ← 子代理，前缀断裂
C5: prompt=[system, COMPACTED, ..., subagent_result]  response="提交patch"     ← C3前缀延续
```

**Per Request**：5 条独立 trace，每条包含自己的 prompt_ids + response_ids，reward 广播到每条。简单无损，但产生大量短 trace（论文实测 3 步产生 1,185 条更新），且 outcome reward 广播导致 reward hacking。

**Prefix Merging**：先检测前缀关系分链，再链内合并。

分链结果：G1=[C1,C2]，G2=[C3,C5]，G3=[C4]。压缩、子代理、prompt 重写自动断链。

链内合并（以 G1 为例）：C2 的 prompt 包含了 C1 response 的 canonical 渲染版本，而非原始采样 token（可能存在 retokenization drift）。Polar 的处理：

```
合并后 response_ids = a1 + u1 + a2
  a1 = C1 采样的 response token（"我需要看源码"）  → loss_mask=1
  u1 = harness 插入的 interstitial（[eot, tool_result]）→ loss_mask=0
  a2 = C2 采样的 response token（"找到bug，编辑"）→ loss_mask=1
```

正确性不变式：**每个可训练 token 严格匹配行为策略的采样输出，所有非生成 token 被 loss mask 遮蔽。** interstitial 的 logprobs 用合成值占位以对齐长度，但 loss_mask=0 确保不参与训练。

| 维度 | Per Request | Prefix Merging |
|---|---|---|
| trace 数量 | 5 | 3 |
| 可训练 token | response 全部可训练 | 采样 token 可训练，interstitial mask 掉 |
| 训练效率 | 1,185 更新 / 189.5 min | 218 更新 / 35.2 min（5.39× 加速） |
| reward hacking 风险 | 高（短 trace + reward 广播） | 低（更长 trace，信用噪声更小） |

---

**Q6：【发散思考】超长轨迹是否影响训练？**

会。如果 harness 无压缩/子代理/prompt 重写，prefix merging 会将整个 session 合并为一条 trace。论文离线数据案例显示平均 51 个 assistant 轮次，长尾超 200 轮。

**两个缓解机制**：(1) 自然分链——任何破坏 token 前缀关系的操作都断链（压缩、子代理、prompt 重写）；(2) 硬性上限——推理服务器的 `max_model_len` 限制单次调用的总长度。

**但仍存在三个影响**：

1. **显存压力**：30K token 的 trace 比 3K token 的 trace 占用约 10 倍显存，限制 batch size
2. **信用分配稀释**：一条覆盖 20 轮的 trace 只有一个 reward，中间走弯路的步骤也获得相同正信号
3. **训练稳定性**：超长序列的梯度反传可能衰减或失真

论文未显式处理此问题，但实验中的 harness 都较复杂（自然分链频率高）。简单无压缩的 harness 可能产生极长单链 trace，这是未测试的边界。

---

**Q7：【定义理解】Asynchronous Rollout Staging 和 Harness Adapter 如何理解？**

两者都是工程优化，不涉及 RL 算法创新。

**Harness Adapter**：降低用户接入成本的适配器模式。封装常见 harness 的准备工作（安装配置、注册工具、写入 provider 设置、启动命令），用户只需 `"harness": "codex"` 一行配置。内置快捷适配器覆盖 claude_code、codex、gemini_cli、qwen_code、opencode、pi，另有 generic shell command harness 用于自定义 agent。本质是软件工程中的 Adapter Pattern。

**Asynchronous Rollout Staging**：提升资源利用率的流水线架构。核心问题是 agent session 各阶段资源特征不同（启动运行时=CPU+IO，harness 执行=GPU，评估=CPU+IO），顺序执行导致资源空转。Polar 用三级 worker pool + 缓冲区解耦：

```
INIT (启动运行时、装依赖) → READY (缓冲等待执行槽位) → RUNNING (执行harness) → POSTRUN (重建轨迹、评估、回调、回收)
```

两个关键细节：(1) **Evaluator Prewarm**——agent 还在 RUNNING 时就在后台准备评估用的新容器，执行完成即评估，不等容器启动；(2) **超时 session 部分恢复**——超时后仍进入 POSTRUN，重建已捕获的部分轨迹并标记 timeout status。

效果：rollout GPU 利用率从 per_request 的 20.4% 提升到 prefix_merging 的 87.7%。

---

**Q8：【定义理解】评估为何需要干净运行时？超时 session 部分恢复的目的是什么？**

**干净运行时**是为了评估的正确性。agent 执行过程中会污染环境（多次编辑、生成 `__pycache__`、安装调试工具），如果在被污染的环境中跑测试，测的不是"patch 本身是否正确"而是"被各种中间操作污染后的环境能否通过测试"。SWE-Bench 评估要求从 base commit + patch 重建干净环境来跑测试，模拟的是"PR 提交后 CI 跑的结果"。Polar 在评估时启动新容器，只应用最终 patch。

**超时 session 部分恢复**不只是提升利用率，更重要的目的是**不浪费已采集的训练信号**。超时前的模型交互包含真实策略行为，可用作 reward=0 的负样本、process reward 标注、或离线 SFT 数据筛选。"terminal timeout status" 标记让下游训练器知道这是不完整轨迹，可选择如何使用。干净运行时是科学问题（测得准不准），部分恢复是数据问题（已有信号用不用得上）。

---

**Q9：【定义理解】Wall-clock time 含义？离线 SFT 数据生成是否是 RL rollout？**

**Wall-clock time** = 墙上挂钟走过的时间，即实际等待时间。和 CPU time 不同：8 核并行跑 1 小时，wall-clock=1h，CPU time=8h。论文用 wall-clock 衡量训练实验的真实耗时。

**离线 SFT 数据生成不是 RL rollout**。两者本质区别：

| 维度 | 在线 RL Rollout | 离线 SFT 数据生成 |
|---|---|---|
| 目的 | 产生轨迹用于策略梯度更新 | 产生轨迹用于监督微调数据集 |
| 模型是否更新 | 是，每步 rollout 后 trainer 更新 θ | 否，checkpoint 固定不变 |
| reward 用途 | 计算 advantage，驱动策略改进 | 仅用于过滤（通过→保留，未通过→丢弃）|
| 训练方式 | GRPO（on-policy） | SFT（模仿好轨迹） |

Polar 复用了同一套 rollout 基础设施做两件事，区别在于：在线 RL = rollout 服务 + trainer 闭环；离线 SFT = rollout 服务独立运行，轨迹写入磁盘后不回传 trainer。

---

**Q10：【定义理解】多轮 SFT 的 loss mask 如何设计？全训 vs 只训内容有何区别？**

Polar 发布的离线 SFT 数据是多轮对话格式（平均 104 条消息、51 个 assistant 轮次），不包含 token 级 loss_mask——这由消费数据的训练框架决定。

**标准做法**：只训练 assistant 轮次的 token，其他所有轮次（system、user、tool_result）mask 掉。

**全训 vs 只训内容的区别**：都在 assistant 轮次内部讨论，system/user/tool_result 轮次都不训练。

```
assistant 轮次内容：
  [推理文本] [tool_call JSON骨架] [tool_call 参数值] [特殊token]

全训：        ✓           ✓                ✓             ✓
只训内容：    ✓           ✗                ✓             ✗
```

- **全训**：格式可靠（tool call 格式错误是致命的），工程简单（按 role 划分即可），但梯度花在确定性结构上
- **只训内容**：梯度集中在语义内容，训练信号更高效，但格式错误风险高

**通用做法是全训**——工程简单且格式可靠，格式错误的代价远大于多训几个骨架 token 的代价。只训内容更多出现在研究场景中（如探究模型是否理解 tool call 语义）。

**追问：tool call 算模型生成的内容吗？**

算。从推理服务器角度，tool call 的 JSON 结构是模型逐 token 采样生成的，全部有 logprobs，全部是策略 $\pi_\theta$ 的输出。从 harness 角度，tool call 在 assistant 消息的 `tool_calls` 字段内，而非 tool_result 的 user 消息内。

---

**Q11：【发散思考】信用分配悬置的含义？reward 如何变化？从系统层到算法层怎么理解？reward hacking 的形式与原因？**

**悬置 = 搁置未解决。** 一个 session 最终 reward=1，但包含几十轮决策，Polar 把 reward=1 广播到整条 trace 的所有可训练 token 上，无法区分哪一步真正关键。

**Reward 的变化**：在不同配置下 reward 传播方式不同。Per request 将 session reward 广播到每条独立 trace；prefix merging 将 reward 广播到每条合并链；理想但未实现的是 process reward——每步有独立评分。三者都未解决信用分配，只是噪声形态不同。

**从系统层到算法层**：Polar 的定位是系统基础设施，不负责算法创新。它完成了轨迹采集、token-faithful 重建、outcome reward 评估，但如何把 reward 精确归因到每步决策——这是算法问题。论文 §4.1 明确说 PRM 和 session normalization "outside the scope of this work"。

**Reward hacking 的形式与原因**：

形式：模型找到与成功相关但不因果的捷径。如 session_1 成功，其中 C1（"看源码"）和 C3（"编辑代码"）获得相同正 advantage，但 C1 可能只是通用操作。模型学到"不管什么任务先说'看源码'"→ 这句话总出现在成功轨迹中 → 增大其概率。

根本原因：**信用分配粒度过粗 + 负相关步骤被正 reward 掩盖。** agent 走大量弯路但碰巧成功的 session 中，所有弯路步骤都获得 reward=1 的正信号。模型学到的不是"怎么修 bug"而是"这堆操作之后有时会成功"。

Prefix merging 缓解但未根治：更长的 trace 让 GRPO 的 advantage 基于更完整的决策片段，但只要 outcome reward 是 session 级的，任何重建策略都无法根治——这是信息缺失问题，不是方差问题。

---

**Q12：【发散思考】信用分配是 Polar 特有问题还是 agentic RL 通用问题？前沿算法能否解决？**

**通用问题。** 根源是 outcome reward 的稀疏性——只在 session 结束时拿一个 0/1 信号，中间有几十步决策。这和轨迹怎么重建无关，Per request 和 prefix merging 都没有解决，只是噪声形态不同。传统单步 RL 没这个问题，因为一次生成本身就是完整策略行为。

前沿算法分析：

| 层次 | 方法 | 能否解决 |
|---|---|---|
| 更好的 outcome-level 算法（GRPO 变体、GSPO） | 缓解，不根治 | 仍在 outcome 粒度上做文章，无法区分哪步关键——信息缺失而非方差问题 |
| Value-based（PPO + Critic） | 理论有希望，实践困难 | 状态空间巨大（30K+ token 对话）、非平稳性严重、信用传递仍需长期依赖建模 |
| PRM | 最直接的解法 | 每步有独立训练信号，但训练数据瓶颈（人工标注太贵、MC rollout 太慢、弱监督有噪声） |
| 对比式学习 | 研究阶段 | 成功/失败轨迹对比定位分歧步骤 |
| 反事实估计 | 理论阶段 | "如果第 K 步换了决策会怎样"——计算量巨大 |

Math-Shepherd 是 PRM 领域的代表性弱监督方法：用最终答案对错标签自动推断每步 soft label（从第 k 步之后重新采样看成功率），但隐含假设（线性推理链、步骤可比较、采样成本低、outcome 可即时验证）在 agentic 场景下被挑战——链是树状/图状的、步骤类型异质、每次 rollout 分钟级、代码测试运行慢。

根本解法需要引入更丰富的每步信号（PRM 或等价物），但这本身又是一个数据难题。

---

**Q13：【发散思考】Agentic RL on harness 的本质目的与价值定位？harness self-evolving 的讨论？**

**价值主张的三个层次**：

```
层次 1：让模型会用某个 harness 的工具接口（操作适配）→ Codex +22.6 的大部分来自这里
层次 2：让模型在 harness 执行路径下做更好的决策（策略改进）→ 和 harness 上下文管理、工具组合相关
层次 3：让模型成为更好的程序员，harness 只是训练载体（通用能力）→ 可迁移到其他 harness
```

实验结果的暗示：Qwen Code（原生 harness）仅 +0.6，说明层次 1 和部分层次 2 已内嵌在预训练/SFT 中，简单 GRPO + sparse outcome reward 在层次 3 上几乎无贡献。**Polar + 简单 GRPO 的主要价值在层次 1（harness 适配），这本身有价值，但不应被过度解读为通用能力提升。** Polar 解锁了"让任何 harness 可被 RL 训练"这扇门，但门后面有什么取决于算法。

**追问：harness 适配是否就是目的？**

对"让模型在特定 harness 上 work"这个目标，是的。但 agentic RL 的终极目标应更高——通过 harness 交互获得可迁移的决策能力。价值定位应分两层：系统层价值（Polar 的贡献）= 让任何 harness 都可以被 RL 训练；算法层价值（Polar 不负责）= 让训练信号超越适配层，触及策略改进和通用能力。

**追问：harness self-evolving 与 Polar 的矛盾？**

Polar 隐含假设 harness 是固定的，但现实中 harness 应该也是进化的——agent 总在某类任务上卡住则 harness 优化 prompt 模板，频繁超时则调整压缩策略，tool call 格式经常出错则改进容错。这带来两个问题：

1. **轨迹一致性被破坏**：训练早期 harness v1 采集的轨迹与晚期 v2 采集的轨迹分布不一致，混合训练有风险。Polar 的 metadata 有 `policy_version` 但没有 `harness_version`
2. **黑箱假设的边界**：harness 改变行为后，Polar 只能看到前缀关系断裂，但不知道为什么断裂

更深层：当前范式是 harness=环境、agent=策略的单边优化。未来可能是 agent+harness 的联合优化——agent 学会更好地使用 bash → harness 提供更复杂 bash 工具 → harness 引入新搜索工具 → agent 需要学会使用。Polar 的黑箱方法在工程上合理但理论上不完备——固定 harness 使问题可解，但限制了只能单边优化。这是当前阶段的合理取舍，长期需突破。

---

**Polar 解决 retokenization drift 的方式是：永远不 decode→re-encode 训练用的 token。模型采样的 token 直接从推理响应里复制，interstitial 用 canonical 版但 mask 掉。两条路径各取所需，从不交叉.** 

---

场景设定
模型是 Qwen（ChatML 格式），修 bug，2 轮对话。

---
第 1 轮：C1
Harness 构建的 prompt，发给 Proxy → SGLang：
messages = [
  {"role": "system", "content": "你是助手"},
  {"role": "user", "content": "修bug"},
]
SGLang 做 3 件事：
1. 用 tokenizer 把 messages tokenize 成 prompt_ids
2. 模型采样生成 response_ids
3. 把 response_ids decode 成文本返回
SGLang 返回的响应里同时包含：
  prompt_ids    = [151644, 872, 你是助手, 151645, 151644, 770, 修bug, 151645, 151644, 77091]
                    <im_start> sys  .....   <im_end>  <im_start> usr  ...   <im_end> <im_start> asst
                                                                                             ↑ 生成开始位置
  response_ids  = [9211, 374, 264, 151645]
                  "The" "is"  "a"  <im_end>
                  (模型实际一个一个 token 采出来的)
  response_logprobs = [-0.2, -0.5, -1.1, -0.3]
Proxy 存下 C1 的 CompletionRecord：
C1.prompt_ids   = [151644, 872, 你是助手, 151645, 151644, 770, 修bug, 151645, 151644, 77091]
C1.response_ids = [9211, 374, 264, 151645]
C1.logprobs     = [-0.2, -0.5, -1.1, -0.3]
同时 Proxy 把 response decode 成文本返回给 Harness：
decoded text: "The is a"

---

Harness 执行 tool，构建第 2 轮 prompt
Harness 拿到 "The is a"，执行 tool，然后构建新 prompt：
messages = [
  {"role": "system", "content": "你是助手"},
  {"role": "user", "content": "修bug"},
  {"role": "assistant", "content": "The is a"},           ← 上一轮的回复文本
  {"role": "tool", "content": "def foo(): pass"},          ← tool 执行结果
  {"role": "user", "content": "继续"},                      ← Harness 插入
]
关键：Harness 手里只有 decode 后的文本 "The is a"，没有原始 token 序列 [9211, 374, 264, 151645]！

---

第 2 轮：C2
SGLang 把这个新的 messages tokenize：
C2.prompt_ids = [
  151644, 872, 你是助手, 151645,                        ← system
  151644, 770, 修bug, 151645,                            ← user
  151644, 77091, 9211, 374, 1109, 264, 151645,           ← assistant ← 注意这里！
                 ├────────────────────┤
                 SGLang 重新 tokenize "The is a" 的结果
                 = [9211, 374, 1109, 264, 151645]         ← 5 个 token！
                 而模型实际采样的是 [9211, 374, 264, 151645] ← 4 个 token！
                                              ^^^^
                                              多了 1109！BPE 切法变了！
  151644, 882, def foo(): pass, 151645,                  ← tool (interstitial)
  151644, 770, 继续, 151645,                             ← user (interstitial)
  151644, 77091                                          ← assistant 生成开始标记
]
这就是 drift 的发生时刻：
模型实际采样:   [9211, 374,       264, 151645]   4个token, logprobs = [-0.2, -0.5, -1.1, -0.3]
SGLang重编码:   [9211, 374, 1109, 264, 151645]   5个token, 没有对应logprobs
                            ^^^^^
                            多出来的 token！
同一个字符串 "The is a"，因为上下文不同（前面多了对话历史），BPE 的切分边界变了，导致 token 序列不同。

---

合并时 Polar 怎么做
步骤 1：前缀对齐，求 canonical tail
C1.prompt_ids 长度 L = 10
C2.prompt_ids = [...前10个和C1一样...] + [9211, 374, 1109, 264, 151645, 151644, 882, ...]
                  ├──── C2.prompt[:10] == C1.prompt ✓ ────┤├───────── canonical_tail ──────────────────────┤
步骤 2：在 canonical_tail 里找第一个 EOT（151645）切割
canonical_tail = [9211, 374, 1109, 264, 151645, 151644, 882, def foo, 151645, 151644, 770, 继续, 151645, 151644, 77091]
                   ├─ C1回复的canonical版 ────────┤  ↑EOT    ├─────────── interstitial ───────────────────────────────────────┤
                   [9211, 374, 1109, 264, 151645]         [151644, 882, def foo, 151645, 151644, 770, 继续, 151645, 151644, 77091]
                    ↑ 这和 C1.response_ids 不一样！          ↑ Harness 塞的内容，没有 raw 版本，只能用 canonical
                    ↑ 有 drift！                              但 loss=0，不影响训练
步骤 3：拼接——用 raw 替换 canonical 版的 assistant 部分
❌ 错误做法（用 canonical 版）:
   response = [9211, 374, 1109, 264, 151645,  ← C1回复的canonical版(5个token, 有drift)
               151644, 882, def foo, ...,       ← interstitial
               ...]                             ← C2.response_ids
   → logprobs 对不上！C1 只有4个logprob，这里有5个token
   → 训练信号错误
✅ Polar 做法（用 raw 版）:
   response = [9211, 374, 264, 151645,         ← C1.response_ids(4个token, 零drift)
               151644, 882, def foo, ...,       ← interstitial(canonical, loss=0)
               C2.response_ids ...]             ← raw, 零drift
               ↑ 来自 Proxy 存的原始采样记录
               ↑ 和 logprobs 完美对齐
最终产物：
prompt_ids   = [151644, 872, 你是助手, 151645, 151644, 770, 修bug, 151645, 151644, 77091]
response_ids = [9211, 374, 264, 151645,    ← raw sampled (C1), 4个token
                151644, 882, def foo, 151645, 151644, 770, 继续, 151645, 151644, 77091,  ← interstitial
                J, K, L, 151645]            ← raw sampled (C2)
loss_mask    = [1,    1,   1,   1,          ← C1回复: 可训练，logprobs完美对齐
                0,    0,   0,     0,    0,    0,    0,     0,    0,    0,     ← interstitial: 不训练
                1,    1,   1,   1]           ← C2回复: 可训练，logprobs完美对齐
logprobs     = [-0.2, -0.5, -1.1, -0.3,     ← C1真实logprobs (4个, 和4个token对齐✓)
                0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  ← 合成值
                -0.4, -0.8, -0.6, -0.2]      ← C2真实logprobs

---

如果不用 raw，整个链条就废了
假设某个训练框架用 canonical 版本训练 C1 的回复部分：
模型实际采样了 token 264，logprob = -1.1
但 canonical 版在这个位置放的是 1109
→ 训练时计算的是 log p(1109 | context)
→ 但模型采样的是 264，不是 1109
→ 梯度推向错误的方向
→ 训练信号全错了
Polar 的核心保证： loss_mask=1 的每个 token 都是模型当时实际采样出来的，对应的 logprob 是推理服务器返回的真实值——整个流程里没有任何 decode→re-encode 发生在训练用的 token 上。
