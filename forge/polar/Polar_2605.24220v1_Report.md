# Polar: Agentic RL on Any Harness at Scale

> **arXiv: 2605.24220v1** | NVIDIA (Binfeng Xu, Hao Zhang 等)

---

## 一句话带走

与其把 agent harness 移植进 RL 框架，不如把 RL 的观察边界移到 harness 外面——在 LLM API 流量上挂一个代理，听模型调用即可重建可训练轨迹，harness 一行代码不改。

---

## 背景与痛点（Why Now?）

Agentic RL 的训练目标不再是单一 gym 环境，而是一个复杂的软件系统——它可能涉及异构工具链、长运行工作流、多语言实现，甚至是闭源二进制分发（如 Claude Code、Codex CLI）。传统做法是把 harness 的内部逻辑重写成框架拥有的环境 API，但这条路越来越走不通：

- 每个 harness 都需要一次框架特定的适配，且 harness 越复杂适配成本越高；
- 闭源 harness 根本不暴露内部实现，传统 RL 集成从技术上不可行。

现有低侵入方案（Agent Lightning、rLLM）降低了适配成本但未消除它——仍然要求 agent 遵循规定的 SDK 回调接口或装饰器协议。

Polar 的 unique angle：**所有 LLM-based agent 都必须调用模型 API，这个接口天然存在于 agent 外部。** 不需要理解 harness 内部如何规划、管理工具或决定何时停止，只需要在它和推理服务器之间放一个代理，监听 API 流量即可。

---

## 核心机制（The Turn）

Polar 的核心设计决策是**将 RL 的集成边界从 harness 内部移到模型 API 边界**。这不是简单的接口替换，而是一次观察视角的范式翻转——从"我需要知道 agent 怎么工作"变成"我只需要知道 agent 对模型说了什么"。

### 代理捕获（Proxy Capture）

Gateway 代理拦截 harness 的每一个模型请求，执行四步：

1. **检测 provider API**：根据请求路径和头部区分 Anthropic Messages / OpenAI Chat / OpenAI Responses / Google generateContent；
2. **归一化请求**：provider transformer 将异构格式统一为 OpenAI Chat Completions 形态，并注入训练所需字段（如 `logprobs=true`）；
3. **捕获 token 级数据**：存储请求/响应消息、prompt token IDs、采样 response token IDs、finish reason、log probabilities；
4. **回写 provider 形状**：将推理服务器响应转换回 harness 预期的格式。流式请求则先获取非流式上游响应，再发射合成的 provider 形状 SSE 流。

这个代理边界的关键属性：**它位于 agent 框架之下**。不需要理解 harness 的规划逻辑、工具管理或停止条件，只需要维持 API 兼容性并记录足够的训练信息。

### Token-Faithful Prefix Merging

Polar 提供两种轨迹重建策略，这是一个关键的 trade-off：

| 策略 | 逻辑 | 优势 | 代价 |
|---|---|---|---|
| **Per Request** | 每个 completion 独立成一条 trace | 无损、保守 | 长会话碎片化为数百短样本，训练器负担重 |
| **Prefix Merging** | 利用 append-only 对话前缀关系合并 completion | 样本数大幅缩减，GPU 利用率飙升 | 依赖 token 前缀关系严格成立 |

Prefix merging 的核心约束：新 completion 加入已有链的充要条件是——

$$p_{i_{m+1}}[1:|p_{i_m}|] = p_{i_m}$$

即严格 token 前缀关系。子代理、并行分支、上下文压缩（compaction）、prompt 重写等破坏前缀关系的操作自然形成独立链，而非被强制塞入一条全局 trace。

合并后的 trace 正确性不变式：**每个可训练 token 严格匹配行为策略的采样输出，所有非生成 token 被 loss mask 遮蔽。** 具体实现中，采样的 assistant token $a_m$ 直接复制自推理响应，canonical interstitial $u_m$ 取自服务器渲染的前缀尾部，loss mask 仅在 $a_m$ 上置 1。

### 异步 Rollout Staging

长时 agent rollout 混合了多种异构成本：运行时启动、依赖准备、harness 执行、评估器设置、测试执行、资源回收。Polar 通过 gateway 内部的隔离 worker 池解耦这些阶段：

- **INIT**：启动运行时，执行 prepare 操作；
- **READY**：缓冲已初始化运行时，等待执行槽位；
- **RUNNING**：执行 harness；
- **POSTRUN**：构建轨迹、运行评估器、执行回调、回收资源。

READY 缓冲区允许 CPU 密集的运行时准备在后台推进而不阻塞 GPU 绑定的 agent 执行。评估器预热（evaluator prewarm）在 agent 运行期间就启动评估用运行时准备。超时 session 仍会进入 POSTRUN 以恢复部分轨迹。

### Harness Adapter

按设计尽量小：安装配置、注册 MCP 服务器/skills、写入 provider 设置、返回运行 harness 的 shell 命令。内置快捷适配器覆盖 claude_code、codex、gemini_cli、qwen_code、opencode、pi 等主流 harness。

---

## 结果速览

### SWE-Bench Verified (Qwen3.5-4B + GRPO)

| Harness | Base | Polar RL | Gain |
|---|---|---|---|
| Codex | 3.8% | 26.4% | **+22.6** |
| Claude Code | 29.8% | 34.6% | +4.8 |
| Qwen Code | 34.6% | 35.2% | +0.6 |
| Pi | 34.2% | 40.4% | +6.2 |

Codex 的 22.6 点飞跃最值得关注：Qwen 模型对 Codex 的动作协议、上下文策略和 patch 提交方式完全陌生，Polar 让 GRPO 直接优化了模型在 Codex 执行路径上必须使用的行为。Qwen Code 仅 +0.6 则说明当 base checkpoint 已与 harness 充分对齐时，简单 GRPO 的边际收益有限。

### 轨迹策略消融

同等模型、硬件、拓扑下：prefix_merging 将 trainer stream 从 1,185 个 request 级更新降至 218 个 merged-trace 更新，wall-clock 时间从 189.5 分钟降至 35.2 分钟（**5.39× 加速**），rollout GPU 平均利用率从 20.4% 提升至 87.7%。

Per-request + outcome reward broadcasting 导致显著 reward hacking——request 级 trace 接收 session 级信用而缺乏适当的 session 归一化或 process reward model。

### 离线 SFT 数据生成

Qwen3.5-122B-A10B + pi harness 在 1,638 个 SWE-Gym 实例上生成 504 条接受轨迹（30.8% 接受率），耗时约 64 GPU-hours。平均每条轨迹 104 条消息、51 个 assistant 轮次。

---

## 余味（What Remains）

Polar 打开的问题空间远比它关闭的大：

- **信用分配的悬置**：prefix_merging 把 session 级 reward 广播到合并 trace 的所有可训练 token，但这本质上是把 credit assignment 问题从系统层推给了算法层。per-request 的 reward hacking 已经证明这个问题的严重性——process reward model 和 session normalization 被列为 roadmap 项目，但它们才是让 agentic RL 真正 work 的核心难题。
- **通用性边界未被测试**：所有实验集中在 SWE 代码任务和编码类 harness。代理捕获方案依赖 harness 使用标准 LLM API 格式，对于使用自定义通信协议的 harness（如嵌入式设备控制、游戏环境）是否仍然可行？
- **流式兼容性的隐性成本**：代理对流式请求的做法是先获取非流式响应再合成 SSE 流。这引入了首 token 延迟，且可能破坏依赖流式行为（如基于部分 token 提前终止）的 harness。

---

## 批判性审计（Critical Audit）

**流式请求转换的正确性假设未被验证。** 论文 §3.2 声明对流式请求"obtains a non-streaming upstream response and emits a synthetic provider-shaped stream"，但未提供任何实验验证这种合成流是否影响 harness 行为。具体地：(1) 首 token 延迟（从请求到达至首个 SSE 事件发出）必然增加，因为代理必须等待完整响应后才能开始发射；(2) 某些 harness 可能依赖流式 chunk 的到达时机来触发中间逻辑（如基于部分响应决定是否中断生成、或实时渲染 progress）。如果流式行为差异导致 harness 执行路径偏离生产环境，那么代理捕获的轨迹就不再忠实于部署行为——这直接动摇了"harness 不改、训练忠实"这一核心声明。作者未在正文或 limitations 中提及此问题。

---

## 发散与辩证（Brainstorm Q&A）

**Q1：Codex 的 22.6 点增益到底来自"学到了 Codex 的工具协议"还是"学到了更好的代码编辑策略"？**

如果是前者，增益应在切换到其他 harness 时大幅衰减；如果是后者，增益应可迁移。当前实验设计无法区分——因为每个 harness 只在自己的执行路径上评估。一个简单的验证：将 Codex-trained checkpoint 放到 Qwen Code harness 上评估，看增益保留多少。如果几乎不保留，说明 Polar 学到的主要是 harness-specific 的接口适配而非通用编程能力；这意味着"harness-native RL"的价值定位可能比论文声称的更窄。

**Q2：prefix merging 的 token 前缀检查在什么条件下会静默失败？**

前缀关系 $p_{i_{m+1}}[1:|p_{i_m}|] = p_{i_m}$ 假设 harness 的 prompt 构建是严格 append-only 的。但如果 harness 实现了某种隐式的 prompt 修改（如 re-order 已有消息、修改 system prompt 中的动态字段、注入不可见 control token），前缀关系可能在外表看起来成立但实际 token 序列已有偏移。Polar 的检测是 token 级严格相等，所以偏移会被拒绝合并（降级为独立链），不会产生错误轨迹——但频繁降级会抵消 prefix merging 的效率优势。问题在于：Polar 是否提供了诊断工具让用户知道合并率下降？如果用户只看到 GPU 利用率低而无从诊断，排查成本可能很高。

**Q3：Rollout-as-a-service 的解耦是否在训练效率上付出了隐性代价？**

Polar 的架构将 rollout 和 training 解耦为独立服务，这意味着 trainer 消费的轨迹可能来自旧 policy（stale policy）。论文引用了 Slime 的异步 GRPO 和 stale-policy step 语义，但未量化 staleness 对训练质量的影响。在简单 GRPO + outcome reward 的设置下这可能不显著，但如果换用需要 on-policy 数据的算法（如 PPO 的 importance sampling 对 stale ratio 敏感），stale trajectory 的分布偏移可能成为训练稳定性的瓶颈。这暗示 Polar 的架构优势（异步解耦）和它的算法适用范围之间存在一个未被显式刻画的边界。

**Q4：如果 harness 本身就是被训练对象而非固定环境，代理捕获范式是否仍然成立？**

论文隐含假设 harness 是固定的（作为 RL 的"环境"），但实际场景中 harness 的 prompt 模板、工具定义、上下文策略可能也需要优化。如果 harness 的配置随训练变化，代理捕获的轨迹就会混合不同 harness 配置下的行为——而 proxy 只记录了模型交互，不记录 harness 配置版本。这可能导致训练信号不一致。Polar 的 metadata 机制是否能扩展到捕获 harness 配置快照？
