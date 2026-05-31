# Agent Harness Q&A 深度辨析

> 基于 mini-swe-agent 源码精读与多轮讨论的提炼，聚焦 agent harness 设计中的关键张力与边界澄清。

---

**Q1：有状态 Shell 和无状态 Shell 的区别是什么？mini 为什么选了无状态？**

核心区别：有状态 Shell 维持一个长连接终端（如 pexpect/pty），命令在持续状态中执行；无状态 Shell 每条命令用独立的 `subprocess.run`，无状态延续。

具体例子——有状态 Shell：

```
Step 1: cd /project/src
Step 2: make              ← 自然地在 /project/src 下执行
Step 3: export DEBUG=1
Step 4: ./run_test        ← 带着 DEBUG=1 执行
```

无状态 Shell（mini 的方式）：

```
Step 1: cd /project/src && make
Step 2: cd /project/src && ./run_test          ← 必须重新 cd
Step 3: DEBUG=1 cd /project/src && ./run_test  ← 环境变量也需前缀
```

代价是命令冗余（每次重设 cwd），但收益是架构级的：

1. **可替换性**：从 local → docker 只替换了 `subprocess.run` 的调用方式。有状态 Shell 迁移到 docker 需要在容器内维持 pexpect 会话，涉及 PTY 分配、会话保活、异常后重连等完全不同的问题域。

2. **故障隔离**：一条命令搞崩 shell 状态不影响后续步骤。有状态 Shell 中 `export LD_LIBRARY_PATH=/wrong/path` 会毒化整个会话，后续所有命令全部段错误。

3. **可并行化**：独立命令天然可并行执行。有状态命令必须串行。

mini 的立场：**可替换性和故障隔离的价值，大于命令冗余的成本**。用 LM 的 cwd 前缀技巧（`cd /path && command`）弥补状态缺失，同时保留无状态的架构优势。设计选择应从 deployment requirement 逆向推导（是否要跑在 docker/modal/集群上），而非从 LM 使用体验正向推导。

---

**Q2：纯 Bash 模式下是否需要提供给模型 tool-list？相比 SWE-agent，简化掉的"操作-工具"规则映射怎么理解？**

需要区分 mini 的两种模式：

| 模式 | tool-list | 动作提取方式 |
|------|-----------|------------|
| Toolcall 模式（mini.yaml） | 有，仅 1 个 `bash` 工具 | LM 调用 `bash({"command": "ls"})`，API 返回结构化对象 |
| Text-based 模式（default.yaml） | 无，不用 tool calling | LM 输出文本中 ` ```mswea_bash_command``` ` 代码块，正则提取 |

对比 SWE-agent：

| 层次 | SWE-agent | mini (toolcall) | mini (text-based) |
|------|-----------|-----------------|-------------------|
| 工具数量 | 10+（file_read, file_edit, search...） | 1（bash） | 0 |
| 操作→工具映射 | harness 硬编码 | 无映射，LM 自行决定 bash 命令 | 无映射，LM 自行决定命令文本 |
| 工具描述 | 每工具有详细 description + schema | bash 工具有 description | prompt 中提供命令示例 |

"简化掉操作-工具的规则映射"这个理解方向对，但 mini 不是让 agent 从零发现一切——它走了一条中间路线：**通过 prompt 提供示例而非工具定义**。mini.yaml 中的 "Useful command examples" 段展示了用 bash 创建文件、sed 编辑等模式。这不是 tool description（不会出现在 function calling 的 tools 参数中），而是**示范性知识**——引导 LM 朝这些模式思考，但不限制它只用这些命令。

这本质上是 **soft tool definition**：不限制 LM 只能用定义好的工具，但通过 prompt 引导其操作模式。对比 SWE-agent 的 hard tool definition（只能用定义好的工具调用），mini 把"工具能力边界"从代码层移到了 prompt 层。

**追问：Prompt 软约束下的 LLM 概率生成是否存在 corner case？正则解析能提供硬约束吗？**

不能。正则只能做"有/无"的二值判定——要么匹配到恰好 1 个动作，要么匹配失败抛 FormatError。实际 corner cases：

1. **嵌套代码块**：LM 输出中命令本身包含 ` ``` `（如 heredoc 语法）→ 正则匹配到错误的闭合位置
2. **格式变体**：LM 可能多出空格或用 `~~~` 替代 ` ``` ` → 正则不匹配
3. **输出截断**：max_tokens 截断导致代码块不闭合 → 正则不匹配
4. **零个或多个动作**：LM 输出中没有代码块，或有两个 → 必须恰好 1 个

这些在 toolcall 模式下不会发生（API 返回结构化 `tool_calls` 对象）。FormatError 被主循环捕获后添加纠正消息，LM 在下一步尝试纠正——这是**软修复循环**，依赖 LM 的指令遵循能力，而非代码层面的硬约束。

| 维度 | Tool calling（硬约束） | Text-based + 正则（软约束） |
|------|----------------------|---------------------------|
| 格式保证 | API 层面保证 | 依赖 LM 遵循格式指令 |
| 解析可靠性 | 极高（JSON schema 验证） | 中等（corner case 导致 FormatError） |
| 格式修复 | 不需要 | 需要 FormatError → 重试循环 |
| 灵活性 | 低（只能输出 schema 定义的字段） | 高（可输出任意格式推理文本） |
| token 开销 | 低（无需格式说明） | 高（需要 prompt 详细说明格式要求） |

结论：当前 SOTA 模型指令遵循能力足以让软修复循环可靠运作，但弱模型可能陷入 FormatError 无限循环。

---

**Q3：线性历史超长轨迹是否包含大量冗余信息？对上下文管理和 SFT 训练有什么影响？**

冗余确实存在。典型 SWE-bench 轨迹 30-50 步，冗余来源：重复读取同一文件、错误路径的输出、`grep -r` 等超长匹配结果。mini 在 observation_template 中做了 10000 字符截断（保留 head 5000 + tail 5000），但仍有效率问题。

对上下文管理的影响：线性历史让压缩/记忆更困难。如果 trajectory 是结构化的（`{action_type: "file_read", file: "foo.py", content: "..."}`），压缩器可精确判断冗余；但线性历史中 file content 散布在纯文本里，压缩器必须理解语义才能判断。

对 SFT 数据的影响：不是完全不能用，但需要清洗——只用成功轨迹、对 observation 去重截断、可选地做 trajectory distillation（只保留关键步骤）。

对比压缩历史的优势不在"冗余好"，而在**原始数据的价值 > 处理后的数据**：你可以从原始数据做任何后处理，但无法从压缩数据恢复原始信息。线性历史保留了最大的操作空间。

**追问：Train-Test Mismatch 是什么？为什么线性历史能避免它？**

Train-test mismatch：**训练时模型看到的输入分布 ≠ 推理时模型实际面对的输入分布**，模型在不匹配的分布上预测，性能系统性下降。

用压缩历史训练时：模型学会在"压缩视角"下做决策。但压缩可能遗漏关键细节，模型从未见过原始输出，**不知道自己不知道什么**——无法判断压缩是否丢失了重要信息。更严重的是，压缩器本身的行为不确定（LLM-based compressor 可能对同一段输出产生不同摘要、训练/推理时版本不同），引入额外的分布偏移。

线性历史则：**轨迹 = 训练数据 = 推理输入**。零中间层，零分布偏移。模型训练时看到什么格式，推理时就是什么格式。

更深层的 mismatch 不只是格式问题，而是**信息可见性问题**。训练时压缩历史告诉模型"文件已修改"→ 模型学到"修改后直接验证"；推理时压缩历史也说"文件已修改"，但这次修改引入了语法错误而压缩器没保留细节 → 模型按训练惯性直接验证 → 出问题。模型在训练时从未经历过"压缩遗漏关键信息"的情况，推理时遇到就无法应对。

---

**Q4：这个 harness 设计是为 Coding 任务驱动的吗？迁移到通用/复杂业务场景的可扩展性如何？**

设计确实是 Coding 任务驱动的，代码层面的事实：唯一工具是 bash、完成检测依赖 magic string、提交物是 git patch、prompt 模板中的 workflow 全部是 SWE 流程。

可扩展性分析：

| 维度 | 可扩展 | 结构性不扩展 |
|------|--------|-------------|
| 环境替换 | local → docker → modal（已验证） | — |
| 模型替换 | litellm → openrouter → 自定义（已验证） | — |
| prompt 定制 | 换 YAML（已验证） | — |
| 工具体系 | — | 需要非 bash 工具 → 需改 Model 层 |
| 执行模型 | — | 需要异步/事件驱动 → 需重写主循环 |
| 记忆/状态管理 | — | 需要结构化记忆 → 需引入新 State 组件 |
| 流程编排 | — | 需条件分支/并行/子流程 → 需 workflow engine |

本质原因：mini 的架构假设了 **"单步决策 + 线性累积"** 的任务模型。当任务模型变为"多步规划 + 结构化状态 + 条件分支"时，线性 harness 的抽象层级不够了。

不同业务场景的可扩展性分级：

- **多轮信息收集型**（如客户支持）：中等。可在 bash 中 curl 调 API，但 memory 机制需大改
- **多步决策型**（如供应链优化）：低。需要结构化状态和工作流编排，核心架构需变更
- **实时交互型**（如运维监控）：极低。需要事件驱动架构，根本性变更

**追问：Harness 结构层保持通用，任务特异性通过配置层注入——怎么理解？**

SWE-bench 的提交流程（git diff → verify → echo COMPLETE_...）没有在 harness 中硬编码，而是写在 YAML 的 instance_template 中。修改提交流程改 YAML 即可，不改 Python。

这揭示的关键模式：**harness 的结构层（Agent 主循环、异常处理、消息协议）保持通用，任务特异性（工作流步骤、完成条件、输出格式）通过配置注入**。对比 SWE-agent 把提交逻辑编码为专用工具，mini 的方式使得切换新任务只改 YAML 不改代码。

但这里也有张力——prompt 中的流程指令没有类型检查、没有单元测试、没有 lint。一个 YAML 缩进错误可能导致 LM 完全不理解提交指令。代码中的流程至少有编译器/测试兜底。**prompt 作为"弱类型代码"的可靠性边界在哪里？什么复杂度的逻辑应该留在代码中而非 prompt 中？** 这是一个尚未有共识的设计问题。

---

**Q5：Coding Harness 的 Insight 向工业业务场景的迁移性如何？哪些能迁移，哪些需要推翻？**

可迁移的是架构原则，不可迁移的是具体机制。

✅ **可迁移的架构原则**：

- **三件套解耦**：业务场景中 Environment 从 `subprocess.run` 变为 `erp_client.call_api()`，Agent 和 Model 不改
- **异常驱动控制流**：业务场景有更多中断条件（审批被拒、数据缺失、权限不足），恰好适合用异常建模
- **Protocol 替代 ABC**：业务系统异构性更高，Protocol 意味着每个 adapter 无需继承公共基类
- **配置驱动 prompt**：不同业务场景 SOP 不同，YAML 配置比改代码成本低

❌ **不能直接迁移的具体机制**：

**纯 Bash 作为唯一工具**：Bash 是对文件系统的操作接口，Coding 的原子操作是文件读写编辑。但业务的原子操作是调用 REST API、查询数据库、操作 SaaS、与人交互——用 bash + curl 模拟是在用 bash 做适配层去模拟专用接口，每层转换都引入脆弱性。可迁移的 insight 不是"只用 bash"，而是"**工具接口应该尽量薄、尽量通用**"——定义少量高层业务操作而非暴露底层 API 细节。

**线性历史作为唯一记忆机制**：Coding 中勉强够用，因为文件系统是持久化的外部记忆，agent 每步可以直接 `cat file` 重新获取。但业务中关键信息散布在交互过程中（客户第 3 轮提到的偏好需要在第 15 轮使用），操作有不可逆后果（已发送的邮件无法撤回）。可迁移的 insight 不是"不要记忆机制"，而是"**轨迹存储格式应该所见即所得，记忆机制作为轨迹之上的附加层存在**"——原始轨迹永远保留作为 ground truth，记忆层（摘要/检索/结构化状态）构建其上，可替换可关闭。

**无状态执行**：业务中很多操作需要会话状态（数据库事务、API session）。可迁移的 insight 不是"不要有状态"，而是"**状态由 Environment 显式管理而非隐含在执行引擎中**"——Environment 内部维持连接池/token 刷新，Agent 仍然只发无状态命令。

**Magic string 完成检测**：业务中"完成"可能是模糊的。完成检测应建模为 **Environment 的可插拔策略**，不同场景注入不同检测逻辑。

**追问：这些可迁移 insight 的工作主要是工程职能还是算法职能？**

传统归属上，2.2 中的可迁移项主体是工程。但 agent harness 的特殊性在于——**harness 不是包住算法的壳，它是推理时算法的具现**。传统 ML 中推理算法就是 `model.forward()`，几行代码；agent 中推理算法是整个 harness：prompt 模板、observation 格式、异常处理、完成检测，每一个都在影响 agent 的行为分布。

边界模糊的本质不是"大家都在学全栈"，而是 **harness 这个对象本身就横跨了算法和工程两个维度**。判断标准：

> **如果我改了这个设计选择，agent 在相同任务上的成功率是否会系统性变化？**

- 三件套解耦的边界线改了 → prompt 结构变 → 成功率变 → **算法问题**
- 异常 vs 标志位 → trace 结构变 → 训练数据变 → **算法问题**
- YAML 解析库换了 → 行为不变 → **纯工程**

LLM 算法工程师需要掌握的"工程"能力不是前端或 K8s，而是几项在 agent 时代变成算法核心能力的技能：**接口抽象能力**（Protocol 边界决定训练数据形状和 reward 归因）、**配置系统的设计直觉**（放 prompt 的东西 LM 可适应，放代码的是硬约束——这定义了 agent 的学习空间边界）、**控制流的因果推理**（异常驱动 vs 标志位决定 trace 因果结构，影响 RL 训练的可归因性）。

---

**Q6：异常驱动控制流为什么和 RL 训练不兼容？**

RL 训练需要将交互过程建模为 MDP：`State → Action → Reward → State' → ...`，每个 `(s, a, r, s')` 是一个训练样本。

正常步骤没有问题。但异常中断的步骤打破了因果链：

**Submitted 异常**：当 Environment 检测到 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 时 raise Submitted，异常穿透多层调用到达主循环。问题在于——这个 +1 reward 应该归因给哪个 action？是触发 magic string 的那条命令？还是之前所有命令的累积贡献？异常的"穿透性"使 RL 框架无法追踪"哪个具体 action 触发了完成条件"。

**FormatError 异常**：LM 输出格式错误 → raise FormatError → 添加纠正消息。这是谁的"错"？LM 没遵循格式应该给负 reward？还是 harness prompt 不够清楚不应惩罚 LM？格式错误后的纠正消息应该算 state 的一部分还是独立的 transition？

**LimitsExceeded 异常**：这根本不是 LM 的 action，是 harness 的硬限制，但轨迹突然终止需要 terminal reward。给多少？-1？0？因果归因应该分配给第 N 步还是前面所有步的累积？

核心矛盾：异常驱动的设计意图是**底层组件无需知道谁在调用它们，只需在条件满足时 raise**——这是控制权反转，利于推理时的可扩展性。但 RL 需要**每个状态转移有明确的因果归因**——异常的穿透性正好破坏了这条因果链。

RL 友好的改法：完成信号从异常变为返回值——`_check_finished(output)` 返回 `(finished: bool, submission: str)` 而非 raise。这样每步的 `(s, a, r, s')` 完整可追踪。但这放弃了异常驱动的架构优势——扩展时需要在每个步骤的返回值中传播状态。

---

**Q7：Harness-Agent 联合训练中，截断 trace 作为奖励信号——是纠偏还是误导？一个不够好的 harness 是否会限制 agent 的进化？**

同一信号（截断=失败）对不同质量的 trace 产生截然相反的训练效果：

**场景 A（截断=合理纠偏）**：agent 反复 cat 同一文件、重复 grep 相同模式、漫无目的搜索 → step_limit 截断 → 负 reward → 模型学到"不要冗余操作，要高效决策" ✅

**场景 B（截断=有害误导）**：agent 仔细阅读代码库建立全局理解 → 设计修复方案 → 实施修复 → 集成测试发现更深 bug → 修复更深层问题 → step_limit 截断，但本来再需 3 步就能完成 → 负 reward → 模型学到"不要深入探究，赶紧提交半成品" ❌

这引出一个更根本的问题：**这种训练到底是在训练 agent 服从 harness，还是二者共同进步？**

当前实践几乎都是固定 harness、训练 LM——因为工程方便（harness 是 Python 代码不可微，LM 是神经网络天然可训练）。但这意味着 agent 的进化受限于 harness 的能力天花板：

1. **能力天花板**：harness 定义了 action space。纯 bash harness 下 LM 永远学不会"直接调用数据库 API"——不是能力不够，是 harness 不提供这个接口。训练再多也无法突破 action space 的边界。

2. **奖励信号污染**：如果 harness 的 step_limit=30 但任务平均需要 35 步，100% 的成功 trace 都经过了截断后的"紧急提交"，模型学到"赶工"策略。这不是 agent 的问题，是 harness 设定了错误的学习目标。

3. **过拟合到 harness 特征**：模型可能学到 harness 的特有模式而非任务本质规律。mini 的 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 是 40 字符的 magic string，模型可能学到"输出特定长度命令触发完成"而非"确认修复完成后再提交"。换一个 harness（不同 magic string 或不同完成机制），策略失效。

更理想的框架：**harness 应该提供分辨力而非偏好**。分辨力 = 能区分"agent 真正解决了问题"和"agent 碰巧通过测试"；偏好 = 强加"必须在 30 步内完成"作为 reward 硬条件。step_limit 是工程约束（防成本失控），不是学习目标。LimitsExceeded 不应该给负 reward，而应该**调整 harness**（提高 limit 或优化 prompt 效率）。

最终，这指向一个尚未被充分探索的设计空间：**harness 应该具有 meta-cognition——关于自身性能的认知**。当截断频繁发生时，harness 应自动标记"此处需要调整"而非默默产生负 reward 训练信号。harness 不只是执行框架，还要能诊断自身的不足。
