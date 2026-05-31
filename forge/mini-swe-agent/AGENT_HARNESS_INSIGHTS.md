# Agent Harness Design Insights — 从 mini-swe-agent 提炼

> 面向 LLM 算法工程师的 Agent Harness 设计原则提取
> 基于对 mini-swe-agent v2.3.0 全量源码的精读，聚焦 harness 架构层面的可迁移 insight

---

## The One-Liner

**当 LM 足够强时，harness 的核心价值不在于为 LM 提供多少工具，而在于用最少的结构约束换取最大的可替换性与可审计性。**

mini-swe-agent 的真正贡献不是一个"更简单的 agent"，而是一个实证：在 SWE-bench 这类复杂任务上，一个 170 行的线性 harness 能达到与精心设计工具系统的 SWE-agent 相近的性能——这暗示 2024 年我们投在 tool interface 设计上的大量精力，可能是在替 LM 做它自己就能做的事。

---

## 一、Why Now：领域卡点与回应

### 1.1 旧范式的张力

SWE-agent（2024）的隐含假设是：**LM 能力不足，需要 harness 通过精细的工具接口（ACI, Agent-Computer Interface）来弥补**。这催生了大量工程投入——自定义工具定义、特殊命令语法、多工具编排、历史压缩器等。

一年后，LM 能力跃升（Claude Sonnet 4、GPT-5），但 harness 设计并未同步反思：大多数 agent 仍在堆叠工具、加复杂路由逻辑，harness 日益成为"过拟合的研究制品"——在 benchmark 上刷分有效，但脆弱、难复现、难迁移。

### 1.2 mini-swe-agent 的 unique angle

它的视角是**逆向的**：不是"LM 需要什么"，而是"LM 不再需要什么"。这产生了一个极有价值的实验对照：

> **如果剥离所有工具设计，只给 bash，性能会跌多少？**
> 答案：几乎不跌。SWE-bench Verified >74%。

这个对照本身是比任何算法贡献都重要的 insight——它标定了 harness 复杂度的边际收益递减点。

---

## 二、The Turn：五个核心设计决策及其 Trade-off

### 决策 1：唯一工具 = Bash（放弃工具多样性）

**选择 A（SWE-agent）**: 为每种操作设计专用工具（file_read, file_edit, search 等），每工具有独立接口和约束。

**选择 B（mini）**: 只给 bash，让 LM 自行组合 shell 命令。

| 维度 | 多工具 | 纯 Bash |
|------|--------|---------|
| LM 自主性 | 低（受工具定义约束） | 高（shell 是图灵完备的） |
| 格式可靠性 | 高（tool calling 结构化） | 取决于 LM 遵循指令能力 |
| 模型兼容性 | 仅支持 tool calling 的模型 | 任意模型（含纯文本） |
| 沙箱要求 | 需在沙箱内安装工具依赖 | 只需 bash |
| 可迁移性 | 低（工具定义与任务耦合） | 高（bash 是通用接口） |

**深层 insight**: 工具多样性本质上是在 **harness 侧编码领域知识** vs **让 LM 自行发现领域知识** 之间的取舍。当 LM 能力足够时，后者的边际成本更低、泛化性更强。这是一个类似"编译器优化 vs 程序员手动优化"的张力——随着底层能力提升，手写优化的收益递减。

**对 harness 设计的启示**: 设计工具体系时，先问"这个工具是否在替 LM 做它已经会做的事？"。如果是，删掉它——减少工具数量意味着减少格式错误概率、减少 prompt 长度、减少维护负担。

### 决策 2：subprocess.run 而非有状态 shell（放弃执行连续性）

**选择 A**: 维持一个长运行 shell 会话（如 pexpect/pty），命令在持续状态中执行。

**选择 B（mini）**: 每条命令用独立的 `subprocess.run`，无状态延续。

**trade-off 分析**:

```
有状态 shell:  cd /project && make     ← 自然、直觉
无状态 shell:  cd /project && make     ← 每次必须重设 cwd
```

这看起来是 B 的劣势，但 mini 揭示了更深层的考量：

1. **可替换性**: `subprocess.run` → `docker exec` 是一行替换。有状态 shell → docker 需要完全重写会话管理。
2. **故障隔离**: 一条命令搞崩 shell 状态不影响后续步骤。有状态 shell 中，一个 `export LD_LIBRARY_PATH=...` 错误可能毒化整个会话。
3. **可审计性**: 每步独立，轨迹天然可重放。有状态 shell 的可重放性依赖于精确复现 shell 状态。
4. **可并行化**: 独立命令天然可并行执行。有状态命令必须串行。

**深层 insight**: 这不是"便利性 vs 安全性"的简单取舍，而是**harness 的首要设计目标是什么**的选择。如果首要目标是"让 LM 用得舒服"，有状态 shell 更好；如果首要目标是"让 harness 可替换、可扩展、可审计"，无状态执行更好。mini 用 LM 的 cwd 前缀技巧（`cd /path && command`）弥补了状态缺失，同时保留了无状态的架构优势。

**对 harness 设计的启示**: 执行模型的选择应从 **deployment requirement 逆向推导**，而非从 LM 使用体验正向推导。如果你的 harness 可能运行在 docker、modal、远端集群上，无状态执行是结构性优势。

### 决策 3：线性历史（放弃历史处理）

**选择 A**: 使用历史处理器（history processor）——摘要、滑动窗口、关键信息提取等。

**选择 B（mini）**: 纯 append，轨迹 = 发送给 LM 的消息列表。

```python
# mini 的整个历史管理逻辑：
self.messages.extend(messages)  # 就这一行
```

**trade-off**:

| 维度 | 历史处理 | 线性 append |
|------|---------|-------------|
| 上下文效率 | 高（可压缩冗余） | 低（原始轨迹可能超长） |
| 可调试性 | 低（发送给 LM 的 ≠ 保存的） | 高（所见即所得） |
| 微调友好度 | 低（训练数据 ≠ 推理时输入） | 高（轨迹直接可用于 SFT） |
| 实现复杂度 | 高 | 极低 |
| 一致性保证 | 弱（压缩可能丢失关键信息） | 强（无信息损失） |

**深层 insight**: 线性历史的真正价值不是"简单"，而是**消除了 harness 中最危险的隐式状态——历史处理器对信息的筛选**。在复杂 agent 中，历史处理器是最难调试的组件：LM 行为异常时，你永远不确定是 LM 的错还是历史处理器丢掉了关键上下文。线性历史让这个归因问题消失。

同时，线性历史直接解锁了 **trajectory → training data** 的管道：保存的轨迹就是 SFT 的训练样本，无需任何后处理。这对于做 FT/RL 的研究者是结构性优势。

**对 harness 设计的启示**: 历史处理是一个"可选的优化层"而非"必要的架构层"。先实现线性版本，只有当上下文长度成为实际瓶颈时才引入压缩——而且压缩层应该是可插拔的、可关闭的，而非嵌入主循环。

### 决策 4：异常驱动控制流（放弃状态标志）

**选择 A**: 用状态变量（`self.status = "finished"`）+ 条件分支管理流程。

**选择 B（mini）**: 用异常体系管理所有流程中断。

```python
# mini 的主循环
while True:
    try:
        self.step()
    except InterruptAgentFlow as e:
        self.add_messages(*e.messages)
    if self.messages[-1].get("role") == "exit":
        break
```

异常层级：
```
InterruptAgentFlow          ← 基类，携带消息
├── Submitted               ← Environment 检测到任务完成
├── LimitsExceeded          ← Agent 检测到步数/费用超限
│   └── TimeExceeded        ← 时间超限
├── FormatError             ← Model 检测到输出格式错误
└── UserInterruption        ← 用户中断（InteractiveAgent）
```

**为什么异常比标志位更好**：

1. **跨层级穿透**: `Environment.execute()` 在最底层检测到完成条件，异常自然穿透到 `Agent.run()` 的顶层循环。用标志位则需要层层 return 和检查。

2. **子类化友好**: `InteractiveAgent` 重写 `execute_actions()` 加入用户确认，只需 `raise UserInterruption()`——不需要修改父类的任何循环逻辑。用标志位则需要在每个检查点都加 `if self.status == "interrupted"`。

3. **异常天然携带上下文**: 每个 `InterruptAgentFlow` 子类自带消息（dict），这些消息自动进入轨迹。用标志位则需要额外的"挂载消息"机制。

4. **不可忽视性**: 异常不会被"忘记处理"。标志位可以在新增代码路径中被遗漏检查。

**深层 insight**: 异常驱动控制流的本质是**将控制权从调用者转移到被调用者**——底层组件无需知道"谁在调用我"，只需在条件满足时 raise；顶层循环无需知道"底层发生了什么"，只需统一 catch。这种 **控制流反转** 是 harness 可扩展性的关键：新增一种中断条件只需定义新异常类，不需要修改任何已有代码。

**对 harness 设计的启示**: agent harness 中的控制流本质上是"事件驱动的循环"——query → execute → observe → query。任何可能中断这个循环的事件（完成、超限、格式错误、用户干预）都应该建模为异常而非返回值或标志位。这使得主循环保持极简，扩展点通过异常类型自然打开。

### 决策 5：Protocol 而非抽象基类（放弃编译时类型强制）

```python
class Model(Protocol):
    config: Any
    def query(self, messages, **kwargs) -> dict: ...
    def format_message(self, **kwargs) -> dict: ...
    ...

class Environment(Protocol):
    config: Any
    def execute(self, action, cwd="") -> dict[str, Any]: ...
    ...
```

**trade-off**:

| 维度 | ABC（抽象基类） | Protocol |
|------|-----------------|----------|
| 运行时检查 | 可在实例化时检查 | 不检查 |
| 第三方集成 | 必须显式继承 | 鸭子类型即可 |
| 多继承冲突 | 可能遇到 MRO 问题 | 无继承链 |
| 类型提示 | 需要运行时才能验证 | 静态分析可用 |

**深层 insight**: Protocol 的选择不仅仅是技术偏好，它反映了对 **harness 边界** 的理解：harness 不应该强制外部组件的内部结构，只应该规定交互协议。这和 Unix 哲学"管道协议比实现更重要"一脉相承——只要你的 Model 产出了正确格式的 dict，我不关心你是 litellm、openrouter 还是自己写的。

**对 harness 设计的启示**: harness 的三大组件（Agent / Model / Environment）的接口定义应使用 Protocol 或类似的轻量约束，而非重量级的 ABC。这使得生态中的任何人都能用最小成本接入——不需要 import 你的基类，不需要适配你的构造函数签名。

---

## 三、跨决策的结构性张力

### 张力 1：可审计性 vs 执行效率

线性历史 + 无状态执行 + 异常驱动 = 极致的可审计性（每一步都可追溯、可重放、可序列化），但代价是执行效率（重复 cd、重复设环境变量、无上下文压缩）。

mini 的立场：**可审计性是 harness 的首要非功能性需求**。原因：在 agent 研究中，理解 LM 为什么做了某个决策比让它做得更快更重要。效率问题留给 LM 自行优化（如写入临时文件而非反复 cd），harness 不越俎代庖。

这暗含一个假设：**LM 的指令遵循能力足够强，能够自行适应无状态执行模式**。如果 LM 总是忘记 cd 回正确目录，这个决策就会失败。74% 的 SWE-bench 成功率表明当前 SOTA LM 已跨过这个阈值。

### 张力 2：极简接口 vs 任务特异性

纯 bash 接口是通用的，但特定任务（如 SWE-bench 提交 git patch）有固定流程。mini 的解法不是在 harness 中硬编码流程，而是**将流程知识注入 prompt**：

```yaml
# swebench.yaml 的 instance_template 中
## Submission
Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt`
...
Step 3: Submit (EXACT command required)
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

这揭示了一个关键模式：**harness 的结构层保持通用，任务特异性通过配置（prompt）层注入**。对比 SWE-agent 把提交逻辑编码为专用工具，mini 的方式使得切换到新任务时只需改 YAML，不改代码。

### 张力 3：完成检测的归置

"agent 何时算完成"是 harness 中最微妙的设计点。mini 把完成检测放在 Environment 层（`_check_finished` 检测 magic string），而非 Agent 层。这意味着：

- Agent 不知道完成条件是什么 → Agent 代码更通用
- Environment 定义完成语义 → 换 Environment 自然换完成逻辑
- 完成信号通过异常（`Submitted`）传播 → 不污染返回值类型

但这也带来了一个隐含风险：完成条件依赖 LM 输出特定字符串（`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`），这是一个**跨组件的隐式协议**——prompt 必须教 LM 输出这个字符串，Environment 必须检测这个字符串。任何一端的修改都可能破坏这个协议，而类型系统无法捕获这种破坏。

---

## 四、从源码提取的可迁移 Harness 模式

### 模式 1：三件套解耦（Agent-Model-Environment）

```
Agent:  控制 flow（query → execute → observe 循环）
Model:  控制 LM 交互（调用 API、解析输出、格式化观测）
Environment: 控制执行语义（在哪执行、怎么执行、何时算完成）
```

可迁移原则：
- Agent 不知道命令在哪执行（local? docker? modal?）
- Model 不知道命令怎么执行（只负责解析和格式化）
- Environment 不知道 LM 是什么（只负责执行和检测完成）
- 三者通过 dict 协议通信，无交叉依赖

### 模式 2：消息总线（Messages as Single Source of Truth）

```
messages: list[dict]  ← 唯一的状态载体
```

所有组件只做一件事：往 messages 里 append dict。无隐藏状态、无旁路通信、无共享可变对象。这使得：
- 序列化 = `json.dumps(messages)`
- 调试 = `print(messages[-1])`
- 微调数据 = messages 本身
- 可重放 = 清空 messages，重新执行

### 模式 3：配置驱动的 Prompt 工程

```yaml
agent:
  system_template: |
    You are a helpful assistant...
  instance_template: |
    Please solve: {{task}}
    ...

model:
  observation_template: |
    <returncode>{{output.returncode}}</returncode>
    <output>{{output.output}}</output>
  format_error_template: |
    Format error: {{error}}...
```

关键 insight：**observation 的格式化是 model 的职责而非 environment 的职责**。这看似违反直觉（environment 产出数据，不该由它格式化吗？），但实际是为了支持**同一个 environment 的输出适配不同 LM 的输入格式**——toolcall 模型需要 `{"tool_call_id": ..., "role": "tool"}` 格式，text-based 模型需要纯文本格式。让 model 决定如何呈现 observation，environment 只产出原始数据。

### 模式 4：递归合并配置（配置即组合）

```python
config = recursive_merge(
    default_yaml_config,
    task_specific_yaml_config,
    cli_overrides,
    env_var_overrides,
)
```

用 `UNSET` 哨兵实现"删除键"语义。这使得配置可以通过任意数量的层叠加，且每一层都是独立可测试的。

### 模式 5：可插拔的动作解析策略

```
LitellmModel           → parse_toolcall_actions()      # function calling
LitellmTextbasedModel  → parse_regex_actions()         # 正则提取
LitellmResponseModel   → parse_toolcall_actions_response()  # Responses API
```

三种策略共享同一套后续流程（execute → format_observation → add_messages），仅解析层不同。这展示了 harness 中**策略模式**的正确用法：变化的封装在 Model 层，不变的部分在 Agent 层。

---

## 五、What Remains：mini 打开的问题空间

### 5.1 线性历史的上下文窗口瓶颈

当任务需要 100+ 步时，原始轨迹可能远超 128K 上下文。mini 的态度是"不处理"——但这在长任务上是真实瓶颈。如何在保持"轨迹 = 消息"的透明性前提下引入上下文压缩？滑动窗口？摘要？检索增强？这个设计空间是 open 的。

### 5.2 完成检测的鲁棒性

依赖 magic string（`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`）的完成检测是脆弱的：
- LM 可能输出这个字符串但并未真正完成
- LM 可能完成了但输出了略微不同的变体
- 任何 prompt 修改都可能破坏这个隐式协议

是否有更鲁棒的完成信号机制？例如：harness 层面的完成检测器（而非纯字符串匹配），或多信号融合判定。

### 5.3 多动作编排的空白

mini 支持单步多 tool call（`parallel_tool_calls: true`），但不支持跨步的动作编排（如"先 A，再根据 A 的结果决定是否 B"）。当前的每步都是独立的 query-execute-observe 循环。对于需要条件分支的任务流程，harness 层面是否需要更丰富的编排原语？还是应该继续信任 LM 的自主决策？

### 5.4 错误恢复策略的缺失

当 LM 犯错（错误命令、格式错误）时，mini 依赖 LM 自行从 observation 中学习并纠正。没有显式的错误恢复策略（如回滚文件变更、重置环境状态）。对于不可逆操作，这是否足够？

### 5.5 Harness 的评估方法论

mini 证明了"更简单的 harness 可以同样好"，但这个"同样好"是用 SWE-bench 的通过率衡量的。如果换一个评估维度——例如 token 效率（每 dollar 解决多少问题）、鲁棒性（同一任务多次运行的一致性）、可解释性——极简 harness 是否仍然占优？我们缺乏一个 **harness 本身的评估框架**。

---

## 六、Critical Audit

### 审计 1：极简 harness 的性能声称缺乏消融实验

mini 声称"和 SWE-agent 性能相近"，但项目内没有任何对照实验。74% 的 SWE-bench Verified 数字来自特定模型（Claude Sonnet 4）+ 特定 prompt + 特定配置的组合。我们无法区分：

- 74% 中有多少归功于 harness 架构（线性/无状态/纯bash），多少归功于 prompt 质量，多少归功于模型能力？
- 如果给 SWE-agent 同样的 prompt 但保留其工具系统，性能差异是多少？

没有这个消融，"极简 harness 足够好"的结论可能是"强模型掩盖了 harness 缺陷"的假象。

### 审计 2：异常驱动控制流的组合爆炸风险

mini 的异常层级目前只有 5 种，管理简单。但在复杂 harness 中，异常类型的组合可能产生意外交互。例如：
- `FormatError` 被捕获后添加修正消息 → 消息超长 → `LimitsExceeded` → 需要另一个 catch
- `UserInterruption` 发生在 `Submitted` 的传播路径上 → 完成信号丢失

当前的 `while True` + 单一 catch 块没有处理异常组合的逻辑，在扩展时是结构性隐患。

---

## 七、Brainstorm Q&A

### Q1: 纯 bash 接口在非 SWE 任务上的泛化性边界在哪？

**假设**: bash 是通用接口，所以适用于所有任务。
**质疑**: SWE 任务的本质是"编辑文本文件并验证"，这恰好是 bash 的强项。但 GUI 操作、数据库交互、实时系统控制等任务中，bash 不是自然接口。纯 bash harness 的性能拐点出现在哪里？是任务类型（编辑 vs 交互）还是工具需求复杂度（1 种 vs 10 种）决定的？

### Q2: 如果未来 LM 普遍支持 tool calling，text-based 解析还有存在价值吗？

**假设**: text-based 模式是给弱模型用的兼容方案。
**反视角**: text-based 模式的一个被忽视的优势是 **prompt 完全可控**——你可以精确指定输出格式并用正则解析，不受 tool calling 框架的 schema 约束。在需要非常规格式输出（如结构化推理链、自定义代码块）时，正则可能比 tool schema 更灵活。这是否意味着 text-based 不应被视为"降级方案"而是"替代范式"？

### Q3: 无状态执行在需要长时环境依赖的任务上是否根本性失效？

**场景**: 任务要求启动一个 web server，然后在另一步中向它发请求。无状态执行意味着每步的 server 是新的。mini 的 cwd 前缀技巧无法解决这个问题。
**追问**: 这是无状态执行的根本性局限，还是可以通过"在后台启动长期进程"的 bash 技巧绕过？如果可以绕过，那"无状态"的承诺就被打破了——harness 声称每步独立，但 LM 学会了创建跨步状态。这种隐式状态的审计难度可能比显式有状态 shell 更高。

### Q4: 把流程知识放在 prompt 中而非代码中，是在降低耦合还是在制造更脆弱的耦合？

**观察**: SWE-bench 的提交流程（git diff → verify → echo COMPLETE_...）硬编码在 prompt 中。修改提交流程需要改 YAML 而非改 Python。
**正面**: 这使得非程序员可以调整流程。
**反面**: prompt 中的流程指令没有类型检查、没有单元测试、没有 lint。一个 YAML 缩进错误可能导致 LM 完全不理解提交指令。代码中的流程至少有编译器/测试兜底。
**核心问题**: prompt 作为"弱类型代码"的可靠性边界在哪里？什么复杂度的逻辑应该留在代码中而非 prompt 中？

### Q5: 异常驱动控制流是否与 RL 训练不兼容？

RL 训练需要将 agent 的执行轨迹建模为 MDP（马尔可夫决策过程）。异常驱动的控制流意味着某些状态转移是通过异常而非正常返回值实现的——`Submitted` 异常不是 LM 的"决策结果"，而是 Environment 检测到的"事件"。

在 SFT 中这不是问题（轨迹直接用于训练），但在 RL 中：
- Reward 应该归因给哪个动作？触发 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 的那个？还是 Environment 检测到的那个？
- 异常中断的步骤是否应该有负 reward？
- `FormatError` 导致的步骤是 LM 的"错误"还是 harness 的"惩罚"？

如果 harness 的首要用户从"推理时使用者"变为"训练时环境"，异常驱动控制流是否需要重新设计？

---

*本文档基于 mini-swe-agent v2.3.0 全量源码精读，采用 paper-note skill 的思考框架，聚焦 agent harness 设计维度的 insight 提取与张力分析。*
