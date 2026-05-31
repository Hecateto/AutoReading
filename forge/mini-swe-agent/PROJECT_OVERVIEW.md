# mini-swe-agent 项目深度解析文档

> 版本: v2.3.0 | 作者: Princeton & Stanford (Kilian Lieret, Carlos E. Jimenez 等) | 许可: MIT

---

## 一、项目概览

### 1.1 是什么

**mini-swe-agent** 是一个极简的 AI 软件工程代理（AI SWE Agent），由 SWE-bench 和 SWE-agent 的原班团队打造。核心理念是：**在保持接近原版 SWE-agent 性能的前提下，将代理架构简化 100 倍。**

它在 SWE-bench Verified 基准上达到 >74% 的正确率，被 Meta、NVIDIA、IBM、Princeton、Stanford 等广泛采用。

### 1.2 核心设计哲学

| 设计决策 | 含义 |
|---------|------|
| **唯一工具 = Bash** | 不需要自定义工具接口，LM 直接利用 shell 的全部能力 |
| **线性历史** | 每步只 append 消息，轨迹 = 发送给 LM 的消息，便于调试和微调 |
| **subprocess.run 执行** | 每个动作独立执行（非有状态 shell 会话），轻松切换到沙箱环境 |
| **异常驱动的控制流** | 用异常而非状态标志管理流程终止/中断，便于扩展和子类化 |

### 1.3 与 SWE-agent 的关系

- **mini-swe-agent**: 默认选择，极简 CLI 工具、线性控制流、更快更稳定的沙箱
- **swe-agent**: 适合需要实验不同工具集、不同历史处理器的场景
- 两者都达到优秀的 SWE-bench 性能，都有轨迹浏览器

---

## 二、项目架构

### 2.1 目录结构

```
mini-swe-agent/
├── src/minisweagent/              # 核心源码
│   ├── __init__.py                # 版本、全局配置、Protocol 定义
│   ├── __main__.py                # python -m 入口
│   ├── exceptions.py              # 异常体系（控制流核心）
│   ├── agents/                    # 代理实现
│   │   ├── default.py             # DefaultAgent（~170行，核心）
│   │   └── interactive.py         # InteractiveAgent（交互式扩展）
│   ├── environments/              # 执行环境
│   │   ├── local.py               # 本地环境（subprocess.run）
│   │   ├── docker.py              # Docker 环境
│   │   ├── singularity.py         # Singularity/Apptainer 环境
│   │   └── extra/                 # 额外环境（bubblewrap, contree, swerex等）
│   ├── models/                    # 语言模型接口
│   │   ├── litellm_model.py       # LitellmModel（默认，toolcall 模式）
│   │   ├── litellm_textbased_model.py  # 文本解析模式（v1兼容）
│   │   ├── litellm_response_model.py   # OpenAI Responses API
│   │   ├── openrouter_*.py        # OpenRouter 系列
│   │   ├── portkey_*.py           # Portkey 系列
│   │   ├── requesty_model.py      # Requesty
│   │   ├── test_models.py         # 测试用确定性模型
│   │   ├── extra/roulette.py      # 轮盘/交替模型（元模型）
│   │   └── utils/                 # 模型工具函数
│   │       ├── actions_toolcall.py          # Toolcall 动作解析
│   │       ├── actions_text.py              # 文本正则动作解析
│   │       ├── actions_toolcall_response.py # Responses API 动作解析
│   │       ├── anthropic_utils.py           # Anthropic 兼容
│   │       ├── cache_control.py             # 缓存控制（Anthropic）
│   │       ├── content_string.py            # 内容字符串提取
│   │       ├── openai_multimodal.py         # 多模态内容处理
│   │       └── retry.py                     # 重试策略（tenacity）
│   ├── config/                    # 配置文件（YAML）
│   │   ├── default.yaml           # 默认配置（text-based）
│   │   ├── mini.yaml              # CLI 默认配置（toolcall）
│   │   ├── mini_textbased.yaml    # CLI text-based 配置
│   │   └── benchmarks/            # 基准测试配置
│   │       ├── swebench.yaml
│   │       ├── programbench.yaml
│   │       └── ...
│   ├── run/                       # 运行脚本（CLI 入口）
│   │   ├── mini.py                # 主 CLI `mini` 命令
│   │   ├── hello_world.py         # 最简示例
│   │   ├── benchmarks/            # 基准运行脚本
│   │   │   ├── swebench.py        # SWE-bench 批量推理
│   │   │   ├── swebench_single.py # SWE-bench 单实例
│   │   │   └── programbench.py    # ProgramBench
│   │   └── utilities/             # 工具命令
│   │       ├── config.py          # 全局配置管理
│   │       ├── inspector.py       # 轨迹浏览器（TUI）
│   │       └── mini_extra.py      # mini-extra 入口
│   └── utils/
│       ├── log.py                 # 日志（RichHandler）
│       └── serialize.py           # 递归合并 & UNSET 标记
├── tests/                         # 测试套件
├── docs/                          # 文档（MkDocs）
└── pyproject.toml                 # 项目元数据 & 依赖
```

### 2.2 核心架构图

```
┌─────────────────────────────────────────────────────────┐
│                      Run Script                         │
│              (mini.py / swebench.py)                    │
│    解析 CLI → 加载 Config → 创建 Model/Env/Agent → run  │
└───────────────┬──────────────────┬──────────────────────┘
                │                  │
    ┌───────────▼────────┐  ┌──────▼───────────┐
    │      Agent         │  │    Config         │
    │  ┌──────────────┐  │  │  YAML + CLI      │
    │  │DefaultAgent  │  │  │  recursive_merge │
    │  │InteractiveAgt│  │  └──────────────────┘
    │  └──┬───────┬───┘  │
    │     │       │      │
    │  query()  execute  │
    │     │       │      │
    └─────┼───────┼──────┘
          │       │
   ┌──────▼──┐ ┌──▼──────────┐
   │  Model  │ │ Environment  │
   │         │ │              │
   │ Litellm │ │   Local      │
   │ TextBased│ │  Docker     │
   │ Response │ │ Singularity │
   │ OpenRouter│ │  ...       │
   │ Portkey  │ │              │
   │ Roulette │ │              │
   └─────────┘ └──────────────┘
```

---

## 三、核心概念深度解析

### 3.1 Protocol 协议体系（`__init__.py`）

项目使用 Python Protocol（结构化子类型/鸭子类型）定义三大核心接口：

```python
class Model(Protocol):
    config: Any
    def query(self, messages, **kwargs) -> dict: ...
    def format_message(self, **kwargs) -> dict: ...
    def format_observation_messages(self, message, outputs, template_vars=None) -> list[dict]: ...
    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...

class Environment(Protocol):
    config: Any
    def execute(self, action, cwd="") -> dict[str, Any]: ...
    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...

class Agent(Protocol):
    config: Any
    def run(self, task, **kwargs) -> dict: ...
    def save(self, path, *extra_dicts) -> dict: ...
```

**关键设计**: 使用 Protocol 而非抽象基类，意味着任何具有匹配方法签名的类都自动满足接口，无需显式继承——这是"鸭子类型 + 静态类型检查"的最佳实践。

### 3.2 异常驱动的控制流（`exceptions.py`）

这是整个架构最精妙的部分。所有流程控制通过异常体系实现：

```
InterruptAgentFlow          # 基类：中断代理流程并携带消息
├── Submitted               # 任务完成，agent 已提交结果
├── LimitsExceeded          # 超出步数/费用限制
│   └── TimeExceeded        # 超出时间限制
├── UserInterruption        # 用户中断
└── FormatError             # LM 输出格式不正确
```

**工作原理**: 每个异常都携带 dict 消息。`DefaultAgent.run()` 的主循环捕获 `InterruptAgentFlow`，将异常中的消息添加到历史中，然后检查最后一条消息的 `role` 是否为 `"exit"` 来决定是否终止。

```python
while True:
    try:
        self.step()
    except InterruptAgentFlow as e:
        self.add_messages(*e.messages)
    except Exception as e:
        self.handle_uncaught_exception(e)
        raise
    finally:
        self.save(self.config.output_path)
    if self.messages[-1].get("role") == "exit":
        break
```

**为何用异常而非状态标志**：
- 可以在调用栈的任意深度抛出（如 Environment.execute 内部检测到任务完成）
- 子类化扩展时，只需 raise 异常，无需修改主循环逻辑
- 异常天然携带上下文（消息），不需要额外传参

### 3.3 消息格式

所有消息都是 `dict`，统一格式：

```python
{
    "role": "system" | "user" | "assistant" | "tool" | "exit",
    "content": "..." | [...],           # 文本或多模态内容列表
    "extra": {                           # 元数据（不发送给LM）
        "actions": [{"command": "..."}], # assistant消息中的动作
        "cost": 0.0,                     # 本次调用费用
        "timestamp": 1234567890.0,       # 时间戳
        "exit_status": "...",            # exit消息的状态
        "submission": "...",             # 提交内容
        ...
    }
}
```

**关键**: `extra` 字段在发送给 LM 前会被 `_prepare_messages_for_api` 过滤掉，仅用于内部记录。

---

## 四、Agent 详解

### 4.1 DefaultAgent（`agents/default.py`）

**仅约 170 行**，是整个项目最核心的类。

#### 配置（AgentConfig）

```python
class AgentConfig(BaseModel):
    system_template: str       # 系统消息模板
    instance_template: str     # 用户任务消息模板
    step_limit: int = 0        # 最大步数（0=无限制）
    cost_limit: float = 3.0    # 费用限制
    wall_time_limit_seconds: int = 0  # 时间限制（0=无限制）
    output_path: Path | None = None   # 轨迹保存路径
```

#### 核心方法链

```
run(task)
 └── 初始化 messages（system + instance 模板渲染）
 └── while True:
      └── step()
           └── query()           # 查询模型
           │    ├── 检查限制
           │    ├── model.query(messages) → 返回消息
           │    └── add_messages(message)
           └── execute_actions(message)
                ├── env.execute(action) × N
                ├── model.format_observation_messages(...)
                └── add_messages(*observation_msgs)
```

#### 模板系统

使用 **Jinja2** 模板引擎渲染 system_template 和 instance_template，模板变量来源（优先级从低到高递归合并）：

1. `config.model_dump()` — Agent 配置
2. `env.get_template_vars()` — 环境变量（含系统信息、OS环境变量）
3. `model.get_template_vars()` — 模型配置
4. 运行时变量（`n_model_calls`, `model_cost`, `elapsed_seconds`）
5. `extra_template_vars` — 额外变量
6. kwargs — 方法调用时传入

### 4.2 InteractiveAgent（`agents/interactive.py`）

在 DefaultAgent 基础上增加了**人机交互**能力，三种模式：

| 模式 | 行为 |
|------|------|
| **human** (`/u`) | 用户直接输入命令执行 |
| **confirm** (`/c`) | LM 的命令需用户确认（默认） |
| **yolo** (`/y`) | LM 的命令直接执行，无需确认 |

扩展点：
- `add_messages()`: 增加终端彩色输出
- `query()`: human 模式下读取用户命令
- `step()`: 捕获 KeyboardInterrupt
- `execute_actions()`: 增加确认流程
- 支持白名单正则（`whitelist_actions`）跳过确认
- `confirm_exit` 控制任务完成时是否确认

---

## 五、Environment 详解

### 5.1 核心接口

所有 Environment 都实现 `execute(action, cwd) -> dict` 方法，返回：

```python
{
    "output": "命令输出",           # stdout
    "returncode": 0,               # 退出码
    "exception_info": "",          # 异常信息
    "extra": {...}                 # 可选额外信息
}
```

### 5.2 LocalEnvironment（`environments/local.py`）

**最简实现**，直接用 `subprocess.run` 在本地执行命令：

```python
result = subprocess.run(
    command, shell=True, text=True,
    cwd=cwd, env=os.environ | self.config.env,
    timeout=timeout or self.config.timeout,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
```

关键特性：
- 每次命令在**独立子 shell** 中执行
- 环境变量不持久（但可通过前缀 `MY_ENV_VAR=MY_VALUE cd /path && ...` 解决）
- `_check_finished()` 检测输出首行是否为 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- 默认超时 30 秒

### 5.3 DockerEnvironment（`environments/docker.py`）

在 Docker 容器中执行命令：
- `docker run -d` 启动容器（带 `sleep` 保持运行）
- `docker exec` 执行每条命令
- 支持 `forward_env` 转发宿主机环境变量
- 自动清理（`__del__` 中 `docker stop/rm`）
- 可自定义可执行文件（docker/podman）

### 5.4 SingularityEnvironment（`environments/singularity.py`）

- `singularity build --sandbox` 构建沙箱
- `singularity exec --writable` 在沙箱中执行
- 构建失败自动重试（`sandbox_build_retries`）
- 支持 `--fakeroot`, `--cleanenv` 等参数

### 5.5 额外环境

通过 `pip install mini-swe-agent[full]` 可用：
- **swerex_docker**: 基于 SWE-ReX 的 Docker 环境
- **swerex_modal**: 基于 Modal 云平台的远程执行
- **bubblewrap**: Linux bubblewrap 沙箱
- **contree**: Contree 容器环境

### 5.6 环境工厂（`environments/__init__.py`）

```python
_ENVIRONMENT_MAPPING = {
    "docker": "minisweagent.environments.docker.DockerEnvironment",
    "local": "minisweagent.environments.local.LocalEnvironment",
    ...
}

def get_environment(config, *, default_type=""): ...
```

支持通过简写名（如 `"docker"`）或完整导入路径指定环境类。

---

## 六、Model 详解

### 6.1 三种动作解析模式

| 模式 | 类 | 原理 | 适用场景 |
|------|---|------|---------|
| **Toolcall** | `LitellmModel` | 使用 LM 的 function calling / tool use 接口 | 支持 toolcall 的模型（推荐） |
| **Text-based** | `LitellmTextbasedModel` | 用正则从文本中提取 ` ```mswea_bash_command ``` ` 代码块 | 不支持 toolcall 的模型（v1兼容） |
| **Response API** | `LitellmResponseModel` | 使用 OpenAI Responses API 格式 | OpenAI Responses API |

### 6.2 LitellmModel（默认模型类）

```python
class LitellmModel:
    def query(self, messages, **kwargs):
        # 1. 准备消息（过滤extra、重排thinking blocks、设置缓存控制）
        prepared = self._prepare_messages_for_api(messages)
        # 2. 带重试的API调用
        response = self._query(prepared, **kwargs)
        # 3. 计算费用
        cost = self._calculate_cost(response)
        # 4. 解析 tool calls → actions
        actions = self._parse_actions(response)
        # 5. 组装消息
        return {"role": "assistant", "content": ..., "extra": {actions, cost, ...}}
```

关键特性：
- 使用 **litellm** 库统一所有 LLM API
- 定义 `BASH_TOOL`（function calling 的 bash 工具定义）
- 支持 `set_cache_control`（Anthropic 缓存优化）
- 支持 `multimodal_regex`（从输出中提取图片等）
- 重试策略：指数退避，最多 10 次，特定异常（认证错误等）直接终止

### 6.3 模型工厂（`models/__init__.py`）

```python
_MODEL_CLASS_MAPPING = {
    "litellm": "minisweagent.models.litellm_model.LitellmModel",
    "litellm_textbased": "...",
    "openrouter": "...",
    "portkey": "...",
    "deterministic": "...",
}
```

模型名解析优先级：`input_model_name` → `config.model_name` → `MSWEA_MODEL_NAME` 环境变量

Anthropic 模型自动启用 `set_cache_control: "default_end"`。

### 6.4 元模型：RouletteModel（`models/extra/roulette.py`）

有趣的实验性模型：
- **RouletteModel**: 每次调用随机选择一个模型
- **InterleavingModel**: 按固定序列交替选择模型

### 6.5 全局模型统计（GlobalModelStats）

线程安全的全局费用/调用次数追踪器，支持：
- 全局费用上限 (`MSWEA_GLOBAL_COST_LIMIT`)
- 全局调用次数上限 (`MSWEA_GLOBAL_CALL_LIMIT`)

---

## 七、Config 系统详解

### 7.1 配置层级

```
YAML 配置文件
    ↓ 递归合并
CLI 参数（-c 指定的多个配置 + 命令行选项）
    ↓
最终运行时配置
```

### 7.2 配置文件查找

`get_config_path()` 按以下顺序查找：
1. 原始路径
2. `$MSWEA_CONFIG_DIR/` 目录
3. 内置配置目录 (`src/minisweagent/config/`)
4. 内置 extra 目录
5. 内置 benchmarks 目录

### 7.3 递归合并（`utils/serialize.py`）

`recursive_merge(*dicts)` 是配置系统的核心：
- 后面的 dict 优先级更高
- 嵌套 dict 递归合并
- `UNSET` 哨兵值表示"删除此键"

### 7.4 Key-Value 行内配置

CLI 支持直接传入嵌套配置：

```bash
mini -c mini.yaml -c model.model_kwargs.temperature=0.5
```

`"model.model_kwargs.temperature=0.5"` 会被解析为：
```python
{"model": {"model_kwargs": {"temperature": 0.5}}}
```

### 7.5 配置文件结构

每个 YAML 文件包含四大块：

```yaml
agent:            # Agent 配置
  system_template: ...
  instance_template: ...
  step_limit: 0
  cost_limit: 3.0

environment:      # Environment 配置
  cwd: ""
  timeout: 30
  env: {...}

model:            # Model 配置
  observation_template: ...
  format_error_template: ...
  model_kwargs: {...}

run:              # Run 脚本特有配置
  task: ...
```

### 7.6 重要配置对比

| 配置 | default.yaml | mini.yaml | swebench.yaml |
|------|-------------|-----------|---------------|
| 动作模式 | text-based (regex) | toolcall | toolcall |
| step_limit | 0 | 0 | 250 |
| cost_limit | 0 | 3.0 | 3.0 |
| timeout | 30 | 30 | 60 |
| 环境 | local | local | docker |
| cwd | - | - | /testbed |
| temperature | - | - | 0.0 |
| parallel_tool_calls | - | - | true |

### 7.7 全局配置文件

位置: `~/.config/mini-swe-agent/.env`（或 `$MSWEA_GLOBAL_CONFIG_DIR/.env`）

通过 `mini-extra config` 管理的关键变量：
- `MSWEA_MODEL_NAME`: 默认模型
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` 等: API 密钥
- `MSWEA_CONFIGURED`: 是否已完成首次配置

---

## 八、Run 脚本详解

### 8.1 `mini` 命令（`run/mini.py`）

主 CLI 入口，核心流程：

```
1. 首次运行检测 → configure_if_first_time()
2. 构建配置 → 递归合并多个 config spec
3. 如果未指定 task → 交互式提示
4. 创建 Model → get_model(config)
5. 创建 Environment → get_environment(config, default_type="local")
6. 创建 Agent → get_agent(model, env, config, default_type="interactive")
7. 运行 → agent.run(task)
```

关键 CLI 选项：
- `-m/--model`: 模型名
- `-t/--task`: 任务描述
- `-y/--yolo`: 无需确认直接执行
- `-c/--config`: 配置文件/键值对（可多个）
- `-o/--output`: 输出轨迹文件
- `--exit-immediately`: 任务完成时不确认直接退出

### 8.2 `mini-swebench` 命令（`run/benchmarks/swebench.py`）

SWE-bench 批量推理，核心特性：
- 支持多个数据集子集（lite/verified/full/multimodal 等）
- 多线程并行处理 (`-w/--workers`)
- 实时进度显示（Rich Live）
- 断点续传（跳过已完成的实例）
- 实例过滤/切片/随机化
- 自动管理 Docker 容器生命周期

### 8.3 Inspector（`run/utilities/inspector.py`）

基于 **Textual** 框架的 TUI 轨迹浏览器：
- 按步骤浏览代理对话轨迹
- 支持多轨迹文件切换
- 支持 jless 打开原始 JSON
- Vim 风格快捷键（h/l/j/k）

### 8.4 `mini-extra` 命令（`run/utilities/mini_extra.py`）

所有额外功能的统一入口：
- `mini-extra config`: 管理全局配置
- `mini-extra inspect` / `i`: 轨迹浏览器
- `mini-extra swebench`: SWE-bench 批量推理
- `mini-extra swebench-single`: SWE-bench 单实例
- `mini-extra programbench`: ProgramBench

---

## 九、任务完成机制

### 9.1 提交流程

代理通过执行特定格式的命令来提交任务：

```
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
<可选的提交内容>
```

当 Environment 的 `_check_finished()` 检测到：
1. 输出首行是 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
2. 返回码为 0

就会抛出 `Submitted` 异常，携带 `role="exit"` 的消息，终止主循环。

### 9.2 SWE-bench 特殊提交流程

在 swebench 配置中，提交更严格：
1. 先 `git diff -- path/to/files > patch.txt` 生成补丁
2. 验证补丁内容
3. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` 提交

---

## 十、输出与轨迹

### 10.1 轨迹文件格式（`.traj.json`）

```json
{
  "info": {
    "model_stats": {"instance_cost": 0.42, "api_calls": 12},
    "config": {
      "agent": {...},
      "model": {...},
      "environment": {...}
    },
    "mini_version": "2.3.0",
    "exit_status": "Submitted",
    "submission": "patch content..."
  },
  "messages": [...],
  "trajectory_format": "mini-swe-agent-1.1"
}
```

### 10.2 SWE-bench 预测文件（`preds.json`）

```json
{
  "instance_id": {
    "model_name_or_path": "...",
    "instance_id": "...",
    "model_patch": "..."
  }
}
```

---

## 十一、Python Bindings（编程接口）

最简单的使用方式：

```python
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

agent = DefaultAgent(
    LitellmModel(model_name="anthropic/claude-sonnet-4-5-20250929"),
    LocalEnvironment(),
)
agent.run("Write a sudoku game")
```

使用配置文件：

```python
import yaml
from pathlib import Path
from minisweagent import package_dir

config = yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"]
agent = DefaultAgent(model, env, **config)
agent.run(task)
```

---

## 十二、依赖体系

### 核心依赖

| 包 | 用途 |
|---|------|
| `litellm` | 统一 LLM API 接口 |
| `pydantic` | 配置模型验证（v2+） |
| `jinja2` | 模板渲染 |
| `pyyaml` | YAML 配置文件 |
| `tenacity` | API 调用重试 |
| `typer` | CLI 框架 |
| `rich` | 终端美化输出 |
| `textual` | TUI（Inspector） |
| `python-dotenv` | 环境变量管理 |
| `platformdirs` | 跨平台配置目录 |
| `datasets` | HuggingFace 数据集加载 |
| `requests` | HTTP 请求 |

### 可选依赖

| 包 | 用途 |
|---|------|
| `swe-rex` | SWE-ReX 执行环境 |
| `modal` | Modal 云平台 |
| `contree-sdk` | Contree 容器 |
| `portkey-ai` | Portkey API |

---

## 十三、测试体系

测试位于 `tests/` 目录，按模块组织：

```
tests/
├── agents/          # Agent 测试
├── environments/    # Environment 测试
├── models/          # Model 测试
├── run/             # Run 脚本集成测试
├── config/          # Config 解析测试
└── utils/           # 工具函数测试
```

使用 `pytest` + `pytest-asyncio` + `pytest-xdist`（并行），标记 `@pytest.mark.slow` 区分慢测试。

---

## 十四、环境变量速查

| 变量 | 用途 |
|------|------|
| `MSWEA_MODEL_NAME` | 默认模型名 |
| `MSWEA_GLOBAL_CONFIG_DIR` | 全局配置目录 |
| `MSWEA_SILENT_STARTUP` | 静默启动 |
| `MSWEA_COST_TRACKING` | 费用追踪模式 |
| `MSWEA_GLOBAL_COST_LIMIT` | 全局费用上限 |
| `MSWEA_GLOBAL_CALL_LIMIT` | 全局调用次数上限 |
| `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT` | 重试次数 |
| `MSWEA_DOCKER_EXECUTABLE` | Docker 可执行文件路径 |
| `MSWEA_SINGULARITY_EXECUTABLE` | Singularity 可执行文件路径 |
| `MSWEA_CONFIG_DIR` | 配置文件搜索目录 |
| `MSWEA_MINI_CONFIG_PATH` | mini 命令的配置文件路径 |
| `MSWEA_CONFIGURED` | 是否已完成配置 |
| `LITELLM_MODEL_REGISTRY_PATH` | 自定义模型注册表 |

---

## 十五、设计模式总结

### 15.1 使用的设计模式

1. **Protocol 模式（结构化子类型）**: Model/Environment/Agent 接口定义，无需继承
2. **工厂模式**: `get_model()`, `get_environment()`, `get_agent()` 根据配置创建实例
3. **策略模式**: 不同的动作解析策略（toolcall/text-based/response API）
4. **模板方法模式**: DefaultAgent 定义算法骨架，InteractiveAgent 通过重写方法扩展
5. **异常驱动控制流**: 用异常体系管理流程中断
6. **递归合并模式**: 配置系统的层层覆盖
7. **哨兵对象**: `UNSET` 用于标记"删除此配置项"

### 15.2 扩展指南

要添加新的：
- **Model**: 创建类实现 `query/format_message/format_observation_messages/get_template_vars/serialize`，注册到 `_MODEL_CLASS_MAPPING`
- **Environment**: 创建类实现 `execute/get_template_vars/serialize`（含 `_check_finished`），注册到 `_ENVIRONMENT_MAPPING`
- **Agent**: 继承 `DefaultAgent`，重写 `query/step/execute_actions` 等方法

---

## 十六、关键文件索引

| 文件 | 行数 | 核心性 | 说明 |
|------|------|--------|------|
| `agents/default.py` | ~170 | ★★★ | 最核心的 Agent 类，整个项目的精神所在 |
| `environments/local.py` | ~79 | ★★★ | 最简执行环境，展示核心接口 |
| `models/litellm_model.py` | ~147 | ★★★ | 默认模型类，toolcall 模式 |
| `exceptions.py` | ~26 | ★★★ | 异常体系，控制流核心 |
| `__init__.py` | ~92 | ★★☆ | Protocol 定义，全局配置 |
| `run/mini.py` | ~109 | ★★☆ | 主 CLI 入口 |
| `config/default.yaml` | ~167 | ★★☆ | 完整默认配置（含提示词） |
| `models/utils/actions_toolcall.py` | ~103 | ★★☆ | Toolcall 动作解析 |
| `agents/interactive.py` | ~183 | ★★☆ | 交互式 Agent 扩展 |
| `run/benchmarks/swebench.py` | ~275 | ★☆☆ | 批量推理脚本 |
| `run/utilities/inspector.py` | ~290 | ★☆☆ | 轨迹浏览器 TUI |
| `utils/serialize.py` | ~29 | ★★☆ | 递归合并 + UNSET |

---

*本文档基于 mini-swe-agent v2.3.0 源码精读生成，涵盖项目全貌、架构设计、核心源码解析。*
