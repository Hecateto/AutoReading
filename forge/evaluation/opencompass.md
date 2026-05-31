---                                                                                                                                                                      
                                                                                                                                                                            
   # OpenCompass 架构系统分析报告                                                                                                                                           
                                                                                                                                                                            
   ## 一、项目概览                                                                                                                                                          
                                                                                                                                                                            
   OpenCompass 是上海人工智能实验室（Shanghai AI Lab）开源的 LLM 评测框架，定位为"LLM 评测领域的瑞士军刀"。当前代码规模：**3017 个 Python 文件，约 14.5 万行代码**。        
                                                                                                                                                                            
   核心能力：用统一框架对 LLM 在多种 benchmark 上做 **推理(infer) → 评分(eval) → 汇总(viz)** 三阶段流水线评测。                                                             
                                                                                                                                                                            
   ---                                                                                                                                                                      
                                                                                                                                                                            
   ## 二、架构总览：四层设计                                                                                                                                                
                                                                                                                                                                            
   ```                                                                                                                                                                      
   ┌─────────────────────────────────────────────────┐                                                                                                                      
   │  CLI / Config Layer (入口层)                     │                                                                                                                     
   │  run.py → cli/main.py → mmengine.Config         │                                                                                                                      
   ├─────────────────────────────────────────────────┤                                                                                                                      
   │  Scheduling Layer (调度层)                       │                                                                                                                     
   │  Partitioner → Runner (Local/Slurm/DLC)         │                                                                                                                      
   ├─────────────────────────────────────────────────┤                                                                                                                      
   │  Task Layer (任务层)                             │                                                                                                                     
   │  OpenICLInferTask / OpenICLEvalTask / ...       │                                                                                                                      
   ├─────────────────────────────────────────────────┤                                                                                                                      
   │  Engine Layer (引擎层)                           │                                                                                                                     
   │  openicl: Retriever → Inferencer → Evaluator    │                                                                                                                      
   │  models: BaseModel / BaseAPIModel / HuggingFace │                                                                                                                      
   │  datasets: 200+ benchmark datasets              │                                                                                                                      
   └─────────────────────────────────────────────────┘                                                                                                                      
   ```                                                                                                                                                                      
                                                                                                                                                                            
   ---                                                                                                                                                                      
                                                                                                                                                                            
   ## 三、核心数据流：一次完整评测的全链路                                                                                                                                  
                                                                                                                                                                            
   ```                                                                                                                                                                      
   用户 Config (models + datasets)                                                                                                                                          
     │                                                                                                                                                                      
     ▼                                                                                                                                                                      
   main() → get_config_from_arg()                                                                                                                                           
     │                                                                                                                                                                      
     ├─[infer mode]───────────────────────────────────────┐                                                                                                                 
     │  Partitioner(Naive/Size/NumWorker)                  │                                                                                                                
     │    把 M个模型 × N个数据集 拆成独立 Task 列表         │                                                                                                               
     │  Runner(Local/Slurm/DLC)                            │                                                                                                                
     │    调度 Task 执行，GPU 分配，并发控制                 │                                                                                                              
     │  OpenICLInferTask.run()                             │                                                                                                                
     │    ├─ build_model_from_cfg() → BaseModel 实例       │                                                                                                                
     │    ├─ build_dataset_from_cfg() → HF Dataset         │                                                                                                                
     │    ├─ Retriever.retrieve() → in-context examples    │                                                                                                                
     │    ├─ PromptTemplate → 拼装 prompt                  │                                                                                                                
     │    ├─ Inferencer (Gen/PPL/Chat/SC/ToT/Agent)        │                                                                                                                
     │    │    ├─ GenInferencer: model.generate() → 预测文本 │                                                                                                              
     │    │    └─ PPLInferencer: model.get_ppl() → 概率排序  │                                                                                                              
     │    └─ 输出 → work_dir/predictions/{model}_{dataset}.json│                                                                                                            
     │                                                      │                                                                                                               
     ├─[eval mode]─────────────────────────────────────────┤                                                                                                                
     │  OpenICLEvalTask.run()                              │                                                                                                                
     │    ├─ 读 predictions JSON                           │                                                                                                                
     │    ├─ TextPostprocessor → 提取答案(A/B/C/D等)        │                                                                                                               
     │    ├─ Evaluator.score()                             │                                                                                                                
     │    │    ├─ AccEvaluator: 精确匹配                    │                                                                                                               
     │    │    ├─ GenericLLMEvaluator: LLM-as-Judge        │                                                                                                                
     │    │    ├─ CascadeEvaluator: 规则先行+LLM兜底        │                                                                                                               
     │    │    └─ 20+ 特化 Evaluator (Code/Rouge/Toxic...)  │                                                                                                               
     │    └─ 输出 → work_dir/results/{model}_{dataset}.json │                                                                                                               
     │                                                      │                                                                                                               
     └─[viz mode]──────────────────────────────────────────┘                                                                                                                
        DefaultSummarizer                                                                                                                                                   
          ├─ 读取所有 results JSON                                                                                                                                          
          ├─ METRIC_WHITELIST/BLACKLIST 过滤                                                                                                                                
          ├─ summary_groups 聚合 (如 MMLU = 57子集平均)                                                                                                                     
          └─ 输出 → Markdown 表格                                                                                                                                           
   ```                                                                                                                                                                      
                                                                                                                                                                            
   ---                                                                                                                                                                      
                                                                                                                                                                            
   ## 四、关键子系统深度分析                                                                                                                                                
                                                                                                                                                                            
   ### 4.1 Registry 注册机制（registry.py）                                                                                                                                 
                                                                                                                                                                            
   **设计思想**：基于 mmengine.Registry 的插件式架构，所有组件通过 `@XXX.register_module()` 注册，通过 `XXX.build(cfg)` 实例化。                                            
                                                                                                                                                                            
   核心 Registry（11 个）：                                                                                                                                                 
                                                                                                                                                                            
   | Registry | 职责 | 典型实现 |                                                                                                                                           
   |----------|------|---------|                                                                                                                                            
   | MODELS | 模型适配器 | HuggingFaceModel, OpenAI, TurboMind, vLLM |                                                                                                      
   | LOAD_DATASET | 数据集加载器 | MMLUDataset, GSM8KDataset, ... |                                                                                                         
   | ICL_INFERENCERS | 推理策略 | GenInferencer, PPLInferencer, ChatInferencer |                                                                                            
   | ICL_EVALUATORS | 评分策略 | AccEvaluator, GenericLLMEvaluator, CascadeEvaluator |                                                                                      
   | ICL_RETRIEVERS | 示例检索 | ZeroRetriever, FixKRetriever, BM25Retriever |                                                                                              
   | ICL_PROMPT_TEMPLATES | Prompt 模板 | PromptTemplate |                                                                                                                  
   | PARTITIONERS | 任务拆分 | NaivePartitioner, SizePartitioner |                                                                                                          
   | RUNNERS | 执行调度 | LocalRunner, SlurmRunner, DLCRunner |                                                                                                             
   | TASKS | 任务类型 | OpenICLInferTask, OpenICLEvalTask |                                                                                                                 
   | TEXT_POSTPROCESSORS | 答案后处理 | match_answer_pattern, first_capital_postprocess |                                                                                   
   | DICT_POSTPROCESSORS | 评分布后处理 | generic_llmjudge_postprocess |                                                                                                    
                                                                                                                                                                            
   **Insight**: 继承了 mmengine 的 `force=True`（允许同名覆盖注册），这是有意为之——让用户自定义配置可以覆盖默认实现。                                                       
                                                                                                                                                                            
   ### 4.2 模型层（models/）                                                                                                                                                
                                                                                                                                                                            
   **双层基类设计**：                                                                                                                                                       
                                                                                                                                                                            
   - **BaseModel** (515行)：本地模型基类，抽象方法 `generate()` + `get_ppl()`                                                                                               
     - 持有 `LMTemplateParser`，负责 PromptList → 纯文本的渲染                                                                                                              
     - 支持 `sync_rank` 多卡同步                                                                                                                                            
                                                                                                                                                                            
   - **BaseAPIModel** (533行)：API 模型基类，继承 BaseModel                                                                                                                 
     - 核心差异：`TokenBucket` 限速器（RPM 控制），`acquire()/release()` 信号量                                                                                             
     - 持有 `APITemplateParser`，将 PromptList → OpenAI 格式 messages                                                                                                       
     - 内置 retry 机制                                                                                                                                                      
                                                                                                                                                                            
   **模型适配器矩阵**（50+ 个）：                                                                                                                                           
                                                                                                                                                                            
   | 类别 | 数量 | 代表 |                                                                                                                                                   
   |------|------|------|                                                                                                                                                   
   | 本地推理 | ~6 | HuggingFace, vLLM, TurboMind, LMDeploy, ModelScope |                                                                                                   
   | 国际 API | ~8 | OpenAI, Claude, Gemini, Mistral, DeepSeek |                                                                                                            
   | 国内 API | ~15 | GLM, Qwen, Moonshot, Baichuan, Doubao, Baidu... |                                                                                                     
   | 特殊 | ~4 | LAgent, LangChain, InternTrain, Accessory |                                                                                                                
                                                                                                                                                                            
   **Insight**: 模型层设计的关键决策是"prompt 模板下沉"——每个模型自带 `meta_template`（如 ChatML、Llama 格式），由 TemplateParser                                           
   在模型侧渲染，而非在上层统一格式化。这给了每个模型最大灵活性，但也意味着用户需要理解各模型的 chat template 差异。                                                        
                                                                                                                                                                            
   ### 4.3 OpenICL 引擎（openicl/）                                                                                                                                         
                                                                                                                                                                            
   这是 OpenCompass 的"心脏"，一个独立的 In-Context Learning 评测引擎，分四个阶段：                                                                                         
                                                                                                                                                                            
   **Stage 1 - DatasetReader**：加载 HF Dataset，定义 input/output columns                                                                                                  
                                                                                                                                                                            
   **Stage 2 - Retriever**（10 种）：决定给测试样本提供哪些 in-context examples                                                                                             
   - `ZeroRetriever`：最常用，零样本直接测试                                                                                                                                
   - `FixKRetriever` / `RandomRetriever`：固定 K 个示例                                                                                                                     
   - `BM25Retriever` / `TopkRetriever` / `DPPRetriever`：基于相似度/多样性选择                                                                                              
                                                                                                                                                                            
   **Stage 3 - Inferencer**（14 种）：决定如何获取模型输出                                                                                                                  
   - `GenInferencer`：最常用，`model.generate()` 直接生成                                                                                                                   
   - `PPLInferencer`：对每个选项计算 perplexity，选最小的                                                                                                                   
   - `ChatInferencer` / `ChatMLInferencer`：对话格式推理                                                                                                                    
   - `SCInferencer`：Self-Consistency 多次采样投票                                                                                                                          
   - `ToTInferencer`：Tree of Thought 搜索                                                                                                                                  
   - `AgentInferencer`：lagent 代理执行                                                                                                                                     
   - 还有并行版 `*Parallel`、CLP、LL 等变体                                                                                                                                 
                                                                                                                                                                            
   **Stage 4 - Evaluator**（15+ 种）：决定如何打分                                                                                                                          
   - `AccEvaluator`：精确匹配 / 多选准确率                                                                                                                                  
   - `GenericLLMEvaluator`：**LLM-as-Judge**（2025 年新增的重点功能）                                                                                                       
   - `CascadeEvaluator`：**规则先行 + LLM 兜底**（2025.04 新增）                                                                                                            
   - 特化：`CodeEvaluator`(HumanEval)、`RougeEvaluator`、`ToxicEvaluator`、`CircularEvaluator`(MMLU circular eval)等                                                        
                                                                                                                                                                            
   **Insight**: OpenICL 的核心设计哲学是**正交组合**——Retriever × Inferencer × Evaluator 三者独立可配。一个 MMLU 数据集可以用 Zero+Gen+Acc，也可以用                        
   FixK+PPL+Acc，配置驱动而非代码驱动。这就是为什么一个 benchmark 有多个 config 文件（如 `mmlu_gen.py`, `mmlu_ppl.py`, `mmlu_llm_judge_gen.py`）。                          
                                                                                                                                                                            
   ### 4.4 Partitioner + Runner（调度层）                                                                                                                                   
                                                                                                                                                                            
   **Partitioner**（7 种）：                                                                                                                                                
   - `NaivePartitioner`：最简单，1个模型×1个数据集 = 1个task                                                                                                                
   - `SizePartitioner` / `NumWorkerPartitioner`：按数据量/GPU数拆分                                                                                                         
   - `Sub*Partitioner`：进一步细分子任务                                                                                                                                    
                                                                                                                                                                            
   **Runner**（6 种）：                                                                                                                                                     
   - `LocalRunner`：单机本地执行，GPU 调度（`CUDA_VISIBLE_DEVICES`），ThreadPoolExecutor 并发                                                                               
   - `SlurmRunner`：srun 提交到 Slurm 集群                                                                                                                                  
   - `DLCRunner`：阿里云 DLC 平台                                                                                                                                           
   - `LocalAPIRunner`：专门为 API 模型优化                                                                                                                                  
   - `RJobRunner` / `VolcRunner`：火山引擎等                                                                                                                                
                                                                                                                                                                            
   **Insight**: Partitioner 把"模型×数据集"矩阵展平成独立 Task 列表，Runner 负责 Task 的实际执行。两者解耦意味着同一份 config 可以在本地单卡跑，也可以一键切到 Slurm        
   集群——`--slurm -p partition_name` 即可。                                                                                                                                 
                                                                                                                                                                            
   ### 4.5 Evaluator 子系统（evaluator/）                                                                                                                                   
                                                                                                                                                                            
   这是 2024-2025 年重点发展的方向：                                                                                                                                        
                                                                                                                                                                            
   - **GenericLLMEvaluator**：通用 LLM-as-Judge                                                                                                                             
     - 接收 judge_cfg（另一个 LLM 的配置），构建 GenInferencer 对 (prediction, reference) 打分                                                                              
     - 支持 pred_postprocessor + dict_postprocessor 双层后处理                                                                                                              
     - 输出 accuracy / F1 / attempted_ratio 等指标                                                                                                                          
                                                                                                                                                                            
   - **CascadeEvaluator**：级联评测（2025.04 新增）                                                                                                                         
     - 先用规则方法（如精确匹配）打分，标记为错误的样本再用 LLM Judge 复评                                                                                                  
     - 兼顾成本和准确性，是工程上的务实选择                                                                                                                                 
                                                                                                                                                                            
   - **MathEvaluator**：数学评测（MATHVerifyEvaluator），支持 verify 式评测                                                                                                 
                                                                                                                                                                            
   ### 4.6 Summarizer（汇总层）                                                                                                                                             
                                                                                                                                                                            
   - `DefaultSummarizer`：从 results 目录读取 JSON，按 METRIC_WHITELIST 过滤，生成 Markdown 表格                                                                            
   - `summary_groups`：支持聚合指标（如 MMLU = 57 子集的加权平均）                                                                                                          
   - `SubjectiveSummarizer`系列：主观评测汇总（MT-Bench、AlpacaEval、CompassArena 等），使用 Bradley-Terry 模型等特殊聚合方式                                               
                                                                                                                                                                            
   ---                                                                                                                                                                      
                                                                                                                                                                            
   ## 五、配置系统                                                                                                                                                          
                                                                                                                                                                            
   OpenCompass 使用 mmengine.Config 作为配置框架，核心配置结构：                                                                                                            
                                                                                                                                                                            
   ```python                                                                                                                                                                
   # 一个完整的评测 config                                                                                                                                                  
   models = [...]       # 模型列表，每个是 MODELS.build() 可解析的 dict                                                                                                     
   datasets = [...]     # 数据集列表，每个包含 infer_cfg + eval_cfg                                                                                                         
   summarizer = {...}   # 汇总器配置（可选）                                                                                                                                
                                                                                                                                                                            
   # 每个 dataset 的结构                                                                                                                                                    
   dict(                                                                                                                                                                    
       abbr='mmlu_stem',                                                                                                                                                    
       type=MMLUDataset,                                                                                                                                                    
       path='opencompass/mmlu',                                                                                                                                             
       reader_cfg=dict(input_columns=[...], output_column='target'),                                                                                                        
       infer_cfg=dict(                                                                                                                                                      
           prompt_template=dict(type=PromptTemplate, template=...),                                                                                                         
           retriever=dict(type=ZeroRetriever),                                                                                                                              
           inferencer=dict(type=GenInferencer),                                                                                                                             
       ),                                                                                                                                                                   
       eval_cfg=dict(                                                                                                                                                       
           evaluator=dict(type=AccEvaluator),                                                                                                                               
           pred_postprocessor=dict(type=match_answer_pattern, ...),                                                                                                         
       ),                                                                                                                                                                   
   )                                                                                                                                                                        
   ```                                                                                                                                                                      
                                                                                                                                                                            
   配置文件的组织（0.4.0 版本后迁入包内）：                                                                                                                                 
   - `opencompass/configs/datasets/` — 200+ benchmark 的预定义配置（按 benchmark 名组织，每个有 gen/ppl/llm_judge 等变体）                                                  
   - `opencompass/configs/models/` — 50+ 模型的预定义配置（按模型家族组织）                                                                                                 
   - `opencompass/configs/summarizers/` — 汇总器配置                                                                                                                        
                                                                                                                                                                            
   **Insight**: 配置文件名中的哈希后缀（如 `mmlu_gen_b618ea.py`）是 prompt template 的 SHA256 前 6 位，用于区分同一 benchmark 的不同 prompt 变体，避免缓存冲突。            
                                                                                                                                                                            
   ---                                                                                                                                                                      
                                                                                                                                                                            
   ## 六、设计哲学总结                                                                                                                                                      
                                                                                                                                                                            
   ### 6.1 核心设计决策                                                                                                                                                     
                                                                                                                                                                            
   1. **配置驱动 > 代码驱动**：评测的全流程（数据加载 → Prompt 模板 → 推理策略 → 评分方法）全部通过 Python Config dict                                                      
   描述，用户不需要写任何推理/评测代码，只需组合预定义组件。                                                                                                                
                                                                                                                                                                            
   2. **正交组合**：Retriever × Inferencer × Evaluator 三维正交，通过配置自由组合。这是 OpenICL 的核心价值——同一套数据和模型可以尝试多种评测方法。                          
                                                                                                                                                                            
   3. **调度与计算分离**：Partitioner 负责"拆什么"，Runner 负责"怎么跑"。用户可以 `--slurm` 一键切换执行环境。                                                              
                                                                                                                                                                            
   4. **插件式扩展**：11 个 Registry + `register_module()` + `build()` 的模式，新增一个模型/benchmark/评测器只需写一个类+一行注册。                                         
                                                                                                                                                                            
   ### 6.2 工程取舍                                                                                                                                                         
                                                                                                                                                                            
   | 决策 | 收益 | 代价 |                                                                                                                                                   
   |------|------|------|                                                                                                                                                   
   | mmengine 依赖 | 配置系统、Registry 免开发 | 引入重型依赖，调试困难 |                                                                                                   
   | 配置文件哈希命名 | 缓存不冲突 | 文件名不可读，需工具辅助 |                                                                                                             
   | Prompt 渲染下沉到模型 | 模型格式自由度最大 | 用户需理解 meta_template |                                                                                                
   | 同一 benchmark 多 config 变体 | 灵活适配不同评测范式 | 配置爆炸，200+ 个 mmlu 文件 |                                                                                   
                                                                                                                                                                            
   ### 6.3 适合什么场景                                                                                                                                                     
                                                                                                                                                                            
   - **学术评测**：快速在 200+ benchmark 上跑标准化评测                                                                                                                     
   - **模型对比**：M×N 矩阵式评测，自动汇总表格                                                                                                                             
   - **集群评测**：Slurm/DLC 一键提交                                                                                                                                       
   - **LLM-as-Judge**：CascadeEvaluator + GenericLLMEvaluator 支持主观评测                                                                                                  
                                                                                                                                                                            
   ### 6.4 不适合什么场景                                                                                                                                                   
                                                                                                                                                                            
   - **训练中评测**：框架偏重，不适合作为训练循环中的评测 hook                                                                                                              
   - **超轻量快速验证**：单卡跑 1 个小 benchmark，直接用 lm-eval-harness 更轻                                                                                               
   - **自定义复杂评测逻辑**：如果评测逻辑无法被 Acc/LLM-Judge/CodeEval 表达，需要继承 BaseEvaluator 写较多代码                                                              
                                                                                                                                                                            
   ---                                                                                                                                                                      
                                                                                                                                                                            
   ## 七、对你（gelato）的实用建议                                                                                                                                          
                                                                                                                                                                            
   基于你的环境（PAI 容器、自研架构预训练）和当前项目（SkillOpt 评测），OpenCompass 的可利用点：                                                                            
                                                                                                                                                                            
   1. **直接复用 openicl 的评测组件**：`AccEvaluator`、`GenInferencer`、`ZeroRetriever` 可以独立使用，不需要启动整个框架。在你的 SkillOpt eval 脚本中，可以 import          
   这些组件做标准化评测。                                                                                                                                                   
                                                                                                                                                                            
   2. **参考 CascadeEvaluator 的设计**：规则先行 + LLM 兜底的模式，和你 SkillOpt 中 S0 Baseline 规则 + 模型注入规则的思路异曲同工。                                         
                                                                                                                                                                            
   3. **benchmark 配置库**：`opencompass/configs/datasets/` 下 200+ benchmark 的 reader_cfg + eval_cfg 是极好的参考，即使不用框架也可以借鉴其 prompt 模板和后处理逻辑。     
                                                                                                                                                                            
   4. **不建议在 PAI 容器中完整安装 OpenCompass**：mmengine + 众多依赖偏重，且你无外网。按需 import 组件即可。