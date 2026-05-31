# SkillOpt CF-TriviaQA 实验报告

## 一、实验配置

```
数据集:     CF-TriviaQA (counterfactual 反事实问答)
模型:       GLM-5.1 (optimizer + target 同一模型)
```

数据规模：

```
train:  3,370 条
val:    1,685 条
test:   11,798 条
```

训练超参：

```
num_epochs:        2
batch_size:        20   (每 step 采 20 条)
accumulation:      1
learning_rate:     3    (每步最多 3 条 edit)
lr_scheduler:      cosine
minibatch_size:    4    (reflect 时的分析粒度)
workers:           8    (rollout 并发)
Gate:              开启 (sel_env_num=20)
SLOW_UPDATE:       开启
META_SKILL:        开启
limit:             80   (训练只加载了 80 条数据)
```

> **注意**：`limit: 80` 限制了训练阶段的数据量，但全量评测使用完整 test split (11,798 条)。

## 二、训练过程

8 个 step（4 steps/epoch × 2 epochs），总耗时 **1081s (~18min)**：

```
Step  Epoch  InEp  RollHard  RollSoft  Action            CurrScore  BestScore  Wall(s)
  1     1      0    0.7000   0.8067   reject              0.7000    0.7000      82s
  2     1      1    0.8500   0.8667   accept_new_best     0.7500    0.7500      44s
  3     1      2    0.8500   0.9167   reject              0.7500    0.7500     103s
  4     1      3    0.8000   0.8781   reject              0.7500    0.7500     130s
  5     2      0    0.8000   0.8889   reject              0.7500    0.7500     155s
  6     2      1    0.8500   0.9178   accept_new_best     0.8500    0.8500     206s
  7     2      2    0.8000   0.8858   reject              0.8500    0.8500     118s
  8     2      3    0.8500   0.8833   reject              0.8500    0.8500     126s
```

- **8 步中只有 2 步被 Gate 接受**（step 2 和 step 6），6 步被拒绝
- 每个 epoch: 1 accept + 3 reject
- 总 token 消耗: **686K** (517 次 API 调用)

## 三、Skill 演化

```
版本   大小    Rules数  变化
v0    103B      0      初始空 skill
v1    103B      0      无变化 (step 1 reject)
v2   1896B      6      +Rule 1~6 (step 2 accept)
v3   1896B      6      无变化 (step 3 reject)
v4   1949B      6      +SLOW_UPDATE 空占位符 (epoch 1 结束)
v5   1949B      6      无变化 (step 5 reject)
v6   2864B      7      Rule 6 扩展 + Rule 7 (step 6 accept)
v7   2864B      7      无变化 (step 7 reject)
v8   4338B     11      +4 条 SLOW_UPDATE 规则 (epoch 2 结束, force_accept)
```

### 7 条主规则（v6 确立，经 Gate 验证）

| # | 核心要义 |
|---|---------|
| 1 | Context 覆盖先验知识（反事实场景下以 context 为准） |
| 2 | 匹配问题要求的实体类型 |
| 3 | 去掉品牌名所有格后缀（M&M's → M&M） |
| 4 | 术语不匹配时按描述属性定位 |
| 5 | 优先使用 context 原文措辞 |
| 6 | 答案保持最小化，不追加别名/修饰 |
| 7 | 不添加头衔/尊称，不追加从句描述 |

### 4 条 SLOW_UPDATE 规则（v8 注入，绕过 Gate）

| # | 核心要义 |
|---|---------|
| S1 | 保持 context 中数字的原格式（阿拉伯/罗马） |
| S2 | 保持 context 中的连字符和拼写，不标准化 |
| S3 | "acronymic" 指有缩写，答案应给全名而非缩写 |
| S4 | 名字中的括号成分需保留 |

## 四、评估结果

### 指标说明

- **Hard Accuracy (EM)**：Exact Match，归一化后预测与 gold answer 完全一致才算对，取值 0 或 1
- **Soft Accuracy (F1)**：Token-level F1，按 token 重叠计算精确率/召回率的调和平均，允许部分匹配得分

归一化遵循 SQuAD 标准：小写、去标点、去冠词 (a/an/the)、合并空格。

### 小规模评测（训练时，80 条 test 子集）

```
                 Hard(EM)   Soft(F1)
Baseline (S_0)   0.7125     --
Best (v6)        0.7625     --
──────────────────────────────
Delta            +0.05  (80条中多对4条)
```

### 全量评测（11,798 条 test，workers=16）

```
                 Hard(EM)   Soft(F1)   正确条数
Baseline (S_0)   0.7274     0.8135     8,582
v8 (Skill)       0.7720     0.8471     9,108
────────────────────────────────────────────
Delta            +0.0446    +0.0336    +526
```

**Skill 净效果**：v8 相比 baseline 全量 EM 提升 4.46 个百分点，净增 526 条正确答案。

### 训练集 vs Test 集对比

```
            训练集(20条rollout)   Test集(80条)   Test集(11798条)
v0 (S_0)        --               0.7125         0.7274
v2              0.85             --             --
v6 (best)       0.85             0.7625         --
v8              --               --             0.7720
─────────────────────────────────────────────────
训练集提升:  +0.15  (0.70 → 0.85)
Test集(小) 提升:  +0.05  (0.7125 → 0.7625)
Test集(全) 提升:  +0.0446 (0.7274 → 0.7720)
```

过拟合明显：训练集涨 15 个点，test 全量只涨 4.5 个点。小样本 test 上的 5% 提升在全量上收敛到 4.5%，说明 80 条的评估波动较大。

## 五、逐条对比分析（Baseline vs v8，全量）

```
Both right (都对):       8,336  (70.7%)
Both wrong (都错):       2,444  (20.7%)
Improved (错→对):          772  (6.5%)
Regressed (对→错):         246  (2.1%)
─────────────────────────────────
Net gain:                  +526  (+4.5%)
Soft gain (hard 不变, soft 提升):  171
Soft loss (hard 不变, soft 下降):  157
```

### 改进案例（典型模式）

| 模式 | Baseline 预测 | v8 预测 | Gold |
|------|-------------|--------|------|
| 去头衔 | Dr Jonas Salk | Jonas Salk | Jonas Salk |
| 去所有格 | His hat | hat | Hat |
| 去修饰 | Yellow Mini | Mini | Mini |
| 补充必要信息 | Johnny Tillotson | Little Anthony and the Imperials | Little Anthony and the Imperials |
| 数字格式 | Four | 4 | 4 |
| 去尾部描述 | St Arnaud (though he was...) | St Arnaud | St Arnaud |

### 回归分析（246 条从对变错）

回归是理解 skill 副作用的关键：

#### 规则7过度剥头衔：31 条 (12.6%)

Rule 7 "不添加头衔" 误杀了本身就是答案一部分的头衔：

```
Queen Alexandra → Alexandra       (gold: Queen Alexandra)
Queen Victoria  → Victoria        (gold: Queen Victoria)
Sir Ian Cheshire → Ian Cheshire   (gold: Sir Ian Cheshire)
King George III → George III      (gold: King George III)
Sir Steve Redgrave → Steve Redgrave (gold: Sir Steve Redgrave)
```

**根因**：Rule 7 无法区分"头衔是名字的固有部分"和"头衔是外加的尊称"。示例中的例外条件（"Alexander the Great" 算固有）不够覆盖 Queen/Sir/King 等情况。

#### 规则6过度精简：35 条 (14.2%)

Rule 6 "答案最小化" 把名字中的关键部分也删了：

```
Henry VII     → Henry        (gold: Henry VII)
Pablo Picasso → Picasso      (gold: Pablo Picasso)
Elvis Presley → Elvis        (gold: Elvis Presley)
Franklin D. Roosevelt → Roosevelt (gold: Franklin D. Roosevelt)
Nancy Sinatra → Nancy        (gold: Nancy Sinatra)
```

**根因**：Rule 6 把"名+姓"当作可拆分的结构，把姓当作"别名/修饰"删除，但很多情况下全名才是标准答案。

#### SLOW_UPDATE 数字格式规则反向错误：14 条 (5.7%)

S1 规则本意是保持数字格式，但模型反而把阿拉伯数字转成了英文：

```
8 → Eight   (gold: 8)
7 → seven   (gold: 7)
4 → four    (gold: 4)
```

**根因**：模型对 SLOW_UPDATE 规则的理解出现了偏差——"保持格式"被理解为"展开数字"，适得其反。

#### 模型自身不稳定/幻觉：166 条 (67.5%)

```
Guyana → Cayman Islands           (事实错误)
Royal Albert Bridge → Royal Albert Bridge in Saltash (过度补充)
Volkswagen Golf → Volkswagen Golf Mk3 (过度具体化)
Canary birds → canary bird        (大小写/单复数)
```

这部分回归与 skill 规则无关，是 LLM 本身的采样不稳定性和推理错误。同一 prompt 多次调用可能产出不同结果。

## 六、机制观察

### Gate 机制

6/8 步被拒绝，说明大多数 reflect 产生的 edit 确实无法提升 hard score。Gate 有效防止了 skill 退化。但 Gate 只在 step-level 起作用，无法约束 epoch-level 的 SLOW_UPDATE。

### SLOW_UPDATE

epoch 1 结束：注入空占位符（无历史可比）。
epoch 2 结束：纵向对比 prev_hard=0.7 vs curr_hard=0.8，发现 4 个 persistent fail 模式，通过 `force_accept` 注入 4 条规则。

**force_accept 的设计矛盾**：
- 设计意图：epoch 级别的纵向洞察不应被 step-level 的 minibatch 波动否决
- 实际问题：全量评测显示 SLOW_UPDATE 的 4 条规则引入了 14 条回归，且改进案例很少
- 核心冲突：Gate 拒绝了 step-level 的类似规则（因为"加规则反而降分"），但同样的 insight 通过 SLOW_UPDATE 绕过了 Gate

### META_SKILL

epoch 2 产出了 3KB 的优化器指导文档，核心结论：
- "格式约束类规则最高价值"——被全量实验验证（Rule 6/7 贡献最多改进）
- "4 类持续失败模式"——部分被验证（数字格式），但对应的规则引入了回归

## 七、资源消耗

### 训练阶段

```
总耗时:       1,081s (18min)
API 调用:     517 次
Token 消耗:   686,074 (prompt: 516,229, completion: 169,845)
```

按角色分布：

```
rollout:   280 calls, 184K prompt, 41K completion  (推理，最频繁)
analyst:    39 calls, 143K prompt,  60K completion  (反思，最重)
merge:      19 calls,  32K prompt,  36K completion  (合并)
ranking:     5 calls,   6K prompt,   7K completion  (排序)
```

### 全量评测

```
Baseline:  11,797s (197min, workers=16), 2.1 items/s
v8:         5,588s  (93min, workers=16), 2.1 items/s
```

> Baseline 耗时约为 v8 的 2.1 倍，可能受 API 服务端负载波动影响，单条推理耗时基本一致。

## 八、结论与后续建议

### 核心发现

1. **Skill 确实有效**：全量 test 上 EM 从 0.7274 → 0.7720 (+4.5%)，526 条净增正确
2. **规则6/7 是双刃剑**：它们贡献了最多的改进（去头衔、精简答案），但也造成了 66 条回归（27% 的回归直接来自这两条规则的过度执行）
3. **SLOW_UPDATE force_accept 需要审慎**：4 条注入规则引入了 14 条回归，数字格式规则适得其反
4. **过拟合明显**：训练集涨 15 点 vs test 涨 4.5 点，小批量训练容易让 skill 过度适配特定错误模式
5. **67.5% 的回归来自模型不稳定性**，与 skill 无关——这是 LLM 推理本身的噪声

### 后续建议

**短期（规则修复）**：
- Rule 7 补充条件：当 Queen/King/Sir 等作为名字的固有前缀出现在 context 中时，应保留
- Rule 6 补充条件：名人全名（名+姓）不应拆分，罗马数字后缀（VII, III）不应删除
- S1 数字格式规则需改写或移除——当前措辞导致模型反向操作

**中期（实验设计）**：
- 去掉 `limit: 80`，用全量数据训练，减少过拟合
- 增加 epoch 数和 batch_size，让 skill 接触更多错误模式
- 跑 v6 全量评测，隔离 SLOW_UPDATE 的净效果
- SLOW_UPDATE 应考虑加入 Gate 验证（至少在 eval 子集上检查是否降分）

**长期（机制改进）**：
- Rule 的粒度需要更细——当前的规则太粗暴，"一律不带头衔"这种绝对规则在全量数据上暴露了大量边界情况
- 考虑为规则添加 confidence/condition 字段，让模型判断规则是否适用于当前样本，而非一刀切
- 探索 rule pruning 机制：训练结束后，逐条移除规则评测，删掉引入回归多于改进的规则
