# MENET Agent 设计说明

## 1. 产品定位

MENET 是实际执行遗传育种分析的计算引擎。本项目包含两个入口：面向网页用户、内部结合 LLM 的 MENET Agent，以及面向 Codex/Claude 等外部 AI、只提供结构化工具的 MENET MCP Server。

```text
网页用户 ── REST ──> MENET Agent ── LLM、上下文、规划 ──┐
                                                         ├─> 共享业务服务
Codex/Claude ─ MCP ─> MENET MCP Server ── 结构化工具 ──┘    │
                                                            ▼
                                                   MENET 工作流
                                                            │
                                                   Worker / GPU
                                                            │
                                                   MENET 算法核心
```

推荐的职责边界如下：

```text
网页 Agent：理解用户表达，提取参数，维护对话上下文，解释结果
外部 AI：通过 MCP 理解用户表达，规划并选择 MENET 工具
MCP Server：暴露结构化工具，校验参数，提交任务，返回结果
共享业务服务：数据集、任务、模型和结果管理
Agent代码：校验参数，选择工作流，授权和执行工具，维护状态
工作流引擎：管理步骤依赖、重试、暂停、恢复和取消
MENET：训练模型、预测表型、计算遗传相关性
存储：保存任务、配置、状态、模型和结果工件
```

大语言模型不能直接执行任意 Python 或 shell 命令，也不能自行编造训练指标。所有关键计算结果必须来自 MENET 工具。

### 当前开发边界

当前只开发和验证一个本机优先版本：

```text
Web UI -> MENET Agent（本地 Ollama）-> MENET 工作流 -> 本机 GPU
Codex/Claude -> 本地 MCP Server -> 同一套 MENET 工作流
```

现阶段不解决任意新电脑的一键安装、通用 Windows 兼容、服务器多租户、云端部署和 C++ 桌面外壳。这些属于后续交付阶段。当前优先完成模型资产管理、新材料预测、数据理解、结果反思、真实 LLM 和本地 MCP 的完整闭环。

Web Agent 的 LLM 设置必须在界面中可见，包括启用状态、服务类型、服务地址、意图识别模型、结果分析模型、连接测试和规则降级状态。意图模型承担高频分类和参数抽取，分析模型承担结果解释和复杂科研问答。本地 Ollama 无需 API Key；第三方服务的密钥只保存在被 Git 忽略的本地配置中。配置失败时不能影响确定性 MENET 工作流。

## MCP Server and External AI

MCP 不是第二个 Agent。使用网页时，MENET Agent 内部可以结合 LLM；使用 Codex 或 Claude 时，外部 AI 已经提供 LLM、规划和对话上下文，MENET MCP Server 只提供可靠的领域工具。

```text
Web chat:
  用户 -> MENET Agent -> LLM -> 共享业务服务 -> MENET

External AI:
  用户 -> Codex/Claude -> MCP tool call -> 共享业务服务 -> MENET
```

推荐暴露高层工具：

```text
menet_list_datasets       menet_inspect_dataset
menet_submit_training     menet_submit_prediction
menet_submit_evaluation   menet_submit_explanation
menet_get_task            menet_cancel_task
menet_list_models         menet_get_result
```

当前本机 stdio 实现提供 8 个工具：`menet_inspect_dataset`、`menet_submit_training`、`menet_submit_prediction`、`menet_submit_evaluation`、`menet_submit_explanation`、`menet_get_task`、`menet_cancel_task` 和 `menet_cleanup_task`。它们直接调用 `MenetWorkflow`，不经过 REST 回环请求，也不调用 LLM。

训练、评估和解释是长任务，工具应快速返回 `task_id`、状态和预计耗时；外部 AI 再调用 `menet_get_task` 或 `menet_get_result` 查询。不要把 `train_repgeno`、`build_relatedness` 等内部步骤暴露给模型，完整依赖关系由 MENET 工作流控制。

REST 和 MCP 都直接调用共享业务服务，不应让 MCP 通过回环 HTTP 调用 REST。REST 面向网页、C++、文件上传下载和常规集成；MCP 面向可发现、可结构化调用的 AI 工具。

本机单用户阶段可以使用 stdio MCP，不需要登录、聊天记忆或 MCP 内置 LLM。MCP Server 也不需要保存用户档案、对话历史或长期文件库。调用方应提供本次分析所需的数据路径或数据引用；服务端只在 MENET 计算期间使用输入数据，并将任务结果以 MCP 的结构化结果返回给 Codex/Claude。

需要区分“用户文件持久化”和“计算临时文件”：训练过程仍可能需要在本地生成临时工作目录、模型检查点、日志和中间矩阵，否则无法完成长时间计算。默认策略是任务完成或失败后按保留策略清理这些临时文件；如果用户要求下载模型、预测表或 HTML 报告，MCP 工具应返回明确的结果摘要，以及可供外部 AI 读取或转交用户的工件引用，而不是把大型文件塞进对话上下文。

因此，MCP 本机模式的边界是：

```text
不保存：登录账号、聊天历史、长期用户记忆、默认用户文件库
必须存在：本次计算可访问的输入数据、任务执行期间的临时文件
返回给 Codex/Claude：task_id、状态、指标摘要、警告、错误和结果工件引用
```

如果 MCP 改为远程 Streamable HTTP 服务，则不能继续假定“无需登录”：远程共享服务至少需要 API Token 或 OAuth，并需要用户、项目、数据和结果的权限隔离。未来提供远程服务时，再增加 Token/OAuth、用户隔离、配额和 HTTPS。

## MCP 连接方式与 Bearer Token

Codex 的自定义 MCP 配置通常提供 `STDIO` 和 `Streamable HTTP` 两种类型。两者的适用场景不同：

```text
STDIO：Codex 启动本机 MCP Server 进程
       本机单用户阶段通常不需要登录或 Bearer Token

Streamable HTTP：Codex 访问 http://host:port/mcp
                 适合独立服务、局域网或远程部署
```

Codex 配置中的“Bearer 令牌环境变量”表示环境变量的名称，例如：

```text
MCP_BEARER_TOKEN
```

Codex 会读取该环境变量的值，并在 HTTP 请求中发送：

```http
Authorization: Bearer <token-value>
```

这只是客户端发送凭证；MENET MCP Server 必须主动验证该请求头，配置才具有安全作用。仅在 Codex 中填写环境变量名，并不会自动为服务端增加认证。

当前项目的 `8010` 服务主要提供 REST 和 Web Agent 接口。MCP Server 完成后，建议提供本机 stdio 启动器，并可选提供 `/mcp` Streamable HTTP 入口。部署阶段遵循以下策略：

```text
本机 WSL：stdio，无登录、无 Token
本机 HTTP：127.0.0.1:8010/mcp，可暂不认证
远程 HTTP：Bearer Token 或 OAuth + HTTPS + 权限隔离
```

## 2. JSON 结构化输出

JSON 是大模型、Agent 和工具之间的通信契约。建议使用 Pydantic 或 JSON Schema 约束所有输入和输出。

### 任务请求

```json
{
  "intent": "train_model",
  "trait": "culmlength",
  "dataset_id": "dataset_001",
  "device": "auto",
  "split_strategy": "random",
  "train_ratio": 0.7,
  "valid_ratio": 0.15,
  "test_ratio": 0.15,
  "explain_snp": true
}
```

### 工具返回

```json
{
  "success": true,
  "status": "completed",
  "job_id": "job_001",
  "data": {
    "sample_count": 1200,
    "snp_count": 850000,
    "test_r2": 0.56,
    "test_mae": 8.31
  },
  "artifacts": [
    {
      "name": "metrics.json",
      "file_id": "file_001"
    }
  ],
  "warnings": [],
  "errors": []
}
```

建议所有工具统一包含 `success`、`status`、`errors`、`warnings` 和 `artifacts` 字段。大文件、完整日志和基因型矩阵不要直接放进对话上下文，只返回摘要和文件 ID。

## 3. 上下文管理

上下文策略取决于入口。Web Agent 需要维护对话上下文；MCP Server 不需要保存聊天上下文，因为 Codex 或 Claude 负责这部分。两种入口都需要保存任务上下文和计算工件，不能把长时间运行任务的状态只放在聊天文本中。

| 层级 | 内容 | 推荐存储 |
| --- | --- | --- |
| Web Agent 对话上下文 | 最近几轮用户对话、当前数据集和性状 | 对话服务或 Redis |
| MCP 会话上下文 | 不由 MENET 保存；由外部 AI 管理 | Codex/Claude |
| 任务上下文 | 参数、步骤、状态和错误 | SQLite（本机）或 PostgreSQL |
| 项目上下文 | 数据格式、模型版本、默认配置 | 项目数据库或配置文件 |
| 长期记忆 | 用户偏好和已确认的规则 | PostgreSQL |
| 大型工件 | CSV、模型、图表、报告 | MinIO/S3 |

任务状态不能只存在聊天文本中。服务重启后，Web Agent 或外部 AI 都应能通过 `task_id` 从数据库恢复任务上下文。

## 4. 工具调用

MENET 工具应该是职责单一、参数明确、结果稳定的方法：

```text
inspect_dataset()
validate_dataset()
create_split()
train_trait_encoder()
build_relatedness()
train_menet()
predict_trait()
evaluate_model()
explain_model()
generate_report()
```

建议同时提供细粒度工具和一个高级工作流工具：

```text
submit_menet_analysis()
```

高级工作流适合普通用户，细粒度工具适合 C++ 客户端、调试和从中间步骤恢复。

训练任务应异步执行：

```python
job = await submit_menet_analysis(task)
# 返回 job_id，不阻塞 HTTP 请求
status = await get_job_status(job.job_id)
result = await get_job_result(job.job_id)
```

不要把大模型生成的文本拼接成 shell 命令。工具调用必须经过白名单、类型检查、范围检查和权限检查。

## 5. 记忆和持久化

MENET Agent 至少需要保存以下数据：

- 用户和项目配置；
- 数据集元数据和校验摘要；
- 数据划分和随机种子；
- 模型配置、代码版本和依赖版本；
- 训练任务状态、日志和重试记录；
- 模型、预测值、指标和解释结果；
- 用户确认过的偏好和隐私设置。

结构化任务和指标放数据库，原始数据和结果文件放对象存储，短期状态和队列信息放 Redis。不要把所有内容都放进向量数据库，也不要用向量检索代替任务状态机。

每个任务建议使用独立存储前缀：

```text
jobs/{user_id}/{job_id}/
    config.json
    data_validation.json
    training.log
    metrics.json
    predictions.csv
    trait_specific_encoder.pt
    genetic_relatedness.pt
    menet_model.pt
    snp_importance.csv
    report.html
```

## 6. 规划和任务分解

用户请求：

```text
分析株高数据，训练 MENET，并找出重要 SNP。
```

可以解析为：

```text
1. 确认目标性状和输入数据
2. 检查样本 ID、SNP 格式和缺失值
3. 创建可复现的数据划分
4. 训练性状特异编码器
5. 生成遗传相关性矩阵
6. 训练 MENET
7. 评估测试集表现
8. 计算 SNP 或染色体贡献
9. 生成报告
```

这里应区分两种计划：

- 观察性计划：大模型可以根据用户意图提出建议；
- 执行性计划：必须使用预定义工作流和代码校验。

MENET 的训练流程有明确依赖关系，不适合完全自由的 Agent Loop。大模型负责选择工作流和填写参数，Agent 代码负责决定哪些步骤可以真正执行。

## 7. 状态管理和工作流

建议使用显式状态机：

```text
created
  -> validating
  -> preparing
  -> training_encoder
  -> building_relatedness
  -> training_menet
  -> evaluating
  -> explaining
  -> completed
```

异常状态包括：

```text
validation_failed
waiting_for_user
failed
cancelled
paused
```

每次状态变化都保存事件：

```json
{
  "job_id": "job_001",
  "from": "training_encoder",
  "to": "building_relatedness",
  "artifact": "trait_specific_encoder.pt",
  "timestamp": "2026-09-01T12:00:00Z"
}
```

工作流需要支持断点恢复、失败重试、任务取消、GPU 排队、超时控制、中间结果保存和人工确认。

第一版可以使用 `FastAPI + Redis + Celery + PostgreSQL`。如果以后需要跨机器可靠恢复、长时间运行和复杂依赖，可以评估 Temporal。Airflow 更适合批量数据管道和定时任务，不一定适合作为对话 Agent 的主状态机。

## 8. 安全和护栏

MENET 处理的基因型数据可能具有敏感性，至少需要：

- 文件大小、类型和内容检查；
- 防止路径穿越和任意输出路径；
- 每个任务使用隔离目录；
- 工具白名单和参数范围限制；
- 禁止任意 shell/Python 执行；
- 用户、项目和文件权限隔离；
- HTTPS、对象存储访问控制和审计日志；
- CPU、GPU、磁盘和并发任务配额；
- 数据保留期限和删除机制；
- 云端大模型只接收必要的统计摘要。

结果解释必须区分：

```text
模型预测结果 != 模型重要性结果
模型重要性结果 != 统计显著性结果
统计显著性结果 != 生物学因果结论
```

例如，Agent 应该说“该 SNP 在当前模型中具有较高贡献”，而不能直接说“该 SNP 是致因基因位点”。

## 9. 可参考的 Agent 设计范式

### ReAct

```text
思考 -> 调用工具 -> 观察结果 -> 再决定下一步
```

适合探索性问答和动态排错，但不应直接控制完整 MENET 训练流程。

### Plan-and-Execute

```text
生成结构化计划 -> 按步骤执行 -> 根据结果修正
```

适合 MENET。计划应当落在预定义工作流中，而不是允许模型生成任意函数调用链。

### Router

先判断请求类型，再进入对应工作流：

```text
inspect_data / train_model / predict_trait /
evaluate_model / explain_model / generate_report
```

适合 MENET，因为数据检查、训练和结果解释的工具集合不同。

### Supervisor

```text
Supervisor
├── DataAgent
├── TrainingAgent
├── ExplanationAgent
└── ReportAgent
```

适合系统扩大后的多模块产品。第一版建议使用一个主 Agent 加确定性工具，避免多个自主 Agent 之间互相传递错误。

### State Machine / Workflow-as-Code

把步骤依赖写进代码：

```text
验证完成才能训练
编码器完成才能生成遗传相关性
遗传相关性完成才能训练 MENET
训练完成才能解释结果
```

这是 MENET 最核心的范式。

### Human-in-the-Loop

在高成本或高风险步骤要求确认：

```text
检测到缺失率为 8.3%，是否继续？
本次训练预计占用 GPU 2 小时，是否开始？
是否允许把汇总指标发送给云端模型？
```

### Event-Driven Agent

工具完成后发出事件，例如 `DatasetValidated`、`MenetTrainingCompleted` 和 `ReportGenerated`，下一个步骤订阅事件后执行。适合后期分布式和高并发部署。

## 10. 适合参考的实现

以下项目可以参考其设计思想和工程实现，不建议直接复制成 MENET 的业务逻辑：

| 项目 | 适合参考的部分 | MENET 中的对应位置 |
| --- | --- | --- |
| LangGraph | 有状态图、节点、条件边、暂停和恢复 | MENET 工作流和状态机 |
| Temporal | 长任务、可靠重试、断点恢复、人工等待 | GPU 训练任务编排 |
| Celery | Python 异步任务、队列和 Worker | 第一版训练 Worker |
| PydanticAI | 类型安全的结构化输出和工具参数 | 任务协议和工具调用 |
| OpenAI Agents SDK | Agent、工具、结构化结果和追踪 | 大模型接入层 |
| AutoGen / CrewAI | 多 Agent 协作概念 | 后期多模块协作研究 |

对 MENET 最有价值的参考组合是：

```text
Pydantic：定义任务和工具协议
LangGraph或自定义状态机：管理 Agent 流程
Celery：执行第一版异步训练任务
Temporal：未来替换为更强的长流程编排
FastAPI：提供外部服务接口
```

## 11. 最小可行实现

第一版建议只支持以下意图：

```text
inspect_data
train_model
evaluate_model
explain_model
generate_report
```

推荐调用链：

```text
用户请求
  -> LLM结构化解析
  -> Pydantic校验
  -> Router选择工作流
  -> 状态机创建任务
  -> 工具调用和Worker执行
  -> 保存结果和状态
  -> LLM解释结果
```

第一阶段的完成标准不是“能聊天”，而是：同一份数据、同一份配置和同一个随机种子可以得到可追踪、可复现、可恢复的 MENET 任务结果。
