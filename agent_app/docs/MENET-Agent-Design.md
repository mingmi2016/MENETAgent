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

### LLM 解释原则：结构化证据 + Harness

LLM 不仅用于训练结果解释，也应参与数据检查、质量判断和科研问答。工具层负责读取文件、计算样本数、SNP 数、匹配率、缺失率、重复样本和错误信息；LLM 负责在这些结构化证据之上组织自然语言解释。每类任务必须通过专用 harness 调用：输入明确包含 `task_intent`、允许使用的证据字段和任务目标，输出必须符合约定的 JSON Schema，并经过服务端校验后才能展示给用户。数据检查 harness 严禁生成 R²、基线排名或输入中不存在的数字；模型性能 harness 才允许解释 R² 和基线比较。LLM 不可用或输出不合规时才降级到规则解释，并在界面明确显示实际来源，不能把规则结果伪装成 LLM 结果。

### 模型推荐：LLM 提议、程序校验、程序执行

模型训练完成后，系统保存 MENET、基因组岭回归、随机森林和 XGBoost 的结构化评估结果。自动预测时，将同一数据划分下的指标、重复验证状态和可用模型信息交给模型服务，由 LLM 输出固定 JSON 格式的推荐方法、MENET 对照方法、置信度、理由和是否需要重复验证。

程序会校验推荐方法是否存在、是否具有真实评估指标、是否属于当前数据集和性状；推荐无效时才使用结构化结果降级。只有通过校验的推荐才会进入预测执行，程序负责加载对应模型文件并生成预测，LLM 不直接操作模型文件，也不能修改评估指标。预测结果明确区分主要预测模型和 MENET 对照，不进行未经验证的平均或加权集成。

### 自然语言入口原则

网页聊天中的用户输入统一先经过意图解析 Harness。`list_models`、`inspect_data`、`train_model`、`predict_trait`、`evaluate_model` 和 `explain_model` 都属于显式意图协议；数据集名称、模型 ID、性状和训练参数作为结构化参数提取，再由程序对数据库记录和路径进行解析。API 层不能用关键词分支替代意图识别。只有登录、权限、文件格式、路径安全和参数类型校验等确定性边界不交给 LLM。模型服务不可用时才使用规则解析，并在回复中标明降级来源。

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
evaluate_model / explain_model
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

### 执行后端与运行环境

执行后端采用配置驱动方式，不根据机器类型、`bsub` 或 Redis 的存在情况自动猜测。部署时通过环境变量选择：

```text
MENET_QUEUE_BACKEND=thread  # 本地进程内线程池，默认
MENET_QUEUE_BACKEND=celery  # Redis/Celery 异步 Worker
MENET_QUEUE_BACKEND=lsf     # 通过 bsub 提交到 LSF 集群
```

因此本地 WSL 和服务器可以使用同一套代码，只需提供不同的 `.env` 配置。`MENET_QUEUE_BACKEND` 决定任务由哪种执行后端承载；`device=auto` 则在任务真正运行的进程中由 PyTorch 判断 CUDA 是否可用，两者不能混为一谈。


## 9. 集群部署后的优化路线

当前版本已经完成登录节点 Web/API 服务、`gpu01` LSF 训练提交、本地线程后端和 Conda 双环境的基本闭环。下一阶段按以下顺序优化：

1. **认证和用户隔离**：服务器设置 `MENET_AUTH_MODE=required`，验证登录会话、管理员创建用户、任务/数据集/模型/结果的用户边界；启用 HTTPS 时设置 `MENET_COOKIE_SECURE=1`。
2. **LSF 状态恢复**：服务重启时根据 `.lsf_job_id` 和 `bjobs` 恢复 `PEND`、`RUN`、`DONE`、`EXIT` 和 `ZOMBI` 状态，不把仍在运行的作业直接标记为失败。
3. **端到端集成测试**：覆盖训练提交、排队、GPU 执行、结果读取、模型登记、取消和失败处理，并验证只有 `train_model` 进入 LSF。
4. **提交链路加固**：使用固定 Worker 脚本和 `task.json` 传参，减少 shell 字符串拼接；记录 `bsub`、`bjobs`、`bkill` 的错误、超时和标准输出。
5. **重复训练并发控制**：重复验证任务按组顺序执行，或设置明确的 GPU 并发上限，避免多次训练同时消耗 GPU 显存。
6. **服务进程管理**：为登录节点 Web 服务配置自动重启、日志轮转和健康检查，减少对 `nohup` 手动启动的依赖。

本地工作站继续使用 `MENET_QUEUE_BACKEND=thread` 和 `.venv`；上述集群加固不改变本地运行方式。


## 9.1 真实数据和超参数策略

### GB 级真实用户数据

当前上传接口采用分块写磁盘，因此不会把整个上传文件一次性放入 Web 进程内存。支持 CSV，也支持上传原始 PLINK `.tar.gz` 与 `.pheno` 文件；一个数据集保存全部检测到的表型性状，用户切换当前性状时不需要重复上传基因型。单性状 CSV 仍然兼容。PLINK 上传后，用户在数据面板点击“开始预处理”，任务提交至 LSF CPU 队列（`MENET_PREPROCESS_QUEUE`，默认 `cpu`）。处理流程校验 `.bed/.bim/.fam` 及表型样本 ID，在整个 `.bim` 范围均匀选择最多 8192 个候选标记，排除 MAF 小于 0.01 或缺失率高于 0.1 的标记，对剩余缺失值使用每标记中位数填补，生成 `genotype/genotype.csv` 与 `genotype/preprocessing_report.json`，原始文件保留。该文件是有界的标记子集，不等同于全基因组转换；训练结果依赖筛选策略。缺失表型值会在检查中报告，训练划分时跳过对应样本。默认数据集和预测基因型上传限制均为每个文件 2 GiB，可通过 `MENET_MAX_UPLOAD_BYTES` 调整。

真实数据上线前必须验证：

- 反向代理和 Web 服务的大文件请求限制、超时和连接保持；
- login 节点与 `gpu01` 对上传目录和结果目录的共享存储可见性；
- 分块数据检查、内存峰值和 GPU 显存需求；
- 每个用户的磁盘配额、任务配额和结果清理策略；
- 原始文件 SHA256、失败上传清理和可恢复上传；
- 大型基因型矩阵转换为 Parquet、memory-mapped 数组或受支持的二进制/基因型格式。

推荐的数据生命周期是：原始文件分块保存并校验 -> 后台建立数据集 -> 转换为内部高效格式 -> 分块检查和资源估算 -> 通过后提交训练。资源不足时应在训练前返回明确原因，而不是让 LSF 作业运行后才因内存或显存失败。

### 超参数控制边界

当前协议支持用户设置 `device`、`epochs`、数据划分策略、随机种子和是否计算 SNP 重要性；学习率、batch size、dropout、weight decay、margin 和网络结构仍来自配置文件。后续采用三层控制：

1. **自动模式**：根据样本数、SNP 数、缺失率、训练集规模和可用资源生成保守初始配置，并在显存不足时降低 batch size 或提前阻止任务。
2. **基础模式**：用户修改设备、训练轮数、数据划分和随机种子；每个参数都显示范围、预计资源和影响。
3. **高级模式**：只开放有明确 schema、范围和类型校验的超参数，不允许用户通过自然语言写入任意网络结构或命令。

自动调参只能使用训练集和验证集：先用少量轮次筛选候选，再对最优候选进行完整训练，最后锁定参数后只评估一次测试集。测试集 R² 不能用于选择超参数。LLM 可以解释欠拟合、过拟合、学习率或正则化问题并提出候选配置，但不能直接覆盖程序边界，也不能把测试集指标当作调参依据。

每个候选配置必须保存到 `training_config.json`，同时记录随机种子、数据划分、验证指标、资源使用、失败原因和最终 checkpoint。前端应明确区分“当前后端已支持的参数”和“计划中的高级调参”，避免出现可以填写但不会生效的控件。

### 前端使用原则

真实数据用户首先看到数据规模、文件大小、格式要求、上传进度和资源预检结果；训练界面提供自动/基础模式，明确显示队列后端、设备、轮数和数据划分；长任务显示排队、执行、GPU 作业号、当前阶段和取消状态。任何尚未接入后端的高级参数都应显示为规划项，而不是伪装成已生效的输入框。

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

## 12. MENET 自建 Harness 与 Codex Harness：比较和升级路线

更新日期：2026-10-13。本节记录当前代码核查结果和后续设计建议，不表示 Codex 集成或自动调参已经实现。前文的“本机优先”“第一版”和多 Agent 范式属于早期设计；当前已有集群部署，本节补充后续演进方向，不改变现有执行方式。

### 12.1 术语与核心判断

这里比较的是 **MENET 自建 harness 与 Codex harness**，不是“本地程序与 Codex”，也不是“是否接入 LLM”。两者都可以部署在本地；部署位置不是能力差异的根源。

- **LLM**：理解目标、分析证据、提出方案和下一步工具调用。
- **Harness**：提供上下文，组织模型与工具之间的执行循环，维护状态、进度和权限边界，处理异常、中断和结束。
- **MENET MCP**：以明确的工具协议暴露领域能力；MCP 本身是接口协议，不会自动产生优化能力。
- **MENET 计算与任务后端**：执行真实的数据处理、训练和预测，负责调度、持久化、资源限制和结果文件。

在使用相同 LLM 的前提下，当前系统与 Codex 的主要差距在 harness。当前 LLM 可能理解“预测效果不好，请优化”，但系统没有将这个开放目标转化为连续调查、实验和验证的通用执行循环。换更强的 LLM 不能单独补齐该缺口。

Codex harness 提供更成熟的通用 Agent 编排机制，适合多步调查和反馈驱动的实验组织；这不等于自动保证预测精度提高。效果仍取决于领域工具、数据质量、实验设计和验证证据，不能承诺未经测量的提升倍数。

### 12.2 当前真实调用链

以“请预测性状为 DTT”为例：

```text
Web 聊天请求
  -> AgentIntentParser.parse：LLM 提取 predict_trait 和 DTT；失败时规则降级
  -> API：结合页面/会话确定数据集、预测输入和可复用训练记录
  -> 必要时触发原始数据预处理；无兼容模型则提示先训练
  -> 部分自动选模路径调用 recommend_prediction_model
  -> 创建并提交后台任务
  -> MenetWorkflow：validate_dataset -> predict_trait
  -> 保存真实预测文件、result.json 和任务状态
  -> 页面请求 explanation 接口
  -> ResultInterpreter：LLM 解释结构化结果；失败时规则降级
  -> 页面展示说明、状态与下载入口
```

当前不是由 LLM 自由生成任意函数调用链；意图映射到 `agent/workflow.py` 中的预定义工作流。LLM 不计算预测值、不参与 PyTorch 梯度更新，也没有自动调参闭环。

关键实现入口：`agent/api.py`、`agent/llm.py`、`agent/workflow.py`、`agent/tools.py`、`agent/runner.py`。

### 12.3 能力比较

| 维度 | MENET 自建 harness（当前） | Codex harness（工具与授权齐备时） |
| --- | --- | --- |
| LLM 的主要用途 | 固定意图识别、参数提取、部分模型推荐、结果解释 | 围绕目标推理并持续选择下一步工具 |
| 标准检查/训练/预测 | 已有确定性流程与后台任务 | 可调用同一业务能力，无需重写 MENET 算法 |
| 多步任务 | 主要依赖预先编码的分支和步骤 | 根据工具反馈动态组织后续步骤 |
| 性能优化 | 尚无调查、调参、验证的自主闭环 | 可组织闭环，但不能凭空补出工具能力 |
| 上下文与恢复 | 有会话和任务记录；通用 Agent 状态能力有限 | 有成熟的会话、上下文和执行循环机制；与业务任务的恢复仍需集成 |
| 失败处理 | 主要是预定义错误处理和规则降级 | 可分析失败反馈并选择获准的下一步；不能保证所有失败自动修复 |
| 权限与预算 | 应由业务后端执行校验和限制 | Harness 增加调用/审批控制，不能代替业务鉴权和集群配额 |
| 大文件、LSF、模型存储 | 属于现有产品与任务基础设施 | 仍由 MENET 后端提供，不因接入 Codex 自动解决 |

固定流程并非必须全部取消：单次训练内部的步骤依赖、数据校验和科学规则应保持确定性。需要升级的是外层目标编排，而不是允许 Agent 绕过后端边界。

### 12.4 当前 MCP 能力与缺口

已核查 `agent/mcp_server.py`，源码定义了以下工具；“源码存在”不表示已经验证当前部署的远程 MCP 可用：

- `menet_inspect_dataset`：检查数据。
- `menet_submit_training`：提交训练，参数包括设备、轮数、划分方式、划分种子、训练种子和 SNP 解释开关。
- `menet_submit_prediction`：指定训练目录和新基因型 CSV 提交预测。
- `menet_submit_evaluation`、`menet_submit_explanation`：读取已有模型评估与解释证据。
- `menet_get_task`、`menet_cancel_task`、`menet_cleanup_task`：查询、请求取消和受限临时结果清理。

当前 MCP 使用 stdio、本进程线程池和内存 `_jobs` 注册表，直接调用 `MenetWorkflow`；并未复用 Web 的持久化任务存储与 LSF 提交链路。它依赖目录路径，不是面向多用户的统一受控任务入口。不能将它直接等同于已经可供真实用户安全使用的远程集群 MCP。

优化闭环要求“能观察 -> 能调整 -> 能验证”。下表是建议新增或完善的能力类别，**不是已实现的接口名称**：

| 能力类别 | 应提供的信息或动作 | 优先级 |
| --- | --- | --- |
| 数据与模型发现 | 按授权列举 dataset_id、性状、model_id 和兼容性，不让 Agent 猜服务器路径 | P0 |
| 统一任务生命周期 | 通过共享后端提交、查询、取消；返回持久 task_id 和结构化失败原因 | P0 |
| 训练证据读取 | 有效配置、种子、划分、逐轮 train/validation 曲线、checkpoint 与指标 | P0 |
| 受控超参数覆盖 | 学习率、batch size、dropout、weight decay、margin 等，白名单、范围校验并确保实际生效 | P1 |
| 可复现实验 | 固定数据版本/划分，登记候选配置、重复种子和实验组 | P1 |
| 实验比较 | 依据验证集或交叉验证，比较均值、波动、资源成本和失败记录 | P1 |
| 数据准备与资源预检 | 已上传原始数据的预处理状态、配置和资源估计，不引入普通用户服务器路径导入 | P1 |
| 工件安全读取 | 按授权返回有界日志、曲线与结果摘要/下载引用，不将 GB 原始矩阵直接发给 LLM | P1 |
| 预算与审批 | 实验次数、GPU 时间、并发和停止条件，在服务端强制执行 | P0 |

### 12.5 推荐目标架构

保留现有 Web 界面与业务后端，优先通过外部 Codex + MCP 验证能力，随后评估将 Codex harness 嵌入产品。普通用户继续上传文件、选择数据集和查看进度，不需要服务器账号或操作终端。

```text
入口 A：现有 Web -> 自建 harness（兼容/回退） ──────┐
入口 B：外部 Codex harness -> MENET MCP ────────────┤
入口 C：现有 Web -> 嵌入 Codex harness -> MENET MCP ┤（后续）
                                                  ↓
                   统一业务服务：鉴权、配置、预算、任务与工件
                                                  ↓
                       固定单任务工作流 -> Worker / LSF
                                                  ↓
                                MENET / 基线模型 / 存储
```

LLM 负责提出行动，Codex harness 组织调用，MCP 暴露受控能力，业务后端是权限、科学规则和资源约束的最终执行者。Codex 会话与 MENET task_id 建立关联，但不是同一对象；Agent 中断不应默默取消训练，也不应在恢复时重复提交。

Codex 不必安装在 GPU 计算节点：先确定 harness 部署位置、认证方式、MCP 网络连通性、数据出域政策和模型访问兼容性，再决定运行组件。不能假定当前任意 openai-compatible 服务地址和密钥可直接用于 Codex。

### 12.6 分阶段升级方案

**阶段一：梳理和统一工具基础（最先做）。**

1. 为现有 Web/API/MCP 建立能力清单、输入输出 schema 和缺口测试。
2. 抽取共享任务服务，MCP 与 Web 使用同一鉴权、TaskStore、任务调度和工件访问；服务器训练继续使用 LSF，本地运行继续支持 thread 和 `.venv`。
3. 以 dataset_id/model_id 替代面向用户的任意路径参数；绑定可信调用身份，不能靠 LLM 填写 user_id 证明权限。
4. 长任务立即返回 task_id，通过状态查询或事件继续；增加幂等提交、持久化恢复和预算限制。

验收：同一任务从 Web 与 MCP 均可查询；服务重启后记录可恢复；重试不重复提交；跨用户访问被拒绝。先做只读核查和受限冒烟测试，不自动创建大量训练。

**阶段二：外部 Codex + MENET MCP 概念验证。**

提供领域指引，说明性状/模型兼容性、数据要求、模型选择政策、验证规则和高成本操作确认。先验证“读取已有 DTT 结果并解释问题”，再验证“检查数据 -> 选择已训练模型 -> 预测 -> 查询结果”。不得绕过工具直接使用任意 shell 修改生产代码或训练配置。

验收：工具调用、证据与解释可追踪，缺模型会提示而不擅自训练，任务失败有明确下一步；对比时尽量固定 LLM、数据和工具，区分模型差异与 harness 差异。

**阶段三：实现预算约束的性能优化闭环。**

开放实际生效的受控超参数与实验比较接口，然后支持目标型请求：

```text
“DTT 预测效果不好，请在我批准的 GPU 预算内优化”
  -> 读取数据质量、划分、配置和训练曲线
  -> 提出有证据的假设与候选实验，给出成本和停止条件
  -> 用户批准后进行小规模候选训练
  -> 根据验证集/交叉验证结果筛选并重复验证
  -> 达到预算或停止条件后锁定方案
  -> 最终测试评估、推荐配置和完整实验记录
```

目标优化的是可验证的泛化表现，不是保证 MENET 一定胜过所有基线。无可靠提升时如实报告；缺少必要工具或证据时停止并解释缺口，不编造原因或指标。

**阶段四：将 Codex harness 嵌入现有 Web。**

评估 Codex app-server 的会话、事件流、中断和审批集成；SDK 可用于程序化概念验证。前端增加计划、工具执行进度、审批、预算与实验比较视图。认证/计费、并发隔离、服务可用性和 LSF 事件衔接需实际验证。保留自建 harness 的标准流程作为兼容入口，迁移失败不影响已有任务。

以上是路线，不是新增配置开关或已完成部署；本次文档更新不安装 Codex、不切换 8013 服务、不提交训练。

### 12.7 科学规则、模型选择与安全边界

- 调参只能用训练集和验证集或交叉验证，不反复使用最终测试集 R² 筛选候选。冻结配置后执行最终测试评估；已有测试指标只能作历史描述，不能宣称其在反复选择后仍是独立评估。
- 保存数据/标记版本、样本划分、随机种子、配置快照、代码版本、验证结果、模型文件与资源消耗。必要时加入亲缘结构、群体或批次感知的划分检查。
- GPU 预算、并发、最大实验次数、审批和幂等性必须由后端强制执行，不能只写在 prompt 中。
- 固定模型切换规则与 Agent 灵活规划可以并存。当前 `tools.py::_select_prediction_method` 使用 R² **绝对差值 > 0.05** 切换到最佳基线，否则保留 MENET；不是“相对提高 5%”，也不是统计显著性检验。
- 当前另有 `llm.py::recommend_prediction_model` 路径，其降级逻辑选择最高测试 R²，与上述阈值规则不一致。后续应统一并版本化模型选择政策，记录决策来源、理由、候选指标和覆盖权限；本节不修改现有规则。自动调参不能复用该测试集切换规则来选择超参数。
- Cloud LLM 默认只接收必要的结构化摘要；原始基因型、表型和个人信息出域需明确授权。日志、文件内容和工具结果不是执行指令。
- 工具应提供标准错误、运行阶段和可恢复信息；取消/中断/重试分别定义语义，避免无限重试和重复 GPU 作业。

### 12.8 官方依据与下一步

[OpenAI 官方：Codex as a platform — build on the open agent harness](https://developers.openai.com/blog/codex-as-a-platform) 说明 harness 承担 Agent 循环、会话状态、工具交互与执行事件，应用可保留自身界面、记录和控制，并通过 MCP 提供业务能力；也区分了 SDK 与 app-server 的集成定位。本节据此建议复用成熟 harness，而不是继续自行重建通用 Agent 运行时。具体部署和认证须在实施时再次核对。

**下一步建议：先完成阶段一的能力审计与共享任务服务设计，再开展外部 Codex + MCP 的受限概念验证；暂不直接启用自动调参或更换生产聊天入口。**
