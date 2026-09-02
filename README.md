### MeNet

MeNet is a mixed-effects deep neural network architecture for multi-environment agronomic traits prediction. This repository includes model deployment, training steps, prediction steps, and model parameter settings.

<img src="save/MeNet.png" alt="MeNet Overview" style="zoom: 150%;" />

### Set up

Set up environments using the following command:

```python
conda create -n MeNet python=3.9.17
conda activate MeNet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data Preparation

- The genotype data and phenotype datasets should be in a data folder, organizing as follows:

  ```python
  + data
      + gene
        ++ genotype.csv
      + phen
         ++ phenotype.csv
      + splits

  ```




### RepGeno to generate genetic relatedness

- To generate genetic relatedness, train the trait specific encoder of the RepGeno. The resulting genetic relatedness is then saved in `save/genetic_relatedness.pt`.

  ```python
  python train_trait_specific_encoder.py [parameters]

  [parameters]:
      -m --margin      # Contrastive loss margin (distance threshold)
      -p --phen_name   # Name of the target phenotype
      -d --device      # Device to use: 'cuda' or 'cpu'
      -f --flag         #  Sampling mode: 0 for trait-based, 1 for population-based positive/negative pairs
  ```

  The parameter settings are provided in [`configs/contrastive_learning.json`](configs/contrastive_learning.json). The constructed genetic relatedness will have a significant impact on the prediction performance of MeNet.



### Train MeNet and analyze the contribution between VE and RepGenp

- To train the MeNet model and evaluate the contributions of the VE and RepGeno modules, use the following command. The error tracking results during training and the contributions of the VE and RepGeno modules will be saved in `menet.log`

  ```python
  python menet.py [parameters] > menet.log

  [parameters]:
      -p --phen_name   # Name of the target phenotype
      -w --windows   # Windows mechanism 0: don't use windows_mechanism, 1: windows_mechanism by chromosome
      -d --device      # Device to use: 'cuda' or 'cpu'
  ```

​  The parameter settings are provided in [`configs/MeNet.json`](configs/MeNet.json).



### Transfer Learning

- To apply transfer learning, you can fine-tune a pre-trained model by selectively freezing and unfreezing specific layers. The following code snippet demonstrates how to configure the model after loading a pre-trained model:

  ```python
  # Freeze all parameters in the model
  for param in model.parameters():
      param.requires_grad = False

  # Unfreeze
  for param in model.fusion.field_for_ve.parameters():
      param.requires_grad = True
  for param in model.fusion.output.parameters():
      param.requires_grad = True
  ```

  ##### Explanation

  - **Freezing the model**: Setting `requires_grad = False` for all parameters prevents them from being updated during training, preserving the pre-trained weights.
  - **Selective unfreezing**: Specific layers are set to `requires_grad = True` to allow fine-tuning.
  - Ensure the pre-trained model is loaded before applying these settings.

### Acknowledgements

This work is based on [pytorch](https://pytorch.org/) ,  [scikit-learn](https://scikit-learn.org/),  and [plink](https://www.cog-genomics.org/plink/). The project is developed by following author and supervised by Prof. Xiangchao Gan(gan@njau.edu.cn)

Authors:

Yanhui Li  (huiyl@stu.njau.edu.cn): prototype development, data processing, model validation

Shengjie Ren (sunflower@stu.njau.edu.cn): overall framework design, contrastive learning strategy design, feature fusion mechanism, model architecture formulation, and interpretability and weight contribution analysis


## MENET Agent

### Positioning

MENET is the computation engine. MENET Agent is a domain agent for genetic breeding analysis. It accepts natural-language requests, identifies the user's intent, organizes a valid analysis workflow, calls MENET computation tools, monitors long-running jobs, and explains the results.

MENET is not a traditional single-marker GWAS significance test program. The current implementation is primarily a mixed-effects deep neural network for agronomic trait prediction. It uses an SNP feature branch (VE), a trait-specific genetic representation branch (RepGeno), and a fusion module to predict phenotypes. Current outputs include prediction loss, R2, and the relative contribution of VE and RepGeno. SNP-level association statistics such as p-values, FDR, and Manhattan plots require additional analysis modules.

### Agent architecture

```text
User / C++ client / Web client
              |
              v
       MENET Agent service
       - intent recognition
       - parameter extraction
       - parameter validation
       - workflow orchestration
       - job and conversation state
       - result interpretation
              |
              v
       MENET computation tools
       - validate_dataset()
       - create_split()
       - train_trait_encoder()
       - build_relatedness()
       - train_menet()
       - evaluate_model()
       - explain_model()
       - generate_report()
              |
              v
       MENET Worker / GPU node
```

The API or tool layer does not have to contain an AI model. It provides stable, deterministic methods for data validation, training, prediction, interpretation, and report generation. The agent layer uses a large language model to understand requests and explain results, while application code controls validation, permissions, workflow transitions, and execution. The model should never invent metrics or directly execute arbitrary shell commands.

MCP is not required for the first version. The agent can call internal Python functions or a private REST API, and external clients can use the public REST API. An MCP adapter can be added later if direct tool discovery by third-party AI clients becomes necessary.

### Typical interaction

For a request such as:

```text
Analyze this rice plant-height dataset with MENET, automatically split the data,
use the GPU, and report the most important SNPs.
```

The agent should convert the request into a structured task:

```json
{
  "intent": "train_menet",
  "trait": "plant_height",
  "split_strategy": "auto",
  "device": "cuda",
  "explain_snps": true
}
```

It then executes a controlled workflow:

```text
validate_dataset
        -> create_split
        -> train_trait_encoder
        -> build_relatedness
        -> train_menet
        -> evaluate_model
        -> explain_model
        -> generate_report
```

The agent may ask for missing information, or apply documented defaults. Every result must come from the computation layer and should be saved with the input configuration, software version, logs, metrics, and output files.

### Recommended service deployment

The production system should be deployed as a service. The local project directory is a development and testing location, not a requirement for end users.

```text
ChatGPT / Claude / local LLM
              |
              | model API or function calling
              v
       MENET Agent API
              |
              v
       Workflow orchestrator
              |
       Redis/Celery task queue
              |
              v
       MENET GPU Worker
              |
       Object storage + database
```

The first production version can use:

- FastAPI for chat, file upload, job status, and result download APIs;
- an OpenAI-compatible model API for intent recognition and result explanation;
- an orchestrator implemented in Python for deterministic workflow control;
- Redis and Celery for asynchronous training jobs;
- PostgreSQL or SQLite for users, jobs, configurations, and status;
- MinIO or S3-compatible storage for input data, models, predictions, plots, and reports;
- a CUDA-enabled worker for PyTorch training.

### Web application

The repository includes a lightweight web application served by the same FastAPI service. It is intended for interactive use by researchers and shares the exact same HTTP/JSON API used by C++ and other AI clients.

Start the service in WSL:

```bash
cd /home/mingmi/workspace/MenetAgent/MENET
./scripts/start_agent.sh
```

Open the following URL in a browser:

```text
http://localhost:8010/app/
```

The web application supports local researcher profiles, multiple CSV genotype/phenotype datasets, shared crop demos, natural-language task submission, task polling, result explanation, and artifact downloads. It also provides dataset scale summaries, long-task confirmation, real epoch progress, queue/execution timing, cooperative cancellation, recent task history, automatic reuse of compatible trained models, and actionable failure guidance. API exploration is available at `http://localhost:8010/docs`.

When no LLM is configured, the chat endpoint uses a deterministic rule parser. The following requests are supported examples; replace `culmlength` with the phenotype filename stem:

```text
请检查数据，性状为 culmlength
请训练模型，性状为 culmlength，随机划分，使用 GPU，训练 50 轮
请预测性状为 culmlength
请评估模型，性状为 culmlength
请解释模型，性状为 culmlength
请生成报告，性状为 culmlength，随机划分，使用 GPU，训练 50 轮
请检查小麦数据
请分析水稻株高第二批
请使用上一个数据集重新训练 10 轮
```

Prediction, evaluation, and explanation reuse artifacts already present in the selected output directory. The report request runs the full training workflow before generating an HTML report.

Four reproducible public-data demos are available immediately after startup: two rice traits derived from RAINBOWR's Rice Diversity Panel data and two wheat environments derived from BGLR's CIMMYT data. See [`data/demo/README.md`](data/demo/README.md) for provenance, citations, licenses, and reproduction instructions. They are software demonstrations, not substitutes for an appropriately designed GWAS or independent biological validation.

For repeated uploads, give each dataset a meaningful name and species. A conversation records recently used datasets, so the agent can resolve the current selection, an exact dataset name, species phrases such as `水稻` or `小麦`, and references such as `上一个数据集`. Explicit wording in the message takes precedence over the selector currently shown in the web page.

### Local persistence and task isolation

The development service stores its state locally and separates it by researcher profile:

```text
uploads/{user_id}/{dataset_id}/       uploaded genotype and phenotype CSV files
data/demo/                            shared rice and wheat demonstration datasets
runs/agent.db                         SQLite users, tasks, datasets, conversations, and messages
runs/{user_id}/{task_id}/task.json    immutable task input snapshot
runs/{user_id}/{task_id}/             models, metrics, predictions, explanations, and reports
```

Uploaded datasets are registered with their owner, display name, species, original filenames, byte sizes, and SHA-256 hashes. Conversation state records the active and recent datasets, trait, output root, task, and run directory; the web application restores profile-specific state after a page refresh. Private uploads, conversations, tasks, and run directories are isolated by `user_id`, while demo datasets are read-only shared entries.

Random train/validation/test splits are generated inside the task's isolated run directory. The original uploaded or shared dataset is not modified. Prediction, evaluation, and explanation automatically select the newest completed training run belonging to the same user, dataset, and trait; when no compatible model exists, the Agent asks the user to train first instead of starting a doomed task.

The profile selector is deliberately lightweight and does not prove a user's identity. The development server binds to `127.0.0.1` by default, so this is suitable for local workflow testing. Before exposing the service to a laboratory network or the public internet, replace the caller-supplied `user_id` with server-verified login/session tokens and add HTTPS, quotas, audit logs, and authorization checks. Set `MENET_AGENT_HOST=0.0.0.0` only after that deployment boundary is in place.

The task monitor displays the current workflow stage and separates queue wait from execution time. The four bundled demo datasets use measured RTX 5060 benchmark ranges; other datasets initially report that no benchmark is available. After a matching task completes, later runs use the median of that user's completed runs with the same dataset, intent, device, epoch count, and SNP-explanation setting. See [`docs/Local-Benchmark.md`](docs/Local-Benchmark.md) for the measured configuration and results.

SQLite is appropriate here for local use and small teams with one API process because JSON task arguments, LLM structured output, messages, and workflow state can all be stored as text columns transactionally. Move to PostgreSQL or MySQL when the service has multiple API/worker hosts, sustained concurrent writes, high availability or replication requirements, centralized backups, or stricter account and permission management. The storage boundary is encapsulated by `TaskStore`, so that migration need not change the Agent's JSON contract.

Training must be asynchronous because a MENET job can take minutes or hours. A submission should return a job ID rather than hold an HTTP request open:

```http
POST /api/v1/chat
POST /api/v1/datasets/upload
GET  /api/v1/tasks/{task_id}
GET  /api/v1/tasks/{task_id}/explanation
```

Suggested job states are:

```text
queued -> validating -> preparing -> training_encoder ->
building_relatedness -> training_menet -> explaining -> completed
```

Failure and cancellation states should also be recorded. Each job should have an isolated working directory or object-storage prefix, for example:

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

### Agent responsibilities

The agent should support at least these intents:

```text
inspect_data       Check whether input files satisfy MENET's data contract
prepare_data       Clean data and create reproducible train/validation/test splits
train_model        Train the trait-specific encoder and MENET
predict_trait      Predict phenotypes for existing or new samples
evaluate_model     Report loss, R2, MAE, and repeatability metrics
explain_model      Explain VE, RepGeno, chromosome, and SNP contributions
compare_methods    Compare MENET with traditional GWAS or prediction methods
generate_report    Produce a structured HTML or PDF analysis report
```

The agent must validate at least:

- sample IDs between genotype and phenotype files;
- genotype dimensions, numeric encoding, missing values, and duplicate markers;
- SNP naming and chromosome ordering when window mode is enabled;
- phenotype missing values and target trait selection;
- train/validation/test overlap and possible sample leakage;
- CPU/GPU availability and compatible model configuration;
- required intermediate artifacts before each workflow step.

The recommended separation of responsibilities is:

```text
Large language model: understand requests and explain results
Agent code: validate, plan, authorize, execute, and track workflows
MENET: train, predict, calculate relatedness, and produce metrics
```

### C++ client integration

The existing C++ software can remain a client of the service. It can provide file selection, progress display, charts, and report viewing, while calling the remote service over HTTPS:

```text
C++ desktop client -> POST /api/v1/jobs
C++ desktop client -> GET  /api/v1/jobs/{job_id}
C++ desktop client -> download result files
```

For organizations that cannot upload genotype data, the same agent can be deployed inside an enterprise network. A hybrid deployment can keep raw data and GPU computation inside the organization and send only validation summaries, metrics, and selected results to a hosted language model.

### Development roadmap

1. Turn the current scripts into reliable computation tools with structured JSON results.
2. Add data validation, automatic reproducible splits, model checkpoints, predictions, metrics, and training history.
3. Implement the Agent orchestrator and its state machine.
4. Add FastAPI, asynchronous jobs, authentication, storage, and report download.
5. Add SNP-level attribution with repeated training or permutation stability analysis.
6. Add comparisons with MLM, FarmCPU, BLINK, rrBLUP, Elastic Net, or other appropriate baselines.
7. Integrate the C++ client and optionally add an MCP adapter for third-party AI clients.

The first milestone is a service that can validate a dataset and execute a reproducible MENET training job from a structured task. Natural-language interaction should be added on top of that verified computation path.

详细的 Agent 工程设计、JSON 协议、上下文管理、工作流状态机、安全护栏、设计范式和参考项目见：[docs/MENET-Agent-Design.md](docs/MENET-Agent-Design.md)。

当前 Agent 核心代码位于 [agent/](agent/)，包含结构化任务模型、受控工作流和 MENET 工具边界。使用说明见：[agent/README.md](agent/README.md)。


