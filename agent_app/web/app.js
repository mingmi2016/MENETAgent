const $ = (id) => document.getElementById(id);
let currentUserId = localStorage.getItem("menet_user_id") || "user_local";
let activeTaskId = "";
let conversationId = null;
let predictionGenotypePath = "";
let selectedModelId = "";
const taskPollers = new Map();
const taskNodes = new Map();
const knownTaskStatuses = new Map();

const userQuery = () => `user_id=${encodeURIComponent(currentUserId)}`;
const conversationKey = () => `menet_conversation_id:${currentUserId}`;
const taskKey = () => `menet_task_id:${currentUserId}`;
const modelKey = () => `menet_model_id:${currentUserId}`;
const terminalStatuses = new Set(["completed", "failed", "cancelled"]);
const intentLabels = {
  inspect_data: "数据检查",
  train_model: "模型训练",
  predict_trait: "性状预测",
  evaluate_model: "模型评估",
  explain_model: "模型解释",
  generate_report: "完整报告",
};
const statusLabels = {
  created: "排队中",
  running: "启动中",
  validating: "检查数据",
  preparing: "准备数据",
  training_encoder: "训练编码器",
  building_relatedness: "构建相关性",
  training_menet: "训练 MENET",
  evaluating: "评估中",
  explaining: "解释中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function notice(message, error = false) {
  const element = $("notice");
  element.textContent = message;
  element.className = `notice${error ? " error" : ""}`;
}

function clearNotice() {
  $("notice").className = "notice hidden";
}

function showLlmResult(message, error = false) {
  const element = $("llm-result");
  element.textContent = message;
  element.className = "inline-result" + (error ? " error" : "");
}

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.join("；") : data.detail;
    throw new Error(detail || "请求失败");
  }
  return data;
}

function scrollConversation(behavior = "smooth") {
  const messages = $("messages");
  messages.scrollTo({ top: messages.scrollHeight, behavior });
}

function addMessage(text, role, scroll = true) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "M";
  const content = document.createElement("div");
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  content.appendChild(paragraph);
  item.append(avatar, content);
  $("messages").appendChild(item);
  if (scroll) scrollConversation();
  return item;
}

function makeTaskMessage(taskId) {
  if (taskNodes.has(taskId)) return taskNodes.get(taskId);

  const item = document.createElement("div");
  item.className = "message agent task-message-item";
  item.dataset.taskId = taskId;

  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.textContent = "M";

  const content = document.createElement("div");
  content.className = "task-bubble";
  content.innerHTML = `
    <div class="task-bubble-heading">
      <strong class="task-title">正在准备分析</strong>
      <span class="task-status-badge">排队中</span>
    </div>
    <p class="task-runtime">正在获取任务状态...</p>
    <div class="progress"><span></span></div>
    <p class="task-background-note">任务在服务器后台运行，可以离开页面后再回来查看。</p>
    <div class="task-inline-actions">
      <button class="text-button cancel-inline hidden" type="button">取消任务</button>
    </div>
    <div class="chat-result hidden">
      <p class="result-explanation"></p>
      <details class="structured-result"><summary>查看结构化结果</summary><pre></pre></details>
      <div class="artifact-list"></div>
    </div>`;

  content.querySelector(".cancel-inline").addEventListener("click", async () => {
    if (!window.confirm("确定停止这个任务吗？已经完成的训练轮次不会作为完成模型发布。")) return;
    try {
      const response = await request(`/api/v1/tasks/${encodeURIComponent(taskId)}/cancel?${userQuery()}`, { method: "POST" });
      content.querySelector(".task-runtime").textContent = response.message;
      content.querySelector(".cancel-inline").classList.add("hidden");
      scheduleTaskPoll(taskId, 800);
    } catch (error) {
      notice(error.message, true);
    }
  });

  item.append(avatar, content);
  $("messages").appendChild(item);
  taskNodes.set(taskId, content);
  return content;
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value} 秒`;
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  return hours ? `${hours} 小时 ${minutes} 分钟` : `${minutes} 分钟`;
}

function runtimeMessage(data) {
  const errors = data.result?.errors || [];
  if (errors.length) {
    const detail = errors.join("；");
    if (/样本 ID|匹配/.test(detail)) return `${detail}。请确认两个 CSV 第一列的样本 ID 完全一致。`;
    if (/CUDA|GPU/.test(detail)) return `${detail}。请改用自动设备或 CPU 后重试。`;
    if (/缺少必要文件/.test(detail)) return `${detail}。请重新选择数据集或上传完整文件。`;
    return `${detail}。修改参数后可重新提交，原任务记录会保留。`;
  }
  const runtime = data.runtime || {};
  const parts = [`${statusLabels[data.status] || data.status} · 已用 ${formatDuration(runtime.execution_seconds)}`];
  if (runtime.epoch_progress) {
    parts.push(`轮次 ${runtime.epoch_progress.current_epoch}/${runtime.epoch_progress.total_epochs}`);
  }
  if (runtime.queue_seconds > 0) parts.push(`排队 ${formatDuration(runtime.queue_seconds)}`);
  if (!runtime.terminal && runtime.estimated_seconds_low != null) {
    parts.push(`预计总耗时 ${formatDuration(runtime.estimated_seconds_low)} - ${formatDuration(runtime.estimated_seconds_high)}`);
  }
  if (runtime.message) parts.push(runtime.message);
  return parts.join("。");
}

function progressPercent(data) {
  const progress = {
    created: 5,
    running: 10,
    validating: 18,
    preparing: 28,
    training_encoder: 42,
    building_relatedness: 56,
    training_menet: 72,
    evaluating: 86,
    explaining: 94,
    completed: 100,
    failed: 100,
    cancelled: 100,
  };
  let value = progress[data.status] || 10;
  const epoch = data.runtime?.epoch_progress;
  if (epoch?.phase === data.status) {
    const ranges = { training_encoder: [28, 52], training_menet: [58, 84] };
    const range = ranges[data.status];
    if (range) value = range[0] + ((range[1] - range[0]) * epoch.current_epoch) / epoch.total_epochs;
  }
  return value;
}

async function renderArtifacts(taskId, container) {
  const data = await request(`/api/v1/tasks/${encodeURIComponent(taskId)}/artifacts?${userQuery()}`);
  container.replaceChildren();
  const labels = {
    "menet_model.pt": "MENET 模型",
    "trait_specific_encoder.pt": "性状编码器",
    "metrics.json": "评估指标",
    "training_history.json": "训练历史",
    "test_predictions.csv": "测试集预测",
    "snp_importance.csv": "SNP 重要性",
    "report.html": "分析报告",
    "task.json": "任务配置",
    "progress.json": "训练进度记录",
    "quality_report.json": "质量评估报告",
  };
  data.files.forEach((file) => {
    const link = document.createElement("a");
    link.className = "artifact-link";
    link.href = `/api/v1/tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(file.name)}?${userQuery()}`;
    link.textContent = `下载${labels[file.name] ? ` ${labels[file.name]}` : ` ${file.name}`}`;
    container.appendChild(link);
  });
}

function scheduleTaskPoll(taskId, delay = 3000) {
  clearTimeout(taskPollers.get(taskId));
  taskPollers.set(taskId, setTimeout(() => loadTask(taskId), delay));
}

async function renderTask(data, scroll = false) {
  const taskId = data.task_id;
  const node = makeTaskMessage(taskId);
  const terminal = terminalStatuses.has(data.status);
  const previousStatus = knownTaskStatuses.get(taskId);
  knownTaskStatuses.set(taskId, data.status);

  node.querySelector(".task-title").textContent =
    `${intentLabels[data.task?.intent] || "MENET 分析"} · ${data.task?.trait || "未指定性状"}`;
  const badge = node.querySelector(".task-status-badge");
  badge.textContent = statusLabels[data.status] || data.status;
  badge.className = `task-status-badge ${data.status}`;
  node.querySelector(".task-runtime").textContent = runtimeMessage(data);
  node.querySelector(".cancel-inline").classList.toggle("hidden", terminal);
  node.querySelector(".task-background-note").classList.toggle("hidden", terminal);

  const bar = node.querySelector(".progress span");
  bar.style.width = `${progressPercent(data)}%`;
  bar.style.background = data.status === "failed" ? "#b65c5c" : data.status === "cancelled" ? "#8a9690" : "#86a63f";

  const result = node.querySelector(".chat-result");
  if (terminal) {
    result.classList.remove("hidden");
    result.querySelector("pre").textContent = JSON.stringify(data.result || { status: data.status }, null, 2);
    if (data.status === "completed") {
      try {
        const explanation = await request(`/api/v1/tasks/${encodeURIComponent(taskId)}/explanation?${userQuery()}`);
        result.querySelector(".result-explanation").textContent = explanation.answer;
      } catch {
        result.querySelector(".result-explanation").textContent = "分析已完成，暂无文字解释。";
      }
    } else {
      result.querySelector(".result-explanation").textContent = runtimeMessage(data);
    }
    await renderArtifacts(taskId, result.querySelector(".artifact-list"));
    clearTimeout(taskPollers.get(taskId));
    taskPollers.delete(taskId);
    if (previousStatus && !terminalStatuses.has(previousStatus)) {
      document.title = `${intentLabels[data.task?.intent] || "MENET 任务"}${data.status === "completed" ? "已完成" : "已结束"} · MENET Agent`;
      await loadConversationHistory();
      if (["train_model", "generate_report"].includes(data.task?.intent)) await loadModels();
    }
  } else {
    result.classList.add("hidden");
    scheduleTaskPoll(taskId);
  }
  if (scroll) scrollConversation();
}

async function loadTask(taskId = activeTaskId, scroll = false) {
  if (!taskId) return;
  try {
    const data = await request(`/api/v1/tasks/${encodeURIComponent(taskId)}?${userQuery()}`);
    activeTaskId = taskId;
    localStorage.setItem(taskKey(), taskId);
    await renderTask(data, scroll);
  } catch (error) {
    notice(error.message, true);
  }
}

async function checkHealth() {
  try {
    const data = await request("/health");
    $("health").innerHTML = `<span class="dot" style="background:#86a63f"></span> service online / ${data.queue}`;
  } catch {
    $("health").innerHTML = '<span class="dot"></span> service unavailable';
  }
}

function updateLlmStatus(settings) {
  const active = settings.enabled && settings.mode === "llm";
  const status = $("llm-status");
  const summary = $("llm-panel-summary");
  status.classList.toggle("active", active);
  status.querySelector(".dot").style.background = active ? "#86a63f" : "#e7ab55";
  const model = settings.model || settings.intent_model || "未选择模型";
  const label = active ? (settings.provider === "ollama" ? "Ollama" : "模型服务") + " · " + model : "规则模式";
  status.querySelector("span:last-child").textContent = label;
  if (summary) summary.textContent = label;
}

function updateLlmProviderFields() {
  const ollama = $("llm-provider").value === "ollama";
  $("llm-api-key-row").classList.toggle("hidden", ollama);
  if (ollama && !$("llm-base-url").value.trim()) $("llm-base-url").value = "http://127.0.0.1:11434";
}

async function loadLlmModels(selectedIntent = "", selectedAnalysis = "") {
  const provider = $("llm-provider").value;
  const baseUrl = $("llm-base-url").value.trim();
  const intentSelect = $("llm-model");
  if (!intentSelect) return;
  const wantedIntent = selectedIntent || intentSelect.value;
  intentSelect.replaceChildren(new Option("正在读取模型...", ""));
  try {
    const query = new URLSearchParams({ provider, base_url: baseUrl });
    const data = await request("/api/v1/settings/llm/models?" + query);
    intentSelect.replaceChildren(new Option("请选择模型", ""));
    data.models.forEach((model) => intentSelect.appendChild(new Option(model, model)));
    if (wantedIntent && !data.models.includes(wantedIntent)) intentSelect.appendChild(new Option(wantedIntent, wantedIntent));
    intentSelect.value = wantedIntent || data.models[0] || "";
    if (!data.models.length) showLlmResult("模型服务可以访问，但没有发现可用模型。", true);
  } catch (error) {
    intentSelect.replaceChildren(new Option(wantedIntent || "请先测试连接", wantedIntent));
    showLlmResult(error.message, true);
  }
}

async function loadLlmSettings() {
  const settings = await request("/api/v1/settings/llm");
  $("llm-enabled").checked = settings.enabled;
  $("llm-provider").value = settings.provider;
  $("llm-base-url").value = settings.base_url;
  $("llm-api-key").value = "";
  $("llm-api-key").placeholder = settings.api_key_masked || "请输入 API Key";
  updateLlmProviderFields();
  await loadLlmModels(settings.model || settings.intent_model, settings.analysis_model);
  updateLlmStatus(settings);
}

async function saveLlmSettings() {
  const payload = {
    enabled: $("llm-enabled").checked,
    provider: $("llm-provider").value,
    base_url: $("llm-base-url").value.trim(),
    model: $("llm-model").value,
    intent_model: $("llm-model").value,
    analysis_model: $("llm-model").value,
    api_key: $("llm-api-key").value.trim() || null,
  };
  const settings = await request("/api/v1/settings/llm", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  $("llm-api-key").value = "";
  $("llm-api-key").placeholder = settings.api_key_masked || "请输入 API Key";
  updateLlmStatus(settings);
  return settings;
}

async function loadUsers() {
  const data = await request("/api/v1/users");
  if (!data.users.some((user) => user.user_id === currentUserId)) currentUserId = "user_local";
  const select = $("user-select");
  select.replaceChildren();
  data.users.forEach((user) => select.appendChild(new Option(user.display_name, user.user_id)));
  select.value = currentUserId;
  localStorage.setItem("menet_user_id", currentUserId);
}

async function loadDatasets(selectedId = "") {
  const data = await request(`/api/v1/datasets?${userQuery()}`);
  const select = $("dataset-select");
  select.replaceChildren(new Option("请选择或上传数据", ""));
  data.datasets.forEach((dataset) => {
    const demo = dataset.is_demo ? "示例 · " : "";
    const scale = dataset.sample_count && dataset.snp_count ? ` · ${dataset.sample_count} 样本 / ${dataset.snp_count} SNP` : "";
    const option = new Option(`${demo}${dataset.name || dataset.trait}${scale}`, dataset.dataset_id);
    option.dataset.directory = dataset.dataset_dir;
    option.dataset.trait = dataset.trait;
    option.dataset.name = dataset.name;
    option.dataset.species = dataset.species;
    option.dataset.samples = dataset.sample_count || "";
    option.dataset.snps = dataset.snp_count || "";
    option.dataset.demo = dataset.is_demo ? "1" : "";
    select.appendChild(option);
  });
  const defaultId = selectedId || data.datasets.find((dataset) => dataset.dataset_id === "demo_rice_plant_height")?.dataset_id;
  if (defaultId) {
    select.value = defaultId;
    applySelectedDataset();
  }
}

function formatMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "暂无";
}

function applySelectedModel() {
  const select = $("trained-model-select");
  const option = select.selectedOptions[0];
  selectedModelId = option?.value || "";
  localStorage.setItem(modelKey(), selectedModelId);
  $("rename-model").disabled = !selectedModelId;
  $("archive-model").disabled = !selectedModelId;
  $("repeat-model").disabled = !selectedModelId;
  if (!selectedModelId) {
    $("trained-model-summary").textContent = "完成一次训练后，模型会自动登记在这里。";
    $("trained-model-summary").classList.add("empty");
    if (!$("model-panel-summary").textContent) $("model-panel-summary").textContent = "0 个模型";
    return;
  }
  const metrics = JSON.parse(option.dataset.metrics || "{}");
  const qualityLabels = { pass: "通过", warning: "需关注", fail: "未通过", unknown: "待评估" };
  $("trained-model-summary").textContent =
    `${option.dataset.trait} · ${option.dataset.samples || "?"} 个样本 · ${option.dataset.snps || "?"} 个 SNP · 测试 R² ${formatMetric(metrics.test_r2)} · 质量${qualityLabels[option.dataset.quality] || "待评估"}`;
  $("trained-model-summary").classList.remove("empty");
  $("model-panel-summary").textContent = option.dataset.name || "已选择模型";
  loadModelComparison(option.dataset.trait, option.dataset.datasetId || "");
}

async function loadModelComparison(trait = "", datasetId = "") {
  const target = $("model-comparison");
  const params = new URLSearchParams({ user_id: currentUserId });
  if (trait) params.set("trait", trait);
  if (datasetId) params.set("dataset_id", datasetId);
  try {
    const data = await request(`/api/v1/models/compare?${params}`);
    const summary = data.summary;
    if (!summary.run_count) {
      target.textContent = "尚无可比较的测试指标。";
    } else if (!summary.stable_enough_to_estimate) {
      target.textContent = `当前只有 1 次训练（测试 R² ${summary.test_r2_mean.toFixed(3)}），再训练一次后可估计稳定性。`;
    } else {
      const stability = { stable: "稳定", variable: "存在波动", unstable: "稳定性需关注" }[summary.stability] || "待判断";
      target.textContent = `${summary.run_count} 次训练 · 平均测试 R² ${summary.test_r2_mean.toFixed(3)} · 标准差 ${summary.test_r2_std.toFixed(3)} · 范围 ${summary.test_r2_min.toFixed(3)}–${summary.test_r2_max.toFixed(3)} · ${stability}`;
    }
    target.classList.toggle("empty", !summary.stable_enough_to_estimate);
  } catch (error) {
    target.textContent = error.message;
    target.classList.add("empty");
  }
}

async function loadModels(preferredId = "") {
  const data = await request(`/api/v1/models?${userQuery()}`);
  const select = $("trained-model-select");
  const wanted = preferredId || selectedModelId || localStorage.getItem(modelKey()) || "";
  select.replaceChildren(new Option(data.models.length ? "自动选择最近兼容模型" : "尚无训练模型", ""));
  data.models.forEach((model) => {
    const option = new Option(`${model.name} · ${model.trait}`, model.model_id);
    option.dataset.trait = model.trait;
    option.dataset.samples = model.sample_count || "";
    option.dataset.snps = model.snp_count || "";
    option.dataset.metrics = JSON.stringify(model.metrics || {});
    option.dataset.quality = model.quality_report?.overall_status || "unknown";
    option.dataset.name = model.name;
    option.dataset.datasetId = model.dataset_id || "";
    select.appendChild(option);
  });
  $("model-panel-summary").textContent = `${data.models.length} 个模型`;
  select.value = data.models.some((model) => model.model_id === wanted) ? wanted : (data.models[0]?.model_id || "");
  applySelectedModel();
}

function applySelectedDataset() {
  const option = $("dataset-select").selectedOptions[0];
  if (!option?.value) {
    $("dataset-summary").textContent = "选择数据后将在这里显示物种、性状和规模。";
    $("dataset-summary").classList.add("empty");
    $("dataset-panel-summary").textContent = "未选择";
    $("dataset-dir").value = "data";
    $("dataset-inspection-result").textContent = "请先选择数据集。";
    return;
  }
  $("trait").value = option.dataset.trait;
  $("dataset-dir").value = option.dataset.directory;
  $("dataset-name").value = option.dataset.name || "";
  $("species").value = option.dataset.species || "";
  const scale = option.dataset.samples ? `${option.dataset.samples} 个样本 · ${option.dataset.snps} 个 SNP` : "规模将在首次检查后补充";
  $("dataset-summary").textContent =
    `${option.dataset.demo ? "共享示例" : "私人数据"} · ${option.dataset.species || "未填写物种"} · ${option.dataset.trait} · ${scale}`;
  $("dataset-summary").classList.remove("empty");
  $("dataset-panel-summary").textContent = `${option.dataset.species || "未填写物种"} · ${option.dataset.trait}`;
  $("dataset-inspection-result").textContent = "展开后执行只读质量检查。";
  $("dataset-inspection-result").classList.add("empty");
}

async function loadDatasetInspection() {
  const datasetId = $("dataset-select").value;
  const target = $("dataset-inspection-result");
  if (!datasetId) return;
  target.textContent = "正在检查数据...";
  try {
    const response = await request(`/api/v1/datasets/${encodeURIComponent(datasetId)}/inspection?${userQuery()}`);
    const result = response.inspection;
    const data = result.data || {};
    const lines = [
      `${result.success ? "可用于 MENET" : "暂不可用于 MENET"} · 匹配 ${data.matched_sample_count ?? "?"} 个样本 · ${data.snp_count ?? "?"} 个 SNP`,
      `基因型缺失 ${data.genotype_missing_count ?? "?"}（${((data.genotype_missing_rate || 0) * 100).toFixed(2)}%） · 重复材料 ${data.duplicate_genotype_sample_count ?? "?"}`,
    ];
    if (result.errors?.length) lines.push(`错误：${result.errors.join("；")}`);
    if (result.warnings?.length) lines.push(`提醒：${result.warnings.join("；")}`);
    if (!result.errors?.length && !result.warnings?.length) lines.push("未发现格式或质量提醒。");
    target.textContent = lines.join("\n");
    target.classList.toggle("empty", !result.success);
  } catch (error) {
    target.textContent = `检查失败：${error.message}`;
    target.classList.add("empty");
  }
}

async function restoreConversation(conversation) {
  const response = await request(`/api/v1/tasks?${userQuery()}`);
  const activeFromState = conversation.state?.active_task_id;
  const tasks = response.tasks.filter((task) =>
    task.task?.metadata?.conversation_id === conversationId || task.task_id === activeFromState
  );
  const events = [
    ...(conversation.messages || [])
      .filter((message) => !/^任务已创建：task_/.test(message.content))
      .map((message) => ({ kind: "message", at: message.created_at, value: message })),
    ...tasks.map((task) => ({ kind: "task", at: task.created_at, value: task })),
  ].sort((a, b) => new Date(a.at) - new Date(b.at));

  for (const event of events) {
    if (event.kind === "message") addMessage(event.value.content, event.value.role, false);
    else {
      makeTaskMessage(event.value.task_id);
      await renderTask(event.value, false);
    }
  }
  scrollConversation("auto");
}

async function loadConversationHistory() {
  const container = $("conversation-history");
  if (!container) return;
  try {
    const data = await request(`/api/v1/conversations?${userQuery()}`);
    container.replaceChildren();
    if (!data.conversations.length) {
      container.textContent = "暂无历史会话";
      container.className = "task-history empty-history";
      $("history-panel-summary").textContent = "0 个会话";
      return;
    }
    container.className = "task-history";
    $("history-panel-summary").textContent = `${data.conversations.length} 个会话`;
    data.conversations.forEach((conversation) => {
      const button = document.createElement("button");
      button.className = `history-row${conversation.conversation_id === conversationId ? " active" : ""}`;
      const title = document.createElement("strong");
      title.textContent = conversation.title;
      const meta = document.createElement("span");
      meta.textContent = new Date(conversation.updated_at).toLocaleString();
      button.append(title, meta);
      button.addEventListener("click", () => {
        if (conversation.conversation_id === conversationId) return;
        localStorage.setItem(conversationKey(), conversation.conversation_id);
        localStorage.removeItem(taskKey());
        location.reload();
      });
      container.appendChild(button);
    });
  } catch (error) {
    container.textContent = `任务历史加载失败：${error.message}`;
  }
}

async function initialize() {
  await loadLlmSettings();
  await loadUsers();
  conversationId = localStorage.getItem(conversationKey());
  activeTaskId = localStorage.getItem(taskKey()) || "";
  let conversation;
  if (conversationId) {
    try {
      conversation = await request(`/api/v1/conversations/${encodeURIComponent(conversationId)}?${userQuery()}`);
    } catch {
      conversationId = null;
    }
  }
  if (!conversationId) {
    conversation = await request("/api/v1/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentUserId }),
    });
    conversationId = conversation.conversation_id;
    localStorage.setItem(conversationKey(), conversationId);
  }
  const state = conversation.state || {};
  if (state.active_trait) $("trait").value = state.active_trait;
  if (state.active_dataset_dir) $("dataset-dir").value = state.active_dataset_dir;
  if (state.active_output_base_dir) $("output-dir").value = state.active_output_base_dir;
  if (state.active_task_id) {
    activeTaskId = state.active_task_id;
    localStorage.setItem(taskKey(), activeTaskId);
  }
  await loadDatasets(state.active_dataset_id);
  await loadModels(state.active_model_id);
  await restoreConversation(conversation);
  await loadConversationHistory();
}

const appReady = initialize().catch((error) => notice(error.message, true));

$("llm-status").addEventListener("click", () => {
  $("llm-panel").open = true;
  $("llm-panel").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("llm-provider").addEventListener("change", () => {
  updateLlmProviderFields();
  loadLlmModels();
});
$("llm-refresh-models").addEventListener("click", () => loadLlmModels());
$("llm-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const settings = await saveLlmSettings();
    showLlmResult(settings.mode === "llm" ? "已启用 " + settings.model : "设置已保存，当前使用规则模式。");
  } catch (error) {
    showLlmResult(error.message, true);
  }
});
$("llm-test").addEventListener("click", async () => {
  try {
    await saveLlmSettings();
    showLlmResult("正在测试模型连接...");
    const result = await request("/api/v1/settings/llm/test", { method: "POST" });
    const summary = result.models.map((item) => item.model + " " + item.latency_seconds + " 秒").join("；");
    showLlmResult("连接成功：" + result.provider + "；" + summary + "。");
  } catch (error) {
    showLlmResult(error.message, true);
  }
});

$("user-select").addEventListener("change", (event) => {
  localStorage.setItem("menet_user_id", event.target.value);
  location.reload();
});

$("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("prediction-genotype").files[0];
  if (!file) return;
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await request(`/api/v1/prediction-genotypes/upload?${userQuery()}`, { method: "POST", body: form });
    predictionGenotypePath = data.path;
    $("prediction-result").textContent = "已设置：" + data.filename;
    $("prediction-result").classList.remove("hidden");
    notice("新基因型文件已设置，点击“预测性状”后发送即可。", false);
  } catch (error) {
    notice(error.message, true);
  }
});

$("new-conversation").addEventListener("click", async () => {
  try {
    const conversation = await request("/api/v1/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentUserId }),
    });
    localStorage.setItem(conversationKey(), conversation.conversation_id);
    localStorage.removeItem(taskKey());
    location.reload();
  } catch (error) {
    notice(error.message, true);
  }
});

$("add-user").addEventListener("click", async () => {
  const displayName = window.prompt("请输入本地用户名称");
  if (!displayName?.trim()) return;
  try {
    const user = await request("/api/v1/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName.trim() }),
    });
    localStorage.setItem("menet_user_id", user.user_id);
    location.reload();
  } catch (error) {
    notice(error.message, true);
  }
});

$("dataset-select").addEventListener("change", applySelectedDataset);
$("trained-model-select").addEventListener("change", applySelectedModel);
$("dataset-inspection").addEventListener("toggle", (event) => {
  if (event.currentTarget.open) loadDatasetInspection();
});

$("repeat-model").addEventListener("click", async () => {
  if (!selectedModelId) return;
  const raw = window.prompt("重复训练次数（2–5 次）", "3");
  if (raw === null) return;
  const repeats = Number.parseInt(raw, 10);
  if (!Number.isInteger(repeats) || repeats < 2 || repeats > 5) {
    notice("重复次数必须是 2 到 5 的整数。", true);
    return;
  }
  if (!window.confirm(`将提交 ${repeats} 次独立训练，并使用不同的数据划分与训练种子。任务会依次占用 GPU，确定继续吗？`)) return;
  try {
    const response = await request(`/api/v1/models/${encodeURIComponent(selectedModelId)}/repeat-training`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentUserId, conversation_id: conversationId, repeats }),
    });
    for (const item of response.tasks) {
      makeTaskMessage(item.task_id);
      loadTask(item.task_id);
    }
    activeTaskId = response.tasks.at(-1).task_id;
    localStorage.setItem(taskKey(), activeTaskId);
    notice(`已提交 ${repeats} 次重复训练。完成后“同类训练对比”会自动汇总稳定性。`);
    scrollConversation();
  } catch (error) {
    notice(error.message, true);
  }
});

$("rename-model").addEventListener("click", async () => {
  if (!selectedModelId) return;
  const current = $("trained-model-select").selectedOptions[0]?.dataset.name || "";
  const name = window.prompt("请输入新的模型名称", current);
  if (!name?.trim()) return;
  try {
    await request(`/api/v1/models/${encodeURIComponent(selectedModelId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentUserId, name: name.trim() }),
    });
    await loadModels(selectedModelId);
    notice("模型名称已更新。");
  } catch (error) { notice(error.message, true); }
});

$("archive-model").addEventListener("click", async () => {
  if (!selectedModelId || !window.confirm("将模型移出可选列表？训练文件仍会保留。")) return;
  try {
    await request(`/api/v1/models/${encodeURIComponent(selectedModelId)}?${userQuery()}`, { method: "DELETE" });
    selectedModelId = "";
    localStorage.removeItem(modelKey());
    await loadModels();
    notice("模型已移出可选列表。");
  } catch (error) { notice(error.message, true); }
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearNotice();
  await appReady;
  const form = new FormData();
  form.append("genotype", $("genotype").files[0]);
  form.append("phenotype", $("phenotype").files[0]);
  const query = new URLSearchParams({
    trait: $("trait").value,
    name: $("dataset-name").value,
    species: $("species").value,
    conversation_id: conversationId,
    user_id: currentUserId,
  });
  try {
    const data = await request(`/api/v1/datasets/upload?${query}`, { method: "POST", body: form });
    $("dataset-dir").value = data.dataset_dir;
    $("dataset-result").textContent = `数据已登记：${data.name}`;
    $("dataset-result").classList.remove("hidden");
    await loadDatasets(data.dataset_id);
    notice("数据上传完成，并已设为当前数据。");
  } catch (error) {
    notice(error.message, true);
  }
});

document.querySelectorAll("[data-template]").forEach((button) => button.addEventListener("click", () => {
  const trait = $("trait").value.trim() || "plant_height";
  $("message").value = button.dataset.template.replaceAll("{trait}", trait);
  $("message").focus();
}));

function confirmLongTask(text) {
  if (!/(训练|报告|train|report)/i.test(text)) return Promise.resolve(true);
  const selected = $("dataset-select").selectedOptions[0];
  $("confirm-summary").textContent = `${selected?.textContent || "当前数据"}；性状 ${$("trait").value || "未指定"}。`;
  const dialog = $("task-confirm");
  dialog.showModal();
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (dialog.open) dialog.close();
      resolve(value);
    };
    $("cancel-run").onclick = () => finish(false);
    $("confirm-run").onclick = () => finish(true);
    dialog.oncancel = (event) => {
      event.preventDefault();
      finish(false);
    };
  });
}

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearNotice();
  await appReady;
  const text = $("message").value.trim();
  if (!text) return;
  if (!$("dataset-select").value) {
    notice("请先选择一套示例数据，或展开“上传新数据”完成上传。", true);
    $("upload-details").open = true;
    return;
  }
  if (!(await confirmLongTask(text))) return;
  addMessage(text, "user");
  $("message").value = "";
  try {
    const data = await request("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        user_id: currentUserId,
        dataset_id: $("dataset-select").value,
        conversation_id: conversationId,
        prediction_genotype_path: predictionGenotypePath || null,
        model_id: selectedModelId || null,
      }),
    });
    if (data.task_id) {
      activeTaskId = data.task_id;
      localStorage.setItem(taskKey(), activeTaskId);
      makeTaskMessage(activeTaskId);
      await loadTask(activeTaskId, true);
      await loadConversationHistory();
    } else if (data.type === "needs_input") {
      addMessage(`请补充：${(data.parsed?.missing_fields || []).join("、")}`, "agent");
    } else if (data.type === "needs_model") {
      addMessage(data.message, "agent");
    } else {
      addMessage(`任务参数无效：${(data.errors || []).join("；")}`, "agent");
    }
  } catch (error) {
    addMessage(`请求没有成功：${error.message}`, "agent");
    notice(error.message, true);
  }
});

$("message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});

$("refresh-history").addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  loadConversationHistory();
});
checkHealth();
