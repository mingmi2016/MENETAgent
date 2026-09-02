const $ = (id) => document.getElementById(id);
let currentUserId = localStorage.getItem("menet_user_id") || "user_local";
let activeTaskId = "";
let conversationId = null;
let pollTimer = null;

const userQuery = () => `user_id=${encodeURIComponent(currentUserId)}`;
const conversationKey = () => `menet_conversation_id:${currentUserId}`;
const taskKey = () => `menet_task_id:${currentUserId}`;

function notice(message, error = false) {
  const element = $("notice");
  element.textContent = message;
  element.className = `notice${error ? " error" : ""}`;
}

function clearNotice() { $("notice").className = "notice hidden"; }

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.join("；") : data.detail;
    throw new Error(detail || "请求失败");
  }
  return data;
}

function addMessage(text, role) {
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
  item.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function checkHealth() {
  try {
    const data = await request("/health");
    $("health").innerHTML = `<span class="dot" style="background:#86a63f"></span> service online / ${data.queue}`;
  } catch {
    $("health").innerHTML = '<span class="dot"></span> service unavailable';
  }
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

async function initialize() {
  await loadUsers();
  conversationId = localStorage.getItem(conversationKey());
  activeTaskId = localStorage.getItem(taskKey()) || "";
  let conversation;
  if (conversationId) {
    try { conversation = await request(`/api/v1/conversations/${encodeURIComponent(conversationId)}?${userQuery()}`); }
    catch { conversationId = null; }
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
  (conversation.messages || []).forEach((message) => addMessage(message.content, message.role));
  if (state.active_task_id) {
    activeTaskId = state.active_task_id;
    localStorage.setItem(taskKey(), activeTaskId);
  }
  await loadDatasets(state.active_dataset_id);
  await loadTaskHistory();
  if (activeTaskId) {
    $("lookup-id").value = activeTaskId;
    await loadTask();
  }
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

function applySelectedDataset() {
  const option = $("dataset-select").selectedOptions[0];
  if (!option?.value) {
    $("dataset-summary").textContent = "选择数据后将在这里显示物种、性状和规模。";
    $("dataset-summary").classList.add("empty");
    return;
  }
  $("trait").value = option.dataset.trait;
  $("dataset-dir").value = option.dataset.directory;
  $("dataset-name").value = option.dataset.name || "";
  $("species").value = option.dataset.species || "";
  const scale = option.dataset.samples ? `${option.dataset.samples} 个样本 · ${option.dataset.snps} 个 SNP` : "规模将在首次检查后补充";
  $("dataset-summary").textContent = `${option.dataset.demo ? "共享示例" : "私人数据"} · ${option.dataset.species || "未填写物种"} · ${option.dataset.trait} · ${scale}`;
  $("dataset-summary").classList.remove("empty");
}

const appReady = initialize().catch((error) => notice(error.message, true));

$("user-select").addEventListener("change", (event) => {
  localStorage.setItem("menet_user_id", event.target.value);
  location.reload();
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
  } catch (error) { notice(error.message, true); }
});

$("dataset-select").addEventListener("change", applySelectedDataset);

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
    $("dataset-result").textContent = `数据已登记：${data.name}（${data.dataset_id}）`;
    $("dataset-result").classList.remove("hidden");
    await loadDatasets(data.dataset_id);
    notice("数据上传完成，并已设为当前会话的数据集。");
  } catch (error) { notice(error.message, true); }
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
    dialog.oncancel = (event) => { event.preventDefault(); finish(false); };
  });
}

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearNotice();
  await appReady;
  const text = $("message").value.trim();
  if (!text) return;
  if (!$("dataset-select").value && $("dataset-dir").value.trim() === "data") {
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
        dataset_id: $("dataset-select").value || null,
        dataset_dir: $("dataset-dir").value,
        output_dir: $("output-dir").value,
        conversation_id: conversationId,
      }),
    });
    if (data.task_id) {
      addMessage(`任务已创建：${data.task_id}。结果将独立保存到 ${data.output_dir}。`, "agent");
      activeTaskId = data.task_id;
      localStorage.setItem(taskKey(), activeTaskId);
      $("lookup-id").value = activeTaskId;
      await loadTaskHistory();
      await loadTask();
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

async function renderArtifacts() {
  const data = await request(`/api/v1/tasks/${encodeURIComponent(activeTaskId)}/artifacts?${userQuery()}`);
  const container = $("artifacts");
  container.replaceChildren();
  data.files.forEach((file) => {
    const link = document.createElement("a");
    link.className = "artifact-link";
    link.href = `/api/v1/tasks/${encodeURIComponent(activeTaskId)}/artifacts/${encodeURIComponent(file.name)}?${userQuery()}`;
    const labels = { "menet_model.pt": "MENET 模型", "trait_specific_encoder.pt": "性状编码器", "metrics.json": "评估指标", "training_history.json": "训练历史", "test_predictions.csv": "测试集预测", "snp_importance.csv": "SNP 重要性", "report.html": "分析报告", "task.json": "任务配置", "progress.json": "训练进度记录" };
    link.textContent = `下载${labels[file.name] ? ` ${labels[file.name]}` : ` ${file.name}`}`;
    container.appendChild(link);
  });
}

async function loadTaskHistory() {
  const container = $("task-history");
  try {
    const data = await request(`/api/v1/tasks?${userQuery()}`);
    container.replaceChildren();
    if (!data.tasks.length) {
      container.textContent = "暂无任务记录";
      container.className = "task-history empty-history";
      return;
    }
    container.className = "task-history";
    const intentLabels = { inspect_data: "数据检查", train_model: "模型训练", predict_trait: "性状预测", evaluate_model: "模型评估", explain_model: "模型解释", generate_report: "完整报告" };
    const statusLabels = { created: "排队中", running: "启动中", validating: "检查数据", preparing: "准备数据", training_encoder: "训练编码器", building_relatedness: "构建相关性", training_menet: "训练 MENET", evaluating: "评估中", explaining: "解释中", completed: "已完成", failed: "失败", cancelled: "已取消" };
    data.tasks.slice(0, 8).forEach((task) => {
      const button = document.createElement("button");
      button.className = `history-row ${task.status}`;
      const title = document.createElement("strong");
      title.textContent = `${intentLabels[task.task.intent] || task.task.intent} · ${task.task.trait}`;
      const meta = document.createElement("span");
      meta.textContent = `${statusLabels[task.status] || task.status} · ${new Date(task.created_at).toLocaleString()}`;
      button.append(title, meta);
      button.addEventListener("click", async () => {
        activeTaskId = task.task_id;
        localStorage.setItem(taskKey(), activeTaskId);
        $("lookup-id").value = activeTaskId;
        await loadTask();
      });
      container.appendChild(button);
    });
  } catch (error) {
    container.textContent = `任务历史加载失败：${error.message}`;
  }
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
    if (/样本 ID|匹配/.test(detail)) return `${detail}。请确认基因型和表型 CSV 第一列的样本 ID 完全一致。`;
    if (/CUDA|GPU/.test(detail)) return `${detail}。请改用自动设备或 CPU 后重试。`;
    if (/缺少必要文件/.test(detail)) return `${detail}。请重新选择数据集或上传完整文件。`;
    return `${detail}。可修改参数后重新提交，原任务记录会保留。`;
  }
  const labels = {
    created: "等待执行", running: "开始执行", validating: "检查数据",
    preparing: "准备数据", training_encoder: "训练性状编码器",
    building_relatedness: "生成遗传相关性", training_menet: "训练 MENET",
    evaluating: "评估模型", explaining: "生成解释",
    completed: "分析完成", failed: "执行失败", cancelled: "已取消",
  };
  const runtime = data.runtime || {};
  const parts = [`${labels[data.status] || data.status} · 执行 ${formatDuration(runtime.execution_seconds)}`];
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

async function loadTask() {
  if (!activeTaskId) return;
  clearTimeout(pollTimer);
  try {
    const data = await request(`/api/v1/tasks/${encodeURIComponent(activeTaskId)}?${userQuery()}`);
    $("task-empty").classList.add("hidden");
    $("task-view").classList.remove("hidden");
    $("task-id").textContent = activeTaskId;
    const statusLabels = { created: "排队中", running: "启动中", validating: "检查数据", preparing: "准备数据", training_encoder: "训练编码器", building_relatedness: "构建相关性", training_menet: "训练 MENET", evaluating: "评估中", explaining: "解释中", completed: "已完成", failed: "失败", cancelled: "已取消" };
    $("task-status").textContent = statusLabels[data.status] || data.status || "未知";
    $("task-trait").textContent = data.task?.trait || "-";
    const terminal = ["completed", "failed", "cancelled"].includes(data.status);
    $("cancel-task").classList.toggle("hidden", terminal);
    const progress = { created: 5, running: 10, validating: 18, preparing: 28, training_encoder: 42, building_relatedness: 56, training_menet: 72, evaluating: 86, explaining: 94, completed: 100, failed: 100, cancelled: 100 };
    let progressPercent = progress[data.status] || 10;
    const epoch = data.runtime?.epoch_progress;
    if (epoch?.phase === data.status) {
      const ranges = { training_encoder: [28, 52], training_menet: [58, 84] };
      const range = ranges[data.status];
      if (range) progressPercent = range[0] + (range[1] - range[0]) * epoch.current_epoch / epoch.total_epochs;
    }
    $("progress-bar").style.width = `${progressPercent}%`;
    $("progress-bar").style.background = data.status === "failed" ? "#b65c5c" : data.status === "cancelled" ? "#8a9690" : "#86a63f";
    $("task-message").textContent = runtimeMessage(data);
    if (terminal && data.result) {
      $("result-view").classList.remove("hidden");
      $("result-json").textContent = JSON.stringify(data.result, null, 2);
      if (data.status === "completed") {
        try {
          const explanation = await request(`/api/v1/tasks/${encodeURIComponent(activeTaskId)}/explanation?${userQuery()}`);
          $("explanation").textContent = explanation.answer;
        } catch { $("explanation").textContent = "暂无解释结果。"; }
      }
      await renderArtifacts();
      await loadTaskHistory();
    }
    if (!terminal) pollTimer = setTimeout(loadTask, 5000);
  } catch (error) { notice(error.message, true); }
}

$("lookup").addEventListener("click", () => {
  activeTaskId = $("lookup-id").value.trim();
  localStorage.setItem(taskKey(), activeTaskId);
  loadTask();
});
$("refresh").addEventListener("click", loadTask);
$("refresh-history").addEventListener("click", loadTaskHistory);
$("cancel-task").addEventListener("click", async () => {
  if (!activeTaskId || !window.confirm("确定停止当前任务吗？已经完成的训练轮次不会作为完成模型发布。")) return;
  try {
    const data = await request(`/api/v1/tasks/${encodeURIComponent(activeTaskId)}/cancel?${userQuery()}`, { method: "POST" });
    notice(data.message);
    $("cancel-task").classList.add("hidden");
    setTimeout(loadTask, 1000);
  } catch (error) { notice(error.message, true); }
});
checkHealth();
