"use strict";

// 报告页：通过 QWebChannel 向 Python（ReportBridge）读取课程记录，图表用本地 ECharts 绘制

const STATUS_LABELS = { recording: "記録中", completed: "終了", interrupted: "中断" };
const FONT = getComputedStyle(document.documentElement).getPropertyValue("--font");
const ACCENT = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();

const state = { bridge: null, sessions: [], selectedId: null, chart: null, refreshSeq: 0 };

const $ = (id) => document.getElementById(id);

function call(method, ...args) {
  return new Promise((resolve) => {
    state.bridge[method](...args, (json) => resolve(JSON.parse(json)));
  });
}

function formatDate(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString("ja-JP", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h > 0 ? `${h}時間${m}分` : `${m}分${s}秒`;
}

function sessionLabel(s) {
  const name = s.course_name || "授業名なし";
  const status = s.status === "completed" ? "" : `（${STATUS_LABELS[s.status] || s.status}）`;
  return `${formatDate(s.started_at)}　${name}${status}`;
}

function showMessage(pageState, text) {
  document.body.dataset.state = pageState;
  $("message").textContent = text;
  $("message").hidden = !text;
}

function renderSessionList() {
  const select = $("session-select");
  select.replaceChildren(...state.sessions.map((s) => new Option(sessionLabel(s), String(s.id))));
  select.value = String(state.selectedId);
  $("session-count").textContent = String(state.sessions.length);
}

function renderSession(record) {
  const s = record.session;
  $("stat-date").textContent = formatDate(s.started_at);
  $("stat-duration").textContent = formatDuration(s.seconds);
  $("stat-score").textContent = s.avg_score === null ? "--" : String(Math.round(s.avg_score));
  $("stat-status").textContent = STATUS_LABELS[s.status] || s.status;

  if (!state.chart) {
    state.chart = echarts.init($("score-chart"));
  }
  state.chart.setOption({
    textStyle: { fontFamily: FONT },
    grid: { left: 48, right: 24, top: 16, bottom: 64 },
    tooltip: {
      trigger: "axis",
      valueFormatter: (v) => String(Math.round(v)),
    },
    xAxis: { type: "value", name: "分", min: 0, max: "dataMax" },
    yAxis: { type: "value", min: 0, max: 100 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 12 }],
    series: [{
      name: "エンゲージメント",
      type: "line",
      showSymbol: false,
      sampling: "lttb",
      lineStyle: { color: ACCENT, width: 1.5 },
      itemStyle: { color: ACCENT },
      data: record.samples.map((x) => [x.t / 60, x.score]),
    }],
  }, true);
}

async function loadSelected(seq) {
  const res = await call("load_session", state.selectedId);
  if (seq !== state.refreshSeq) return;
  if (res.error || !res.data) {
    showMessage("error", "データを読み込めませんでした。");
    return;
  }
  showMessage("ready", "");  // 先显示区域再画图，否则 ECharts 在隐藏元素上初始化尺寸为 0
  renderSession(res.data);
}

async function refresh() {
  const seq = ++state.refreshSeq;
  const res = await call("list_sessions");
  if (seq !== state.refreshSeq) return;
  if (res.error) {
    showMessage("error", "データを読み込めませんでした。");
    return;
  }
  const knownIds = new Set(state.sessions.map((s) => s.id));
  state.sessions = res.data;
  if (state.sessions.length === 0) {
    state.selectedId = null;
    renderSessionList();
    showMessage("empty", "まだ記録がありません。モニター画面で「開始」を押すと、授業の記録が始まります。");
    return;
  }
  // 有新记录（如刚结束的课）时显示最新的一条；否则保持当前选择，所选记录已不存在时也退回最新的
  const hasNew = state.sessions.some((s) => !knownIds.has(s.id));
  if (hasNew || !state.sessions.some((s) => s.id === state.selectedId)) {
    state.selectedId = state.sessions[0].id;
  }
  renderSessionList();
  await loadSelected(seq);
}

function selectTab(view) {
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === view));
  }
  $("view-single").hidden = view !== "single";
  $("view-summary").hidden = view !== "summary";
  if (view === "single" && state.chart) state.chart.resize();
}

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.addEventListener("click", () => selectTab(tab.dataset.view));
}

$("session-select").addEventListener("change", (e) => {
  state.selectedId = Number(e.target.value);
  loadSelected(++state.refreshSeq);
});

window.addEventListener("resize", () => state.chart && state.chart.resize());

new QWebChannel(qt.webChannelTransport, (channel) => {
  state.bridge = channel.objects.bridge;
  state.bridge.data_changed_signal.connect(refresh);
  refresh();
});
