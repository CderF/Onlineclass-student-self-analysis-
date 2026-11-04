"use strict";

// 报告页：通过 QWebChannel 向 Python（ReportBridge）读取课程记录，图表用本地 ECharts 绘制

const STATUS_LABELS = { recording: "記録中", completed: "終了", interrupted: "中断" };
const FONT = getComputedStyle(document.documentElement).getPropertyValue("--font");

const state = {
  bridge: null, sessions: [], selectedId: null, chart: null, refreshSeq: 0, view: "single",
  summary: null, summaryShown: [], summaryChart: null, summarySeq: 0, courseFilter: "all",
};

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
  renderRecordTable();
}

// ---------- 記録の管理（#10） ----------

const LOCKED_TITLE = "記録中の授業は変更・削除・書き出しできません";
const ACTIONS = [
  { key: "export", label: "CSV", slot: "export_session", className: "row-action" },
  { key: "rename", label: "名前を変更", slot: "rename_session", className: "row-action" },
  { key: "delete", label: "削除", slot: "delete_session", className: "row-action danger" },
];

function renderRecordTable() {
  $("export-all").disabled = !state.sessions.some((s) => s.status !== "recording");
  $("record-table").tBodies[0].replaceChildren(...state.sessions.map((s) => {
    const cells = [formatDate(s.started_at), s.course_name || "授業名なし", formatDuration(s.seconds),
      STATUS_LABELS[s.status] || s.status].map((text) => Object.assign(document.createElement("td"), { textContent: text }));
    if (!s.course_name) cells[1].className = "no-name";
    const locked = s.status === "recording";
    const actions = document.createElement("td");
    actions.append(...ACTIONS.map((a) => {
      const btn = Object.assign(document.createElement("button"),
        { type: "button", className: a.className, textContent: a.label, disabled: locked });
      btn.dataset.action = a.key;
      if (locked) btn.title = LOCKED_TITLE;
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        a.key === "export" ? exportCsv(a.slot, s.id) : manage(a.slot, s.id);
      });
      return btn;
    }));
    const tr = document.createElement("tr");
    tr.dataset.id = String(s.id);
    tr.setAttribute("aria-selected", String(s.id === state.selectedId));
    tr.replaceChildren(...cells, actions);
    tr.addEventListener("click", () => openSession(s.id));
    return tr;
  }));
}

function notify(tone, text) {
  $("manage-message").hidden = !text;
  $("manage-message").textContent = text;
  $("manage-message").dataset.tone = tone;
}

// 改名 / 删除：确认对话框由 Python 弹出（Qt），返回后重新读取列表；body[data-manage] 为 busy / idle
async function manage(slot, id) {
  document.body.dataset.manage = "busy";
  const res = await call(slot, id);
  notify("error", res.error ? "操作できませんでした。もう一度お試しください。" : "");
  if (res.error || res.data.done) await refresh();
  document.body.dataset.manage = "idle";
}

// CSV 导出（#11）：选择文件夹的对话框由 Python 弹出；完成后显示导出位置
async function exportCsv(slot, ...args) {
  document.body.dataset.manage = "busy";
  const res = await call(slot, ...args);
  if (res.error) {
    notify("error", "書き出せませんでした。書き出し先を確認して、もう一度お試しください。");
    await refresh();
  } else if (res.data.done) {
    notify("ok", `${res.data.sessions} 件の記録を CSV で書き出しました：${res.data.dir}`);
  }
  document.body.dataset.manage = "idle";
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
  state.chart.setOption(SessionChart.buildOption(record, FONT, state.chart), true);
  renderFeedback(record.feedback);
}

// 文字反馈（#9，app/feedback.py 生成）；没有数据时不显示
function renderFeedback(fb) {
  $("feedback-card").hidden = !fb;
  if (!fb) return;
  $("feedback-summary").textContent = fb.summary;

  $("feedback-segments-section").hidden = fb.segments.length === 0;
  $("feedback-segments").replaceChildren(...fb.segments.map((seg) => {
    const color = SessionChart.PROBLEMS.find((p) => p.key === seg.category).color;
    const dot = Object.assign(document.createElement("span"), { className: "segment-dot" });
    dot.style.background = color;
    const link = Object.assign(document.createElement("button"), { type: "button", className: "segment-link" });
    link.append(dot, seg.text);
    link.addEventListener("click", () => zoomTo(seg.start, seg.end));
    const li = document.createElement("li");
    li.append(link);
    return li;
  }));
  $("feedback-more").hidden = !fb.more;
  $("feedback-more").textContent = fb.more ? `ほかに ${fb.more} か所あります（グラフで確認できます）。` : "";

  $("feedback-hints").replaceChildren(...fb.hints.map((h) => Object.assign(document.createElement("li"), { textContent: h })));

  $("feedback-comparison-section").hidden = !fb.comparison;
  if (fb.comparison) {
    $("feedback-comparison-title").textContent = `前回との比較（${fb.comparison.course_name}）`;
    $("feedback-comparison").textContent = fb.comparison.text;
  }
}

// 把曲线缩放到某一时段（前后各留 30 秒），并滚到图表处
function zoomTo(start, end) {
  state.chart.dispatchAction({ type: "dataZoom", startValue: Math.max(0, start - 30) / 60, endValue: (end + 30) / 60 });
  $("chart-card").scrollIntoView({ behavior: "smooth", block: "start" });
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
  if (state.view === "summary") loadSummary();
  await loadSelected(seq);
}

// ---------- 全体の推移（#8） ----------

async function loadSummary() {
  const seq = ++state.summarySeq;
  document.body.dataset.summary = "loading";
  const res = await call("summarize_sessions");
  if (seq !== state.summarySeq) return;
  $("summary-message").hidden = !res.error;
  if (res.error) {
    $("summary-message").textContent = "データを読み込めませんでした。";
    document.body.dataset.summary = "error";
    return;
  }
  state.summary = res.data;
  renderSummary();
  document.body.dataset.summary = "ready";
}

function renderSummary() {
  const { sessions, courses } = state.summary;
  const keys = ["all", ...courses.map((c) => SummaryChart.filterKey(c.course_name))];
  if (!keys.includes(state.courseFilter)) state.courseFilter = "all";
  $("course-filter").replaceChildren(
    new Option("すべて", "all"),
    ...courses.map((c) => new Option(c.course_name || SummaryChart.NO_NAME, SummaryChart.filterKey(c.course_name))),
  );
  $("course-filter").value = state.courseFilter;

  state.summaryShown = SummaryChart.filterSessions(sessions, state.courseFilter);
  if (!state.summaryChart) {
    state.summaryChart = echarts.init($("summary-chart"));
    // 点击某一节课所在的列（任意一行）：切到「授業ごと」显示它
    state.summaryChart.getZr().on("click", (e) => {
      const chart = state.summaryChart;
      for (let i = 0; i < 3; i++) {
        if (chart.containPixel({ gridIndex: i }, [e.offsetX, e.offsetY])) {
          const index = chart.convertFromPixel({ xAxisIndex: i }, e.offsetX);
          const s = state.summaryShown[Math.round(index)];
          if (s) openSession(s.id);
          return;
        }
      }
    });
  }
  state.summaryChart.setOption(SummaryChart.buildOption(state.summaryShown, FONT), true);
  state.summaryChart.resize();

  const headers = ["授業名", "回数", "平均エンゲージメント", "合計受講時間", ...SessionChart.PROBLEMS.map((p) => p.label)];
  const head = document.createElement("tr");
  head.replaceChildren(...headers.map((h) => Object.assign(document.createElement("th"), { textContent: h })));
  $("course-table").tHead.replaceChildren(head);
  $("course-table").tBodies[0].replaceChildren(...courses.map((c) => {
    const row = SummaryChart.courseRow(c);
    const tr = document.createElement("tr");
    tr.dataset.key = row.key;
    tr.setAttribute("aria-selected", String(row.key === state.courseFilter));
    tr.replaceChildren(...[row.name, row.sessions, row.score, row.duration, ...row.problems]
      .map((text) => Object.assign(document.createElement("td"), { textContent: text })));
    tr.addEventListener("click", () => setCourseFilter(row.key === state.courseFilter ? "all" : row.key));
    return tr;
  }));
}

function setCourseFilter(key) {
  state.courseFilter = key;
  renderSummary();
}

function openSession(id) {
  state.selectedId = id;
  renderSessionList();
  selectTab("single");
  loadSelected(++state.refreshSeq);
}

function selectTab(view) {
  state.view = view;
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === view));
  }
  $("view-single").hidden = view !== "single";
  $("view-summary").hidden = view !== "summary";
  $("view-manage").hidden = view !== "manage";
  if (view === "single" && state.chart) state.chart.resize();
  if (view === "summary") loadSummary();
}

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.addEventListener("click", () => selectTab(tab.dataset.view));
}

$("session-select").addEventListener("change", (e) => {
  state.selectedId = Number(e.target.value);
  loadSelected(++state.refreshSeq);
});

$("course-filter").addEventListener("change", (e) => setCourseFilter(e.target.value));
$("export-all").addEventListener("click", () => exportCsv("export_all"));

window.addEventListener("resize", () => {
  if (state.chart) state.chart.resize();
  if (state.summaryChart) state.summaryChart.resize();
});

new QWebChannel(qt.webChannelTransport, (channel) => {
  state.bridge = channel.objects.bridge;
  state.bridge.data_changed_signal.connect(refresh);
  refresh();
});
