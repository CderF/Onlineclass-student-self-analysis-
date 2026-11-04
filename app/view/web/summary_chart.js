"use strict";

// 多节课汇总（#8）：横轴为每一节课（按时间顺序），三行共用横轴——平均参与度、听课时长、问题时段占比
// 三类问题时段可能重叠，所以并列而不堆叠：每类各自是「该类时长 / 听课时长」，不超过 100%
// 数据转换都是纯函数，挂在 window.SummaryChart 上，便于测试直接调用；问题时段的名称与颜色沿用 SessionChart

(function () {
  const INK = "#52514e";
  const MUTED = "#8a8986";
  const GRID_LINE = "#e6e5e1";
  const ACCENT = "#009faa";
  const NO_NAME = "授業名なし";
  const STATUS_SUFFIX = { completed: "", interrupted: "（中断）", recording: "（記録中）" };
  const NAMES = { score: "平均エンゲージメント", duration: "受講時間" };
  const VISIBLE = 15;  // 课次多于此数时出现滑条，默认显示最近的 15 节

  const problems = () => window.SessionChart.PROBLEMS;

  // 筛选值："all" 全部、"none" 未填写课程名、"name:<课程名>"
  function filterKey(courseName) {
    return courseName === null ? "none" : `name:${courseName}`;
  }

  function filterSessions(sessions, key) {
    return key === "all" ? sessions : sessions.filter((s) => filterKey(s.course_name) === key);
  }

  function share(seconds, total) {
    return total > 0 ? (seconds / total) * 100 : 0;
  }

  function formatMinutes(seconds) {
    const m = Math.round(seconds / 60);
    return m >= 60 ? `${Math.floor(m / 60)}時間${m % 60}分` : `${m}分`;
  }

  function formatClock(seconds) {
    const s = Math.round(seconds);
    return `${Math.floor(s / 60)}分${s % 60}秒`;
  }

  function shortDate(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  }

  // 日期、课程名、状态（中断 / 记录中）各一行，避免长名称和状态挤在一起被截断
  function axisLabel(s) {
    const status = (STATUS_SUFFIX[s.status] ?? "").replace(/[（）]/g, "");
    return [shortDate(s.started_at), s.course_name || NO_NAME, status].filter(Boolean).join("\n");
  }

  function describeSession(s) {
    const date = new Date(s.started_at * 1000).toLocaleString("ja-JP", {
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
    });
    const lines = [
      `<b>${date}　${s.course_name || NO_NAME}${STATUS_SUFFIX[s.status] ?? ""}</b>`,
      `${NAMES.score}：${Math.round(s.avg_score)}`,
      `${NAMES.duration}：${formatMinutes(s.seconds)}`,
    ];
    for (const p of problems()) {
      const st = s.problems[p.key];
      lines.push(`${p.label}：${Math.round(share(st.seconds, s.seconds))}%（${st.count}回・${formatClock(st.seconds)}）`);
    }
    lines.push("クリックでこの授業を表示");
    return lines.join("<br>");
  }

  // 课程别汇总表的一行（显示用文字）
  function courseRow(c) {
    return {
      key: filterKey(c.course_name),
      name: c.course_name || NO_NAME,
      sessions: c.interrupted ? `${c.sessions}（うち中断 ${c.interrupted}）` : String(c.sessions),
      score: String(Math.round(c.avg_score)),
      duration: formatMinutes(c.seconds),
      problems: problems().map((p) => `${Math.round(share(c.problems[p.key].seconds, c.seconds))}%`),
    };
  }

  function buildOption(sessions, font) {
    const n = sessions.length;
    const labels = sessions.map(axisLabel);
    const grids = [
      { top: 40, height: 150 },
      { top: 220, height: 90 },
      { top: 340, height: 120 },
    ].map((g) => ({ ...g, left: 72, right: 32 }));
    const xAxes = grids.map((_, i) => ({
      type: "category", gridIndex: i, data: sessions.map((_, j) => String(j)),
      axisTick: { show: i === 2, alignWithLabel: true },
      axisLine: { show: i === 2, lineStyle: { color: MUTED } },
      axisLabel: { show: i === 2, color: INK, interval: 0, width: 72, overflow: "truncate", formatter: (v) => labels[Number(v)] },
      axisPointer: { show: true, type: "shadow", label: { show: false } },
    }));
    const valueAxis = (i, name, extra) => ({
      type: "value", gridIndex: i, min: 0, name, nameLocation: "middle", nameGap: 44,
      nameTextStyle: { color: INK }, axisLabel: { color: INK }, splitLine: { lineStyle: { color: GRID_LINE } },
      ...extra,
    });
    const series = [
      {
        name: NAMES.score, type: "line", xAxisIndex: 0, yAxisIndex: 0, symbolSize: 9,
        lineStyle: { color: ACCENT, width: 2 }, itemStyle: { color: ACCENT },
        // 中断 / 记录中的课用空心点标出
        data: sessions.map((s) => ({
          value: Math.round(s.avg_score * 10) / 10,
          symbol: s.status === "completed" ? "circle" : "emptyCircle",
        })),
      },
      {
        name: NAMES.duration, type: "bar", xAxisIndex: 1, yAxisIndex: 1, barMaxWidth: 28,
        itemStyle: { color: MUTED, borderRadius: [4, 4, 0, 0] },
        data: sessions.map((s) => Math.round((s.seconds / 60) * 10) / 10),
      },
      ...problems().map((p) => ({
        name: p.label, type: "bar", xAxisIndex: 2, yAxisIndex: 2, barMaxWidth: 10, barGap: "15%",
        itemStyle: { color: p.color, borderRadius: [3, 3, 0, 0] },
        data: sessions.map((s) => Math.round(share(s.problems[p.key].seconds, s.seconds) * 10) / 10),
      })),
    ];
    const zoom = n > VISIBLE
      ? [{ type: "inside", xAxisIndex: [0, 1, 2], startValue: n - VISIBLE, endValue: n - 1 },
        { type: "slider", xAxisIndex: [0, 1, 2], startValue: n - VISIBLE, endValue: n - 1, height: 16, bottom: 4, left: 72, right: 32 }]
      : [];
    return {
      animation: false,
      textStyle: { fontFamily: font },
      legend: {
        top: 0, left: 72, right: 32, itemWidth: 14, itemHeight: 10, textStyle: { color: INK },
        data: [NAMES.score, NAMES.duration, ...problems().map((p) => p.label)],
      },
      tooltip: {
        trigger: "axis", confine: true,
        formatter: (params) => (params.length ? describeSession(sessions[params[0].dataIndex]) : ""),
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: grids,
      xAxis: xAxes,
      yAxis: [
        valueAxis(0, NAMES.score, { max: 100 }),
        valueAxis(1, "受講時間（分）"),
        valueAxis(2, "問題時段（%）", { max: 100 }),
      ],
      dataZoom: zoom,
      series,
    };
  }

  window.SummaryChart = {
    NO_NAME, NAMES, VISIBLE, filterKey, filterSessions, share, formatMinutes, axisLabel, describeSession,
    courseRow, buildOption,
  };
})();
