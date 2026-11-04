"use strict";

// 单节课图表：参与度分数、疲劳指数、认知状态色带、空间状态 / 警报标注，四行共用听课时间轴（分钟）
// 数据转换都是纯函数，挂在 window.SessionChart 上，便于测试直接调用

(function () {
  const COGNITIVE = [
    { code: "Understand", label: "理解", color: "#2a78d6" },
    { code: "Doubt", label: "疑問", color: "#eb6834" },
    { code: "Neutral", label: "中立", color: "#d6d5d0" },
    { code: "Disgust", label: "嫌悪", color: "#1baf7a" },
  ];
  // 空间状态按含义分 3 组，每组一行（行名即第二重编码）；颜色经 dataviz 校验（色盲可区分）
  const SPATIAL_GROUPS = [
    { key: "notes", label: "下向き・ノート", color: "#6250d6", codes: ["head_down", "taking_notes", "hidden_notes"] },
    { key: "distracted", label: "注意散漫", color: "#eda100", codes: ["distracted", "hidden_abnormal"] },
    { key: "away", label: "離席", color: "#e34948", codes: ["away", "absent"] },
  ];
  const SPATIAL_LABELS = {
    head_down: "下向き", taking_notes: "ノート記録中", hidden_notes: "ノート記録中（顔未検出）",
    distracted: "注意散漫", hidden_abnormal: "顔未検出（長時間）", away: "離席中", absent: "不在",
  };
  const ALERT_LABELS = { fatigue: "疲労", head_down: "長時間の下向き", hidden: "顔未検出", absence: "離席継続" };
  const INK = "#52514e";
  const MUTED = "#8a8986";
  const GRID_LINE = "#e6e5e1";

  const TIME_ANCHOR = "__time__";
  const NAMES = {
    score: "エンゲージメント", fatigue: "疲労指数", alert: "アラート", pause: "一時停止",
  };
  const ROWS = { cognitive: ["認知状態"], spatial: [NAMES.alert, ...SPATIAL_GROUPS.map((g) => g.label).reverse()] };

  function formatTime(seconds) {
    const total = Math.max(0, Math.floor(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h > 0 ? `${h}時間${m}分${s}秒` : `${m}分${s}秒`;
  }

  // 听课时间轴的终点（秒）：最后一条样本那一秒的末尾
  function endOf(samples) {
    return samples.length ? samples[samples.length - 1].t + 1 : 0;
  }

  // 连续且认知状态相同的秒合并为一段 [start, end)；状态为空或时间轴不连续（暂停）处断开
  function cognitiveRuns(samples) {
    const runs = [];
    let cur = null;
    for (const x of samples) {
      if (cur && x.cognitive_state === cur.state && x.t === cur.end) {
        cur.end = x.t + 1;
        continue;
      }
      cur = x.cognitive_state === null ? null : { state: x.cognitive_state, start: x.t, end: x.t + 1 };
      if (cur) runs.push(cur);
    }
    return runs;
  }

  // 空间状态事件 → 带分组的区间；仍在进行（t_end 为空）的区间画到听课时间轴末尾
  function spatialIntervals(events, end) {
    const groupOf = {};
    for (const g of SPATIAL_GROUPS) for (const code of g.codes) groupOf[code] = g.key;
    return events
      .filter((e) => e.kind === "spatial" && groupOf[e.type])
      .map((e) => ({
        group: groupOf[e.type], type: e.type, label: SPATIAL_LABELS[e.type],
        start: e.t_start, end: e.t_end === null ? Math.max(end, e.t_start) : e.t_end,
      }));
  }

  function alertMarks(events) {
    return events
      .filter((e) => e.kind === "alert")
      .map((e) => ({ type: e.type, label: ALERT_LABELS[e.type] || e.type, t: e.t_start }));
  }

  // 暂停位置与时长（墙上时间）；未继续（暂停中结束）时 seconds 为 null
  function pauseMarks(pauses) {
    return pauses.map((p) => ({
      t: p.t, seconds: p.resumed_at === null ? null : p.resumed_at - p.paused_at,
    }));
  }

  // 悬停提示的内容：某一秒的数值、认知状态，以及 tol 秒范围内的事件（缩小显示时一像素就有好几秒）
  function describeSecond(ctx, t, tol = 0) {
    const lines = [`<b>${formatTime(t)}</b>`];
    const x = ctx.byT.get(t);
    if (x) {
      lines.push(`${NAMES.score}：${Math.round(x.score)}`);
      lines.push(`${NAMES.fatigue}：${x.fatigue_index.toFixed(2)}`);
      const cog = COGNITIVE.find((c) => c.code === x.cognitive_state);
      lines.push(`認知状態：${cog ? cog.label : "なし"}`);
    } else {
      lines.push("データなし");
    }
    for (const iv of ctx.intervals) {
      if (iv.start < t + 1 + tol && iv.end > t - tol) {
        lines.push(`${iv.label}（${formatTime(iv.start)}〜${formatTime(iv.end)}）`);
      }
    }
    for (const a of ctx.alerts) {
      if (a.t < t + 1 + tol && a.t >= t - tol) lines.push(`${NAMES.alert}：${a.label}`);
    }
    for (const p of ctx.pauses) {
      if (p.t < t + 1 + tol && p.t >= t - tol) lines.push(p.seconds === null ? NAMES.pause : `${NAMES.pause}（${formatTime(p.seconds)}）`);
    }
    return lines.join("<br>");
  }

  // 甘特式矩形：data 为 [开始分钟, 结束分钟, 行号]
  function renderBar(params, api) {
    const start = api.coord([api.value(0), api.value(2)]);
    const end = api.coord([api.value(1), api.value(2)]);
    const height = api.size([0, 1])[1] * 0.7;
    const rect = echarts.graphic.clipRectByRect(
      { x: start[0], y: start[1] - height / 2, width: Math.max(end[0] - start[0], 1), height },
      { x: params.coordSys.x, y: params.coordSys.y, width: params.coordSys.width, height: params.coordSys.height },
    );
    return rect && { type: "rect", transition: [], shape: rect, style: api.style() };
  }

  function barSeries(name, color, gridIndex, data) {
    return {
      name, type: "custom", renderItem: renderBar, xAxisIndex: gridIndex, yAxisIndex: gridIndex,
      itemStyle: { color }, encode: { x: [0, 1], y: 2 }, data, silent: true,
    };
  }

  // 不可见的逐秒曲线（不进图例，不会被隐藏）：悬停提示从它读取所指的秒；
  // 色带上的矩形会把横轴值吸附到矩形起点，不能用
  function timeAnchorSeries(gridIndex, samples) {
    return {
      name: TIME_ANCHOR, type: "line", xAxisIndex: gridIndex, yAxisIndex: gridIndex, silent: true, showSymbol: false,
      lineStyle: { opacity: 0 }, itemStyle: { opacity: 0 }, data: samples.map((x) => [x.t / 60, 0]),
    };
  }

  function pauseSeries(gridIndex, pauses, withLabel) {
    return {
      name: NAMES.pause, type: "line", xAxisIndex: gridIndex, yAxisIndex: gridIndex, data: [],
      itemStyle: { color: MUTED },
      markLine: {
        silent: true, symbol: "none", animation: false,
        lineStyle: { color: MUTED, type: "dashed", width: 1 },
        label: withLabel
          ? { show: true, position: "end", color: INK, formatter: (p) => p.data.text }
          : { show: false },
        data: pauses.map((p) => ({
          xAxis: p.t / 60,
          text: p.seconds === null ? NAMES.pause : `${NAMES.pause} ${formatTime(p.seconds)}`,
        })),
      },
    };
  }

  // chart 用于把悬停容差换算成秒（6 像素）；省略时容差为 0
  function buildOption(record, font, chart) {
    const samples = record.samples;
    const end = endOf(samples);
    const runs = cognitiveRuns(samples);
    const intervals = spatialIntervals(record.events, end);
    const alerts = alertMarks(record.events);
    const pauses = pauseMarks(record.pauses);
    const ctx = { byT: new Map(samples.map((x) => [x.t, x])), intervals, alerts, pauses };
    const min = (s) => s / 60;
    const spatialRow = (label) => ROWS.spatial.indexOf(label);

    const grids = [
      { top: 64, height: 170 },
      { top: 262, height: 80 },
      { top: 368, height: 24 },
      { top: 414, height: 88 },
    ].map((g) => ({ ...g, left: 112, right: 32 }));
    const xAxes = grids.map((_, i) => ({
      type: "value", gridIndex: i, min: 0, max: Math.max(min(end), 1 / 60),
      axisLine: { show: i === 3, lineStyle: { color: MUTED } },
      axisTick: { show: i === 3 },
      axisLabel: { show: i === 3, color: INK, formatter: (v) => String(Math.round(v * 10) / 10) },
      splitLine: { show: true, lineStyle: { color: GRID_LINE } },
      name: i === 3 ? "分" : "", nameTextStyle: { color: INK },
      // 悬停竖线用实线，与暂停的虚线区分
      axisPointer: { show: true, triggerTooltip: true, label: { show: false }, lineStyle: { color: MUTED, type: "solid" } },
    }));
    const valueAxis = (i, name, max) => ({
      type: "value", gridIndex: i, min: 0, max, name, nameLocation: "middle", nameGap: 40,
      nameTextStyle: { color: INK }, axisLabel: { color: INK }, splitLine: { lineStyle: { color: GRID_LINE } },
    });
    const categoryAxis = (i, rows) => ({
      type: "category", gridIndex: i, data: rows, axisTick: { show: false },
      axisLine: { show: false }, axisLabel: { color: INK }, splitLine: { show: false },
      axisPointer: { show: false, triggerTooltip: false },  // 悬停提示由横轴（时间）触发，而不是行名
    });

    const series = [
      {
        name: NAMES.score, type: "line", xAxisIndex: 0, yAxisIndex: 0, showSymbol: false, sampling: "lttb",
        lineStyle: { color: "#009faa", width: 2 }, itemStyle: { color: "#009faa" },
        data: samples.map((x) => [min(x.t), x.score]),
      },
      {
        name: NAMES.fatigue, type: "line", xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, sampling: "lttb",
        lineStyle: { color: INK, width: 2 }, itemStyle: { color: INK },
        data: samples.map((x) => [min(x.t), x.fatigue_index]),
      },
      ...COGNITIVE.map((c) => barSeries(c.label, c.color, 2,
        runs.filter((r) => r.state === c.code).map((r) => [min(r.start), min(r.end), 0]))),
      ...SPATIAL_GROUPS.map((g) => barSeries(g.label, g.color, 3,
        intervals.filter((iv) => iv.group === g.key).map((iv) => [min(iv.start), min(iv.end), spatialRow(g.label)]))),
      {
        name: NAMES.alert, type: "scatter", xAxisIndex: 3, yAxisIndex: 3, symbol: "triangle", symbolSize: 10,
        itemStyle: { color: INK }, data: alerts.map((a) => [min(a.t), spatialRow(NAMES.alert)]),
      },
      timeAnchorSeries(3, samples),
      ...grids.map((_, i) => pauseSeries(i, pauses, i === 0)),
    ];

    return {
      animation: false,
      textStyle: { fontFamily: font },
      legend: {
        top: 0, left: 112, right: 32, itemWidth: 14, itemHeight: 10, textStyle: { color: INK },
        data: [NAMES.score, NAMES.fatigue, ...COGNITIVE.map((c) => c.label),
          ...SPATIAL_GROUPS.map((g) => g.label), NAMES.alert, NAMES.pause],
      },
      tooltip: {
        trigger: "axis", confine: true,
        axisPointer: { type: "line" },
        formatter: (params) => {
          const anchor = params.find((p) => p.seriesName === TIME_ANCHOR);
          if (!anchor) return "";
          const tol = chart ? (chart.convertFromPixel({ xAxisIndex: 0 }, 6) - chart.convertFromPixel({ xAxisIndex: 0 }, 0)) * 60 : 0;
          return describeSecond(ctx, Math.round(anchor.axisValue * 60), Math.max(0, tol));
        },
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      grid: grids,
      xAxis: xAxes,
      yAxis: [valueAxis(0, NAMES.score, 100), valueAxis(1, NAMES.fatigue, 1),
        categoryAxis(2, ROWS.cognitive), categoryAxis(3, ROWS.spatial)],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1, 2, 3], filterMode: "none" },
        { type: "slider", xAxisIndex: [0, 1, 2, 3], filterMode: "none", height: 20, bottom: 8, left: 112, right: 32 },
      ],
      series,
    };
  }

  window.SessionChart = {
    COGNITIVE, SPATIAL_GROUPS, SPATIAL_LABELS, ALERT_LABELS, NAMES,
    formatTime, endOf, cognitiveRuns, spatialIntervals, alertMarks, pauseMarks, describeSecond, buildOption,
  };
})();
