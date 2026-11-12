(() => {
  'use strict';
  const data = window.STRUCTURAL_DATA;
  if (!data) {
    document.querySelector('main').textContent = 'Experiment data could not be loaded. Keep data.js next to index.html.';
    return;
  }

  const NS = 'http://www.w3.org/2000/svg';
  const colors = {grid: '#246e74', lhd: '#bd6b3d', baseline: '#688082', pale: '#e2ece8'};
  const state = {budget: 200, seed: 11, mode: 'all', expMode: 'all'};
  const byId = id => document.getElementById(id);
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const fmt = (value, digits = 3) => value.toFixed(digits);

  // Local downloads use nearby files; hosted copies link back to repository sources.
  if (window.location && window.location.protocol !== 'file:') {
    document.querySelectorAll('a[href^="../"]').forEach(link => {
      link.href = 'https://github.com/analozanocastillo/structural-surrogate-modeling/blob/main/' +
        link.getAttribute('href').slice(3);
    });
  }

  function svgNode(parent, tag, attributes = {}, content = null) {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (content !== null) node.textContent = content;
    parent.appendChild(node);
    return node;
  }
  function clear(svg, width, height) {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  }
  function testValues(design, budget, split, mode) {
    const rows = data.tests.filter(row => row.design === design && row.budget === budget &&
      row.split === split && (mode === 'all' || row.mode === Number(mode)));
    const grouped = new Map();
    for (const row of rows) {
      if (!grouped.has(row.seed)) grouped.set(row.seed, []);
      grouped.get(row.seed).push(row.maeHz);
    }
    const perRun = [...grouped.values()].map(mean);
    if (!perRun.length) throw new Error(`Missing test data: ${design} ${budget} ${split} ${mode}`);
    return {mean: mean(perRun), min: Math.min(...perRun), max: Math.max(...perRun)};
  }
  function expValues(design, budget, mode) {
    const rows = data.experimental.filter(row => row.design === design && row.budget === budget &&
      (mode === 'all' || row.mode === Number(mode)));
    if (!rows.length) throw new Error(`Missing experimental data: ${design} ${budget} ${mode}`);
    return {
      surrogateFem: mean(rows.map(row => row.surrogateFemPp)),
      femExperiment: mean(rows.map(row => row.femExperimentPp)),
      surrogateExperiment: mean(rows.map(row => row.surrogateExperimentPp))
    };
  }

  function renderScatter(svgId, design, seed) {
    const svg = byId(svgId);
    const points = data.points.filter(row => row.design === design && row.budget === state.budget && row.seed === seed);
    clear(svg, 500, 300);
    const left = 44, right = 477, top = 16, bottom = 262;
    const x = position => left + (position - 10) / 110 * (right - left);
    const y = mass => bottom - mass / 50 * (bottom - top);
    for (const mass of [0, 10, 20, 30, 40, 50]) {
      svgNode(svg, 'line', {x1: left, x2: right, y1: y(mass), y2: y(mass), class: 'svg-grid'});
      svgNode(svg, 'text', {x: left - 9, y: y(mass) + 4, 'text-anchor': 'end', class: 'svg-axis'}, String(mass));
    }
    for (const position of [20, 40, 60, 80, 100, 120]) {
      svgNode(svg, 'line', {x1: x(position), x2: x(position), y1: top, y2: bottom, class: 'svg-grid'});
      svgNode(svg, 'text', {x: x(position), y: bottom + 23, 'text-anchor': 'middle', class: 'svg-axis'}, String(position));
    }
    for (const point of points) {
      const corner = (point.position === 10 || point.position === 118) &&
        (point.mass === 1 || point.mass === 50);
      svgNode(svg, 'circle', {cx: x(point.position), cy: y(point.mass), r: corner ? 4.4 : 2.9,
        fill: colors[design], opacity: corner ? 1 : .76,
        stroke: corner ? '#ffffff' : 'none', 'stroke-width': corner ? 1.5 : 0});
    }
    svg.setAttribute('aria-label', `${design.toUpperCase()} sampling design with ${points.length} FEM cases, position on the horizontal axis and mass on the vertical axis.`);
    const uniquePositions = new Set(points.map(point => point.position)).size;
    byId(design === 'grid' ? 'gridCount' : 'lhdCount').textContent =
      `${points.length} cases · ${uniquePositions} distinct FEM positions · corners included`;
  }

  function renderAccuracy() {
    const svg = byId('accuracyChart');
    const narrow = window.innerWidth < 680;
    const width = narrow ? 620 : 1000, height = narrow ? 630 : 355;
    clear(svg, width, height);
    const panels = narrow ? [
      {split: 'test', title: 'Interior held-out positions', x: 0, y: 0, w: 620, h: 305},
      {split: 'test_edge', title: '120 cm edge', x: 0, y: 315, w: 620, h: 305}
    ] : [
      {split: 'test', title: 'Interior held-out positions', x: 0, y: 0, w: 500, h: 350},
      {split: 'test_edge', title: '120 cm edge', x: 500, y: 0, w: 500, h: 350}
    ];
    for (const panel of panels) {
      const values = [200, 400].flatMap(budget => ['grid', 'lhd'].map(design =>
        testValues(design, budget, panel.split, state.mode)));
      const original = testValues('original_grid', 966, panel.split, state.mode).mean;
      const topValue = Math.max(...values.map(value => value.max), original) * 1.22;
      const chartTop = panel.y + 54, chartBottom = panel.y + panel.h - 62;
      const chartLeft = panel.x + 54, chartRight = panel.x + panel.w - 23;
      const y = value => chartBottom - value / topValue * (chartBottom - chartTop);
      const budgetX = budget => panel.x + panel.w * (budget === 200 ? .34 : .69);
      svgNode(svg, 'text', {x: panel.x + 22, y: panel.y + 27, class: 'svg-bar-label'}, panel.title);
      for (let tick = 0; tick <= 4; tick++) {
        const value = topValue * tick / 4;
        svgNode(svg, 'line', {x1: chartLeft, x2: chartRight, y1: y(value), y2: y(value), class: 'svg-grid'});
        svgNode(svg, 'text', {x: chartLeft - 9, y: y(value) + 4, 'text-anchor': 'end', class: 'svg-axis'}, fmt(value, topValue < .2 ? 3 : 2));
      }
      svgNode(svg, 'line', {x1: chartLeft, x2: chartRight, y1: y(original), y2: y(original),
        stroke: colors.baseline, 'stroke-width': 1.5, 'stroke-dasharray': '5 5'});
      svgNode(svg, 'text', {x: chartRight - 2, y: y(original) - 6, 'text-anchor': 'end', class: 'svg-note'},
        `966-case reference ${fmt(original, 3)}`);
      for (const budget of [200, 400]) {
        for (const design of ['grid', 'lhd']) {
          const value = testValues(design, budget, panel.split, state.mode);
          const x = budgetX(budget) + (design === 'grid' ? -29 : 29), barWidth = 47;
          svgNode(svg, 'rect', {x: x - barWidth / 2, y: y(value.mean), width: barWidth,
            height: Math.max(0, chartBottom - y(value.mean)), fill: colors[design], rx: 2,
            opacity: budget === state.budget ? 1 : .45});
          svgNode(svg, 'line', {x1: x, x2: x, y1: y(value.max), y2: y(value.min),
            stroke: '#17313a', 'stroke-width': 1.5});
          svgNode(svg, 'line', {x1: x - 5, x2: x + 5, y1: y(value.max), y2: y(value.max),
            stroke: '#17313a', 'stroke-width': 1.5});
          svgNode(svg, 'line', {x1: x - 5, x2: x + 5, y1: y(value.min), y2: y(value.min),
            stroke: '#17313a', 'stroke-width': 1.5});
          svgNode(svg, 'text', {x, y: y(value.max) - 7, 'text-anchor': 'middle', class: 'svg-note'},
            fmt(value.mean, 3));
        }
        svgNode(svg, 'text', {x: budgetX(budget), y: chartBottom + 23,
          'text-anchor': 'middle', class: 'svg-label'}, `${budget} cases`);
      }
    }
    svg.setAttribute('aria-label', `MLP error against FEM for ${state.mode === 'all' ? 'all six modes' : `mode ${state.mode}`}. The figure compares grid and LHD at ${state.budget} and other budgets for interior test and 120 cm edge.`);
    renderMetricCards();
  }
  function renderMetricCards() {
    const container = byId('metricGrid');
    container.replaceChildren();
    for (const split of ['test', 'test_edge']) {
      for (const design of ['grid', 'lhd']) {
        const value = testValues(design, state.budget, split, state.mode);
        const card = document.createElement('div');
        card.className = `metric-card ${design}-card`;
        const label = document.createElement('small');
        label.textContent = `${design.toUpperCase()} · ${split === 'test' ? 'interior test' : '120 cm edge'}`;
        const number = document.createElement('strong');
        number.textContent = `${fmt(value.mean, 3)} Hz`;
        const range = document.createElement('span');
        range.textContent = `Run range ${fmt(value.min, 3)}–${fmt(value.max, 3)} Hz`;
        card.append(label, number, range);
        container.appendChild(card);
      }
    }
  }

  function renderPhysical() {
    const svg = byId('physicalChart');
    clear(svg, 930, 330);
    const grid = expValues('grid', state.budget, state.expMode);
    const lhd = expValues('lhd', state.budget, state.expMode);
    const original = expValues('original_grid', 966, state.expMode);
    const stages = [
      {key: 'surrogateFem', label: 'MLP ↔ FEM'},
      {key: 'femExperiment', label: 'FEM ↔ measurement'},
      {key: 'surrogateExperiment', label: 'MLP ↔ measurement'}
    ];
    const top = 40, bottom = 262, left = 64, right = 902;
    const maxValue = Math.max(...stages.flatMap(stage => [grid[stage.key], lhd[stage.key]]), original.surrogateExperiment) * 1.28;
    const y = value => bottom - value / maxValue * (bottom - top);
    for (let tick = 0; tick <= 4; tick++) {
      const value = maxValue * tick / 4;
      svgNode(svg, 'line', {x1: left, x2: right, y1: y(value), y2: y(value), class: 'svg-grid'});
      svgNode(svg, 'text', {x: left - 8, y: y(value) + 4, 'text-anchor': 'end', class: 'svg-axis'}, fmt(value, 2));
    }
    stages.forEach((stage, index) => {
      const center = 195 + index * 275;
      for (const [design, value, offset] of [['grid', grid[stage.key], -31], ['lhd', lhd[stage.key], 31]]) {
        const x = center + offset;
        svgNode(svg, 'rect', {x: x - 24, y: y(value), width: 48, height: bottom - y(value),
          fill: colors[design], rx: 2});
        svgNode(svg, 'text', {x, y: y(value) - 9, 'text-anchor': 'middle', class: 'svg-bar-label'}, fmt(value, 3));
      }
      svgNode(svg, 'text', {x: center, y: 292, 'text-anchor': 'middle', class: 'svg-label'}, stage.label);
    });
    svgNode(svg, 'rect', {x: 725, y: 14, width: 11, height: 11, fill: colors.grid});
    svgNode(svg, 'text', {x: 742, y: 24, class: 'svg-note'}, 'Grid');
    svgNode(svg, 'rect', {x: 789, y: 14, width: 11, height: 11, fill: colors.lhd});
    svgNode(svg, 'text', {x: 806, y: 24, class: 'svg-note'}, 'LHD');
    svg.setAttribute('aria-label', `Physical comparison in percentage points. Grid MLP versus experiment ${fmt(grid.surrogateExperiment)}; LHD MLP versus experiment ${fmt(lhd.surrogateExperiment)}; FEM versus experiment ${fmt(grid.femExperiment)}.`);
    const summary = byId('physicalSummary');
    summary.replaceChildren();
    for (const [label, html] of [
      ['MLP ↔ FEM', `Grid ${fmt(grid.surrogateFem)} pp · LHD ${fmt(lhd.surrogateFem)} pp`],
      ['FEM ↔ measurement', `${fmt(grid.femExperiment)} pp · independent of DOE`],
      ['MLP ↔ measurement', `Grid ${fmt(grid.surrogateExperiment)} pp · LHD ${fmt(lhd.surrogateExperiment)} pp`]
    ]) {
      const box = document.createElement('div');
      const title = document.createElement('span'); title.textContent = label;
      const value = document.createElement('strong'); value.textContent = html;
      box.append(title, value); summary.appendChild(box);
    }
    summary.lastChild.appendChild(document.createTextNode(`Original 966-case retraining: ${fmt(original.surrogateExperiment)} pp`));
  }

  function render() {
    for (const button of document.querySelectorAll('[data-budget]'))
      button.setAttribute('aria-pressed', String(Number(button.dataset.budget) === state.budget));
    renderScatter('gridScatter', 'grid', 0);
    renderScatter('lhdScatter', 'lhd', state.seed);
    renderAccuracy();
    renderPhysical();
    byId('provenance').textContent = `Results generated from FEM dataset SHA-256 ${data.sourceHash.slice(0, 16)}… · Full hash in experiment manifest`;
  }
  document.querySelectorAll('[data-budget]').forEach(button => button.addEventListener('click', () => {
    state.budget = Number(button.dataset.budget); render();
  }));
  byId('seedSelect').addEventListener('change', event => {state.seed = Number(event.target.value); render();});
  byId('modeSelect').addEventListener('change', event => {state.mode = event.target.value; render();});
  byId('expModeSelect').addEventListener('change', event => {state.expMode = event.target.value; render();});
  let resizeFrame = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(renderAccuracy);
  });
  render();
})();
