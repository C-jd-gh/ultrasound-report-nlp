const state = {
  organ: "thyroid",
  sampleIndex: 0,
  lastAnalysis: null,
};

const typeNames = {
  organ: "器官",
  location: "部位",
  lesion: "病灶",
  echo: "回声",
  boundary: "边界",
  shape: "形态",
  blood: "血流",
  status: "状态",
  measurement: "测量",
  technical: "技术词",
  lymph: "淋巴",
  surgery: "术后",
  lesion_event: "病灶事件",
};

const els = {
  organSelect: document.querySelector("#organSelect"),
  reportText: document.querySelector("#reportText"),
  loadSample: document.querySelector("#loadSample"),
  prevSample: document.querySelector("#prevSample"),
  nextSample: document.querySelector("#nextSample"),
  analyzeBtn: document.querySelector("#analyzeBtn"),
  sampleMeta: document.querySelector("#sampleMeta"),
  tokenCount: document.querySelector("#tokenCount"),
  entityCount: document.querySelector("#entityCount"),
  ruleCount: document.querySelector("#ruleCount"),
  qualityCount: document.querySelector("#qualityCount"),
  tokens: document.querySelector("#tokens"),
  tfidfKeywords: document.querySelector("#tfidfKeywords"),
  freqKeywords: document.querySelector("#freqKeywords"),
  regexRows: document.querySelector("#regexRows"),
  entityRows: document.querySelector("#entityRows"),
  normalizedText: document.querySelector("#normalizedText"),
  templateReport: document.querySelector("#templateReport"),
  qualityList: document.querySelector("#qualityList"),
  similarList: document.querySelector("#similarList"),
  statsGrid: document.querySelector("#statsGrid"),
  dictGrid: document.querySelector("#dictGrid"),
  toast: document.querySelector("#toast"),
};

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.remove("show"), 2200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadSample(delta = 0) {
  state.sampleIndex = Math.max(0, state.sampleIndex + delta);
  const data = await api(`/api/report/sample?organ=${state.organ}&index=${state.sampleIndex}`);
  state.sampleIndex = data.index;
  els.reportText.value = data.finding;
  els.sampleMeta.textContent = `UID ${data.uid} / ${data.split} / 标签 ${data.label} / ${data.index + 1} of ${data.total}`;
  await analyzeCurrent();
}

async function analyzeCurrent() {
  const text = els.reportText.value.trim();
  if (!text) {
    showToast("请先输入或载入一条报告。");
    return;
  }
  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = "分析中";
  try {
    const analysis = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ organ: state.organ, text }),
    });
    state.lastAnalysis = analysis;
    renderAnalysis(analysis);
    await loadSimilar(text);
  } finally {
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.textContent = "开始分析";
  }
}

async function loadSimilar(text) {
  const data = await api("/api/similar", {
    method: "POST",
    body: JSON.stringify({ organ: state.organ, text, limit: 5 }),
  });
  renderSimilar(data.items || []);
}

function renderAnalysis(data) {
  els.tokenCount.textContent = data.tokens.length;
  els.entityCount.textContent = data.entities.length;
  els.ruleCount.textContent = data.regex_matches.length;
  els.qualityCount.textContent = data.quality.length;
  els.tokens.innerHTML = data.tokens.map((token) => `<span class="chip">${escapeHtml(token)}</span>`).join("");
  renderRankList(els.tfidfKeywords, data.keywords_tfidf, "score");
  renderRankList(els.freqKeywords, data.keywords_frequency, "score");
  renderRegexRows(data.regex_matches);
  renderEntityRows(data.entities);
  els.normalizedText.textContent = data.normalized;
  els.templateReport.textContent = data.template_report;
  renderQuality(data.quality);
}

function renderRankList(node, items, key) {
  const max = Math.max(...items.map((item) => Number(item[key]) || 0), 1);
  node.innerHTML = items
    .map((item) => {
      const score = Number(item[key]) || 0;
      const width = Math.max(5, Math.round((score / max) * 100));
      return `
        <div class="rank-item">
          <div>${escapeHtml(item.word)}</div>
          <div>
            <div class="bar"><span style="width:${width}%"></span></div>
            <div class="score">${escapeHtml(score)}</div>
          </div>
        </div>`;
    })
    .join("");
}

function renderRegexRows(items) {
  els.regexRows.innerHTML =
    items
      .slice(0, 80)
      .map(
        (item) => `
        <tr>
          <td>${escapeHtml(typeNames[item.type] || item.type)}</td>
          <td>${escapeHtml(item.value)}</td>
          <td>${item.start}-${item.end}</td>
          <td>${escapeHtml(item.rule)}</td>
        </tr>`
      )
      .join("") || `<tr><td colspan="4">暂无规则命中</td></tr>`;
}

function renderEntityRows(items) {
  els.entityRows.innerHTML =
    items
      .slice(0, 100)
      .map((item) => {
        const value = item.attributes ? `${item.value}\n${JSON.stringify(item.attributes, null, 2)}` : item.value;
        return `
        <tr>
          <td>${escapeHtml(typeNames[item.type] || item.type)}</td>
          <td>${escapeHtml(item.name)}</td>
          <td>${escapeHtml(value)}</td>
          <td>${escapeHtml(item.source)}</td>
        </tr>`;
      })
      .join("") || `<tr><td colspan="4">暂无实体</td></tr>`;
}

function renderQuality(items) {
  els.qualityList.innerHTML = items
    .map(
      (item) => `
      <div class="quality-item ${escapeHtml(item.level)}">
        ${escapeHtml(item.message)}
      </div>`
    )
    .join("");
}

function renderSimilar(items) {
  els.similarList.innerHTML =
    items
      .map(
        (item) => `
        <article class="similar-item">
          <h3>UID ${item.uid} / ${item.split} / 标签 ${item.label} / 相似度 ${item.score}</h3>
          <p>${escapeHtml(item.finding)}</p>
        </article>`
      )
      .join("") || `<div class="similar-item"><p>暂无相似报告</p></div>`;
}

async function loadStats() {
  const data = await api("/api/stats");
  els.statsGrid.innerHTML = Object.entries(data.organs)
    .map(([key, item]) => {
      const labelMax = Math.max(...Object.values(item.labels), 1);
      const labels = Object.entries(item.labels)
        .map(([label, count]) => miniBar(`标签 ${label}`, count, labelMax))
        .join("");
      const splitText = Object.entries(item.splits)
        .map(([split, count]) => `${split}: ${count}`)
        .join(" / ");
      const terms = item.top_terms
        .slice(0, 8)
        .map((term) => `${term.word}(${term.count})`)
        .join("、");
      return `
        <article class="stat-block">
          <h3>${escapeHtml(item.name)} <small>${escapeHtml(key)}</small></h3>
          <p>样本数：${item.total} / 数据划分：${escapeHtml(splitText)}</p>
          <p>高频词：${escapeHtml(terms)}</p>
          <div class="mini-bars">${labels}</div>
        </article>`;
    })
    .join("");
}

function miniBar(label, count, max) {
  const width = Math.max(4, Math.round((count / max) * 100));
  return `
    <div class="mini-bar">
      <span>${escapeHtml(label)}</span>
      <div class="mini-bar-track"><span style="width:${width}%"></span></div>
      <span>${count}</span>
    </div>`;
}

async function loadDictionary() {
  const data = await api("/api/dictionary");
  els.dictGrid.innerHTML = Object.entries(data.categories)
    .map(
      ([name, terms]) => `
      <article class="dict-block">
        <h3>${escapeHtml(typeNames[name] || name)} <small>${terms.length} 项示例</small></h3>
        <div class="dict-terms">${terms.map((term) => `<span>${escapeHtml(term)}</span>`).join("")}</div>
      </article>`
    )
    .join("");
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", async () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.querySelector(`#${tab.dataset.view}`).classList.add("active");
      if (tab.dataset.view === "stats" && !els.statsGrid.dataset.loaded) {
        await loadStats();
        els.statsGrid.dataset.loaded = "true";
      }
      if (tab.dataset.view === "dictionary" && !els.dictGrid.dataset.loaded) {
        await loadDictionary();
        els.dictGrid.dataset.loaded = "true";
      }
    });
  });

  els.organSelect.addEventListener("change", async () => {
    state.organ = els.organSelect.value;
    state.sampleIndex = 0;
    await loadSample();
  });
  els.loadSample.addEventListener("click", () => loadSample());
  els.prevSample.addEventListener("click", () => loadSample(-1));
  els.nextSample.addEventListener("click", () => loadSample(1));
  els.analyzeBtn.addEventListener("click", analyzeCurrent);
}

async function init() {
  bindEvents();
  try {
    await loadSample();
  } catch (error) {
    showToast(error.message);
  }
}

init();
