const state = {
  organ: "thyroid",
  sampleIndex: 0,
  wordcloudUrl: null,
};

const algorithmNames = {
  jieba: "Jieba",
  forward: "FMM",
  reverse: "RMM",
};

const categoryNames = {
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
  phrase: "保护短语",
};

const els = Object.fromEntries(
  [
    "organSelect", "reportText", "loadSample", "prevSample", "nextSample",
    "analyzeBtn", "sampleMeta", "jiebaCount", "medicalCount",
    "placeholderCount", "differenceCount", "cleanedText", "jiebaTokens",
    "forwardTokens", "reverseTokens", "jiebaMeta", "forwardMeta",
    "reverseMeta", "agreementGrid", "differenceRows", "posTags",
    "unknownWords", "placeholders", "evaluationSummary", "organMetrics",
    "errorExamples", "datasetStats", "dictSummary", "stopwords",
    "protectedPhrases", "dictGrid", "wordcloudSummary", "wordcloudStatus",
    "wordcloudFrame", "wordcloudImage", "wordcloudPlaceholder", "toast",
  ].map((id) => [id, document.querySelector(`#${id}`)])
);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("show"), 2200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`请求失败：${response.status}`);
  return response.json();
}

async function loadSample(delta = 0) {
  state.sampleIndex = Math.max(0, state.sampleIndex + delta);
  const data = await api(`/api/report/sample?organ=${state.organ}&index=${state.sampleIndex}`);
  state.sampleIndex = data.index;
  els.reportText.value = data.finding;
  els.sampleMeta.textContent = `UID ${data.uid} / ${data.split} / ${data.index + 1} of ${data.total}`;
  await analyzeCurrent();
}

async function analyzeCurrent() {
  const text = els.reportText.value.trim();
  if (!text) {
    showToast("请先输入或载入报告。");
    return;
  }
  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = "分词中";
  try {
    const data = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ organ: state.organ, text }),
    });
    renderAnalysis(data);
    await loadWordcloud(text, data.wordcloud_summary || {});
  } catch (error) {
    showToast(error.message);
  } finally {
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.textContent = "开始分词";
  }
}

function releaseWordcloudUrl() {
  if (state.wordcloudUrl) {
    URL.revokeObjectURL(state.wordcloudUrl);
    state.wordcloudUrl = null;
  }
}

async function loadWordcloud(text, summary) {
  releaseWordcloudUrl();
  els.wordcloudImage.hidden = true;
  els.wordcloudImage.removeAttribute("src");
  els.wordcloudFrame.classList.remove("has-image", "has-error");
  els.wordcloudPlaceholder.hidden = false;
  els.wordcloudPlaceholder.textContent = "正在生成词云...";
  els.wordcloudStatus.textContent = "正在生成词云...";
  renderWordcloudSummary(summary);
  try {
    const response = await fetch("/api/wordcloud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ organ: state.organ, text }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `词云生成失败：${response.status}`);
      }
      const blob = await response.blob();
      state.wordcloudUrl = URL.createObjectURL(blob);
      await new Promise((resolve, reject) => {
        els.wordcloudImage.onload = resolve;
        els.wordcloudImage.onerror = () => reject(new Error("词云图片加载失败，请重新分词。"));
        els.wordcloudImage.src = state.wordcloudUrl;
      });
      els.wordcloudImage.hidden = false;
      els.wordcloudPlaceholder.hidden = true;
      els.wordcloudFrame.classList.add("has-image");
      els.wordcloudStatus.textContent = `使用 ${summary.word_count || 0} 个医学词生成`;
    } catch (error) {
      releaseWordcloudUrl();
      els.wordcloudImage.hidden = true;
      els.wordcloudImage.removeAttribute("src");
      els.wordcloudFrame.classList.add("has-error");
      els.wordcloudPlaceholder.hidden = false;
      els.wordcloudPlaceholder.textContent = error.message;
      els.wordcloudStatus.textContent = error.message;
    }
  }

function renderWordcloudSummary(summary) {
  const words = (summary.top_words || [])
    .map((item) => `<span>${escapeHtml(item.word)} <small>×${escapeHtml(item.count)}</small></span>`)
    .join("");
  els.wordcloudSummary.innerHTML = `
    <strong>有效医学词 ${escapeHtml(summary.word_count || 0)} 个</strong>
    <div class="wordcloud-terms">${words || '<span class="empty-text">暂无有效词</span>'}</div>`;
}

function renderTokens(node, tokens, dictionaryTokens = new Set()) {
  node.innerHTML = tokens
    .map((token) => `<span class="chip ${dictionaryTokens.has(token) ? "medical" : ""}">${escapeHtml(token)}</span>`)
    .join("");
}

function renderAnalysis(data) {
  const stats = data.token_stats || {};
  const medical = new Set((data.pos_tags || []).filter((item) => item.is_medical_term).map((item) => item.word));
  els.jiebaCount.textContent = stats.jieba_count || 0;
  els.medicalCount.textContent = stats.medical_term_count || 0;
  els.placeholderCount.textContent = stats.placeholder_count || 0;
  els.differenceCount.textContent = data.comparison?.different_boundary_count || 0;
  els.cleanedText.textContent = data.cleaned || "";
  els.jiebaMeta.textContent = `${stats.jieba_count || 0} 词`;
  els.forwardMeta.textContent = `${stats.forward_count || 0} 词`;
  els.reverseMeta.textContent = `${stats.reverse_count || 0} 词`;
  renderTokens(els.jiebaTokens, data.jieba_tokens || [], medical);
  renderTokens(els.forwardTokens, data.forward_tokens || [], medical);
  renderTokens(els.reverseTokens, data.reverse_tokens || [], medical);
  renderAgreement(data.comparison || {});
  renderPosTags(data.pos_tags || []);
  renderSimpleChips(els.unknownWords, stats.unknown_words || [], "暂无未登录词候选");
  renderSimpleChips(els.placeholders, stats.placeholders || [], "当前报告无占位符");
}

function renderAgreement(comparison) {
  const pairwise = comparison.pairwise_agreement || {};
  els.agreementGrid.innerHTML = Object.entries(pairwise)
    .map(([name, value]) => {
      const [left, right] = name.split("_vs_");
      return `<article class="metric-card">
        <strong>${algorithmNames[left]} vs ${algorithmNames[right]}</strong>
        <span>${(Number(value) * 100).toFixed(1)}%</span>
        <small>词边界一致率</small>
      </article>`;
    })
    .join("");
  els.differenceRows.innerHTML =
    (comparison.disagreements || [])
      .slice(0, 80)
      .map((item) => `<tr>
        <td>${escapeHtml(item.position)}</td>
        <td>${escapeHtml(item.context)}</td>
        <td>${(item.algorithms || []).map((name) => algorithmNames[name]).join("、")}</td>
      </tr>`)
      .join("") || `<tr><td colspan="3">三种算法没有发现边界差异</td></tr>`;
}

function renderPosTags(items) {
  els.posTags.innerHTML = items
    .map((item) => `<span class="pos-item ${item.is_medical_term ? "medical" : ""}">
      <strong>${escapeHtml(item.word)}</strong>
      <small>${escapeHtml(item.pos)} / ${escapeHtml(item.pos_name)}</small>
    </span>`)
    .join("");
}

function renderSimpleChips(node, items, emptyText) {
  node.innerHTML = items.length
    ? items.map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("")
    : `<span class="empty-text">${escapeHtml(emptyText)}</span>`;
}

async function loadEvaluation() {
  const data = await api("/api/stats");
  renderEvaluation(data.segmentation_metrics || {});
  renderDatasetStats(data);
}

function metricCard(name, metrics) {
  return `<article class="metric-card">
    <strong>${escapeHtml(algorithmNames[name] || name)}</strong>
    <span>F1 ${(Number(metrics.f1 || 0) * 100).toFixed(2)}%</span>
    <small>P ${(Number(metrics.precision || 0) * 100).toFixed(2)}% / R ${(Number(metrics.recall || 0) * 100).toFixed(2)}%</small>
  </article>`;
}

function renderEvaluation(metrics) {
  if (!metrics.overall) {
    els.evaluationSummary.innerHTML = `<article class="stat-block"><h3>尚未生成评测结果</h3><p>请运行 python scripts/evaluate_segmentation.py</p></article>`;
    els.organMetrics.innerHTML = "";
    els.errorExamples.innerHTML = "";
    return;
  }
  els.evaluationSummary.innerHTML = `
    <article class="stat-block"><h3>人工金标准</h3><p>${escapeHtml(metrics.sample_count)} 条超声报告，按词边界计算 Precision、Recall 和 F1。</p></article>
    <div class="metric-grid">${Object.entries(metrics.overall).map(([name, item]) => metricCard(name, item)).join("")}</div>`;
  els.organMetrics.innerHTML = Object.entries(metrics.by_organ || {})
    .map(([organ, algorithms]) => `<article class="stat-block">
      <h3>${escapeHtml({ thyroid: "甲状腺", mammary: "乳腺", liver: "肝胆胰脾" }[organ] || organ)}</h3>
      <div class="metric-grid">${Object.entries(algorithms).map(([name, item]) => metricCard(name, item)).join("")}</div>
    </article>`)
    .join("");
  els.errorExamples.innerHTML = (metrics.error_examples || [])
    .slice(0, 12)
    .map((item) => `<article class="error-item">
      <h3>${algorithmNames[item.algorithm]} / ${escapeHtml(item.organ)} / F1 ${(Number(item.f1) * 100).toFixed(1)}%</h3>
      <p>${escapeHtml(item.text)}</p>
      <div><strong>人工：</strong>${escapeHtml((item.gold_tokens || []).join(" / "))}</div>
      <div><strong>算法：</strong>${escapeHtml((item.predicted_tokens || []).join(" / "))}</div>
    </article>`)
    .join("") || `<p class="empty-text">暂无分词错误</p>`;
}

function renderDatasetStats(data) {
  els.datasetStats.innerHTML = Object.entries(data.organs || {})
    .map(([organ, item]) => `<article class="stat-block">
      <h3>${escapeHtml(item.name)} <small>${escapeHtml(organ)}</small></h3>
      <p>样本数：${escapeHtml(item.total)} / 平均长度：${escapeHtml(item.average_length)} 字</p>
      <p>数据划分：${Object.entries(item.splits || {}).map(([key, value]) => `${key} ${value}`).join(" / ")}</p>
      <p>占位符：${Object.entries(item.placeholders || {}).map(([key, value]) => `${key}(${value})`).join("、")}</p>
    </article>`)
    .join("");
}

async function loadDictionary() {
  const data = await api("/api/dictionary");
  const summary = data.summary || {};
  els.dictSummary.innerHTML = `
    <article class="stat-block"><h3>完整词典</h3><p>${escapeHtml(summary.total_terms)} 个词条</p></article>
    <article class="stat-block"><h3>动态扩展词</h3><p>${escapeHtml(summary.dynamic_terms)} 个数据集高频短语</p></article>
    <article class="stat-block"><h3>停用词</h3><p>${escapeHtml(summary.stopword_count)} 个</p></article>
    <article class="stat-block"><h3>保护短语</h3><p>${escapeHtml(summary.protected_phrase_count)} 个</p></article>`;
  renderSimpleChips(els.stopwords, data.stopwords || [], "暂无停用词");
  renderSimpleChips(els.protectedPhrases, data.protected_phrases || [], "暂无保护短语");
  els.dictGrid.innerHTML = Object.entries(data.categories || {})
    .map(([name, terms]) => `<article class="dict-block">
      <h3>${escapeHtml(categoryNames[name] || name)} <small>${terms.length} 项</small></h3>
      <div class="dict-terms">${terms.map((term) => `<span>${escapeHtml(term)}</span>`).join("")}</div>
    </article>`)
    .join("");
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", async () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.querySelector(`#${tab.dataset.view}`).classList.add("active");
      if (tab.dataset.view === "evaluation" && !els.evaluationSummary.dataset.loaded) {
        await loadEvaluation();
        els.evaluationSummary.dataset.loaded = "true";
      }
      if (tab.dataset.view === "dictionary" && !els.dictSummary.dataset.loaded) {
        await loadDictionary();
        els.dictSummary.dataset.loaded = "true";
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

window.addEventListener("beforeunload", releaseWordcloudUrl);
