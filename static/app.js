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
  ORGAN: "器官",
  LOCATION: "部位",
  LESION: "病灶",
  ECHO: "回声",
  BOUNDARY: "边界",
  SHAPE: "形态",
  BLOOD: "血流",
  STATUS: "状态",
  MEASUREMENT: "测量",
  LYMPH: "淋巴",
  SURGERY: "术后",
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
  predictionPanel: document.querySelector("#predictionPanel"),
  riskPanel: document.querySelector("#riskPanel"),
  clusterPanel: document.querySelector("#clusterPanel"),
  posTags: document.querySelector("#posTags"),
  textrankKeywords: document.querySelector("#textrankKeywords"),
  sequenceRows: document.querySelector("#sequenceRows"),
  relationRows: document.querySelector("#relationRows"),
  qualityList: document.querySelector("#qualityList"),
  similarList: document.querySelector("#similarList"),
  statsGrid: document.querySelector("#statsGrid"),
  modelMetrics: document.querySelector("#modelMetrics"),
  nerMetrics: document.querySelector("#nerMetrics"),
  vectorMetrics: document.querySelector("#vectorMetrics"),
  dictSummary: document.querySelector("#dictSummary"),
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
  renderPosTags(data.pos_tags || []);
  renderRankList(els.tfidfKeywords, data.keywords_tfidf, "score");
  renderRankList(els.freqKeywords, data.keywords_frequency, "score");
  renderRankList(els.textrankKeywords, data.keywords_textrank || [], "score");
  renderRegexRows(data.regex_matches);
  renderEntityRows(data.entities);
  renderSequenceRows(data.sequence_entities);
  renderRelationRows(data.relations || []);
  els.normalizedText.textContent = data.normalized;
  els.templateReport.textContent = data.template_report;
  renderPrediction(data.prediction);
  renderRisk(data.risk_tendency);
  renderCluster(data.cluster);
  renderQuality(data.quality);
}

function renderPrediction(prediction) {
  if (!prediction || !prediction.model_ready) {
    els.predictionPanel.innerHTML = `
      <article class="prediction-card">
        <h3>模型尚未训练</h3>
        <p>${escapeHtml(prediction?.message || "请先运行 python scripts/train_classifier.py")}</p>
      </article>`;
    return;
  }
  const confidence = Math.round(Number(prediction.confidence || 0) * 1000) / 10;
  const probs = (prediction.top_labels || [])
    .map((item) => {
      const score = Math.round(Number(item.score || 0) * 1000) / 10;
      return `
        <div class="rank-item">
          <div>标签 ${escapeHtml(item.label)}</div>
          <div>
            <div class="bar"><span style="width:${Math.max(4, score)}%"></span></div>
            <div class="score">${score}%</div>
          </div>
        </div>`;
    })
    .join("");
  els.predictionPanel.innerHTML = `
    <article class="prediction-card">
      <div class="prediction-main">
        <div>
          <p>预测标签</p>
          <div class="prediction-label">${escapeHtml(prediction.predicted_label)}</div>
          <p>置信度 ${confidence}%</p>
        </div>
        <div class="prob-list">${probs}</div>
      </div>
    </article>`;
}

function renderRisk(risk) {
  if (!risk) {
    els.riskPanel.innerHTML = `<article class="prediction-card"><p>暂无倾向结果</p></article>`;
    return;
  }
  const suspicious = (risk.suspicious_evidence || []).map((item) => `<span class="chip danger">${escapeHtml(item)}</span>`).join("");
  const benign = (risk.benign_evidence || []).map((item) => `<span class="chip ok">${escapeHtml(item)}</span>`).join("");
  els.riskPanel.innerHTML = `
    <article class="prediction-card">
      <h3>${escapeHtml(risk.level)} <small>得分 ${escapeHtml(risk.score)}</small></h3>
      <p>${escapeHtml(risk.method)}</p>
      <div class="evidence-row">${suspicious || "<span>无可疑征象命中</span>"}</div>
      <div class="evidence-row">${benign || "<span>无良性/正常表达命中</span>"}</div>
      <p>${escapeHtml(risk.disclaimer)}</p>
    </article>`;
}

function renderCluster(cluster) {
  if (!cluster || !cluster.model_ready) {
    els.clusterPanel.innerHTML = `
      <article class="prediction-card">
        <h3>聚类模型尚未训练</h3>
        <p>${escapeHtml(cluster?.message || "请先运行 python scripts/train_vectors.py")}</p>
      </article>`;
    return;
  }
  const terms = (cluster.cluster_terms || []).map((term) => `<span class="chip">${escapeHtml(term)}</span>`).join("");
  const reports = (cluster.similar_reports || [])
    .slice(0, 3)
    .map((item) => `<li>UID ${escapeHtml(item.uid)} / 标签 ${escapeHtml(item.label)} / 相似度 ${escapeHtml(item.score)}</li>`)
    .join("");
  els.clusterPanel.innerHTML = `
    <article class="prediction-card">
      <h3>簇 ${escapeHtml(cluster.cluster)}：${escapeHtml(cluster.profile_name || inferClusterName(cluster.cluster_terms || []))} <small>${escapeHtml(cluster.model_name)}</small></h3>
      <p>${escapeHtml(cluster.profile_summary || "当前报告被归入该文本模式簇。")}</p>
      <div class="chips">${terms}</div>
      <ul class="compact-list">${reports}</ul>
    </article>`;
}

function renderPosTags(items) {
  els.posTags.innerHTML =
    items
      .slice(0, 120)
      .map(
        (item) => `
        <span class="pos-item ${item.is_medical_term ? "medical" : ""}">
          <strong>${escapeHtml(item.word)}</strong>
          <small>${escapeHtml(item.pos)} / ${escapeHtml(item.pos_name)}</small>
        </span>`
      )
      .join("") || `<span class="pos-item">暂无词性标注</span>`;
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

function renderSequenceRows(sequence) {
  if (!sequence || !sequence.model_ready) {
    els.sequenceRows.innerHTML = `<tr><td colspan="4">${escapeHtml(sequence?.message || "CRF 模型尚未训练")}</td></tr>`;
    return;
  }
  els.sequenceRows.innerHTML =
    (sequence.entities || [])
      .slice(0, 100)
      .map(
        (item) => `
        <tr>
          <td>${escapeHtml(typeNames[item.label] || item.label)}</td>
          <td>${escapeHtml(item.value)}</td>
          <td>${escapeHtml(item.start)}-${escapeHtml(item.end)}</td>
          <td>${escapeHtml(item.source)}</td>
        </tr>`
      )
      .join("") || `<tr><td colspan="4">暂无 CRF 实体</td></tr>`;
}

function renderRelationRows(items) {
  els.relationRows.innerHTML =
    items
      .slice(0, 80)
      .map((item) => {
        const attrs = [
          item.size && `尺寸：${item.size}`,
          item.echo && `回声：${item.echo}`,
          item.boundary && `边界：${item.boundary}`,
          item.shape && `形态：${item.shape}`,
          item.blood && `血流：${item.blood}`,
        ]
          .filter(Boolean)
          .join("；");
        return `
        <tr>
          <td>${escapeHtml(item.subject)}</td>
          <td>${escapeHtml(item.predicate)}</td>
          <td>${escapeHtml(item.object)}</td>
          <td>${escapeHtml(attrs || item.rule)}</td>
        </tr>`;
      })
      .join("") || `<tr><td colspan="4">暂无病灶关系</td></tr>`;
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
  await loadModelMetrics();
  await loadNerMetrics();
  await loadVectorMetrics();
}

async function loadModelMetrics() {
  const data = await api("/api/model/metrics");
  if (!data.model_ready) {
    els.modelMetrics.innerHTML = `
      <article class="metric-block">
        <h3>模型尚未训练</h3>
        <p>${escapeHtml(data.message || "请先运行 python scripts/train_classifier.py")}</p>
      </article>`;
    return;
  }
  els.modelMetrics.innerHTML = Object.entries(data.organs)
    .map(([organ, item]) => renderMetricBlock(organ, item))
    .join("");
}

function renderMetricBlock(organ, item) {
  const test = item.test || {};
  return `
    <article class="metric-block">
      <h3>${escapeHtml(item.name)} <small>${escapeHtml(organ)}</small></h3>
      <p>测试集准确率：${escapeHtml(test.accuracy ?? "-")} / 宏平均 F1：${escapeHtml(test.macro_f1 ?? "-")} / 加权 F1：${escapeHtml(test.weighted_f1 ?? "-")}</p>
      ${renderConfusionMatrix(test.labels || [], test.confusion_matrix || [])}
    </article>`;
}

function renderConfusionMatrix(labels, matrix) {
  if (!labels.length || !matrix.length) {
    return `<p>暂无混淆矩阵</p>`;
  }
  const header = labels.map((label) => `<th>预测${escapeHtml(label)}</th>`).join("");
  const rows = matrix
    .map((row, index) => {
      const cells = row.map((value) => `<td>${escapeHtml(value)}</td>`).join("");
      return `<tr><th>真实${escapeHtml(labels[index])}</th>${cells}</tr>`;
    })
    .join("");
  return `
    <div class="confusion-wrap">
      <table class="confusion-table">
        <thead><tr><th></th>${header}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

async function loadNerMetrics() {
  const data = await api("/api/ner/metrics");
  if (!data.model_ready) {
    els.nerMetrics.innerHTML = `
      <article class="metric-block">
        <h3>CRF 模型尚未训练</h3>
        <p>${escapeHtml(data.message || "请先运行 python scripts/train_ner.py")}</p>
      </article>`;
    return;
  }
  els.nerMetrics.innerHTML = Object.entries(data.organs)
    .map(([organ, item]) => {
      return `
        <article class="metric-block">
          <h3>${escapeHtml(item.name)} <small>${escapeHtml(organ)}</small></h3>
          <p>训练样本：${escapeHtml(item.train_count)} / 测试样本：${escapeHtml(item.test_count)} / 加权 F1：${escapeHtml(item.weighted_f1 ?? "-")}</p>
        </article>`;
    })
    .join("");
}

async function loadVectorMetrics() {
  const data = await api("/api/vector/metrics");
  if (!data.model_ready) {
    els.vectorMetrics.innerHTML = `
      <article class="metric-block">
        <h3>向量模型尚未训练</h3>
        <p>${escapeHtml(data.message || "请先运行 python scripts/train_vectors.py")}</p>
      </article>`;
    return;
  }
  const word2vec = data.word2vec || {};
  const blocks = [
    `
      <article class="metric-block">
        <h3>词向量模型 <small>${escapeHtml(word2vec.model_name || "")}</small></h3>
        <p>系统已用超声报告训练领域词向量，可用于发现医学词之间的语义相近关系。词表大小：${escapeHtml(word2vec.vocabulary_size)} / 向量维度：${escapeHtml(word2vec.vector_size)}</p>
      </article>`,
  ];
  const clusterBlocks = Object.entries(data.clusters?.organs || {}).map(([organ, item]) => {
    return `
      <article class="metric-block">
        <h3>${escapeHtml(item.name)} 聚类画像 <small>${escapeHtml(organ)}</small></h3>
        <p>样本数：${escapeHtml(item.sample_count)} / 簇数：${escapeHtml(item.cluster_count)} / 轮廓系数：${escapeHtml(item.silhouette_score ?? "-")}</p>
        <div class="cluster-grid">${(item.clusters || []).map(renderClusterProfile).join("")}</div>
      </article>`;
  });
  els.vectorMetrics.innerHTML = blocks.concat(clusterBlocks).join("");
}

function inferClusterName(terms) {
  const joined = (terms || []).join(" ");
  const rules = [
    ["术后/复查类", ["术后", "切除", "全切", "原区域", "保乳"]],
    ["淋巴结相关类", ["淋巴结", "淋巴门", "肿大淋巴结", "颈部", "腋下"]],
    ["血流变化类", ["CDFI", "血流信号", "血流丰富", "可探及血流信号"]],
    ["囊实混合结节类", ["囊实混合回声", "囊性为主", "实性为主"]],
    ["多发结节类", ["多发结节", "多个结节", "双叶", "双侧", "多个"]],
    ["弥漫性回声改变类", ["回声增粗", "回声增强", "回声欠均匀", "实质回声", "形态饱满"]],
    ["正常/未见异常类", ["大小形态如常", "未见明确占位性病变", "未见异常血流信号", "回声均匀"]],
    ["胆囊/胆管相关类", ["胆囊", "胆管", "扩张"]],
  ];
  let bestName = "文本模式聚类";
  let bestScore = 0;
  rules.forEach(([name, keywords]) => {
    const score = keywords.filter((keyword) => joined.includes(keyword)).length;
    if (score > bestScore) {
      bestName = name;
      bestScore = score;
    }
  });
  return bestName;
}

function renderClusterProfile(cluster) {
  const terms = (cluster.top_terms || [])
    .slice(0, 8)
    .map((term) => `<span class="chip">${escapeHtml(term)}</span>`)
    .join("");
  const representatives = (cluster.representatives || [])
    .slice(0, 2)
    .map((item) => {
      const text = String(item.finding || "");
      const shortText = text.length > 90 ? `${text.slice(0, 90)}...` : text;
      return `<li>UID ${escapeHtml(item.uid)} / 标签 ${escapeHtml(item.label)}：${escapeHtml(shortText)}</li>`;
    })
    .join("");
  const name = cluster.profile_name || inferClusterName(cluster.top_terms || []);
  const summary =
    cluster.profile_summary ||
    `该簇主要由包含“${(cluster.top_terms || []).slice(0, 5).join("、") || "暂无高权重词"}”等词的报告组成。`;
  return `
    <section class="cluster-profile">
      <div class="cluster-profile-head">
        <strong>簇 ${escapeHtml(cluster.cluster)}：${escapeHtml(name)}</strong>
        <span>${escapeHtml(cluster.count)} 条</span>
      </div>
      <p>${escapeHtml(summary)}</p>
      <div class="chips">${terms}</div>
      <ul class="compact-list">${representatives}</ul>
    </section>`;
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
  const summary = data.summary || {};
  els.dictSummary.innerHTML = `
    <article class="stat-block">
      <h3>完整词典规模</h3>
      <p>总词条：${escapeHtml(summary.total_terms ?? data.size)} / 人工词条：${escapeHtml(summary.manual_terms ?? "-")} / 数据扩展词：${escapeHtml(summary.dynamic_terms ?? "-")}</p>
      <p>Jieba 用户词典：${escapeHtml(summary.jieba_userdict || "processed_data/medical_jieba_userdict.txt")}</p>
    </article>
    ${(data.sources || [])
      .map(
        (item) => `
        <article class="stat-block">
          <h3>${escapeHtml(item.name)} <small>${escapeHtml(item.count)} 项</small></h3>
          <p>${escapeHtml(item.description)}</p>
        </article>`
      )
      .join("")}
    <article class="stat-block">
      <h3>数据集高频扩展词 <small>${(data.dynamic_examples || []).length} 项示例</small></h3>
      <div class="dict-terms">${(data.dynamic_examples || []).map((term) => `<span>${escapeHtml(term)}</span>`).join("")}</div>
    </article>`;
  els.dictGrid.innerHTML = Object.entries(data.categories)
    .map(
      ([name, terms]) => `
      <article class="dict-block">
        <h3>${escapeHtml(typeNames[name] || name)} <small>${terms.length} 项</small></h3>
        <div class="dict-terms">${terms.slice(0, 160).map((term) => `<span>${escapeHtml(term)}</span>`).join("")}</div>
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
