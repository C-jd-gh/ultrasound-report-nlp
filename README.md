# 超声报告 NLP 小型网页系统

这是一个面向自然语言处理课程展示的小型系统，基于 `USData` 中的甲状腺、乳腺、肝胆胰脾超声报告，完成数据清洗、Jieba 分词、词性标注、关键词抽取、正则匹配、规则实体识别、CRF 序列实体识别、病灶-器官关系抽取、文本规范化、模板报告生成、质量检查、相似报告检索、标签分类预测、Word2Vec 向量化和 KMeans 聚类。后端使用 Flask 提供网页入口和 JSON API，前端使用原生 HTML、CSS、JavaScript。

## 运行

第一次运行建议先生成预处理数据：

```powershell
python -m pip install -r requirements.txt
python scripts/preprocess_data.py
python scripts/train_classifier.py
python scripts/train_ner.py
python scripts/train_vectors.py
```

然后启动网页系统：

```powershell
python app.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

## 功能

- 独立数据预处理：生成 `processed_data/`，原始 `USData/` 不改动
- 报告类型选择：甲状腺、乳腺、肝胆胰脾
- 样本载入与自定义文本输入
- jieba 分词 + 医学自定义词典
- jieba.posseg 词性标注
- TF-IDF、词频和 TextRank 关键词
- 正则抽取尺寸、部位、回声、边界、形态、血流、淋巴结、术后状态
- 字典和规则混合实体识别
- sklearn-crfsuite 训练 CRF 风格 BIO 序列标注实体识别
- 简易句法/关系抽取：病灶所属器官、部位和属性
- 同义表达和占位符规范化
- 模板化结构摘要生成
- 良恶性倾向提示：复用情感分析中的正负词典计分思想
- 报告质量检查
- TF-IDF 相似报告检索
- TF-IDF + Logistic Regression 标签分类预测
- 分类模型准确率、F1 和混淆矩阵展示
- gensim Word2Vec 领域词向量训练
- TF-IDF + KMeans 报告聚类
- 数据集统计和词典分类展示

## API

- `GET /api/report/sample?organ=thyroid&index=0`
- `POST /api/analyze`
- `POST /api/predict`
- `POST /api/similar`
- `POST /api/cluster`
- `POST /api/vector/nearest`
- `GET /api/stats`
- `GET /api/dictionary`
- `GET /api/model/metrics`
- `GET /api/ner/metrics`
- `GET /api/vector/metrics`

`POST /api/analyze` 示例：

```json
{
  "organ": "thyroid",
  "text": "甲状腺左叶可见一低回声结节，大小约_2DS_，边界清晰，形态规整，CDFI示未探及血流信号。"
}
```

## 测试

```powershell
python -m unittest discover -s tests
```

## 训练分类模型

```powershell
python scripts/train_classifier.py
```

训练完成后会生成：

```text
models/
  thyroid_classifier.pkl
  mammary_classifier.pkl
  liver_classifier.pkl
  classifier_metrics.json
```

网页中的“标签预测”和“分类模型评估”会自动读取这些模型和指标。

## 训练 CRF、Word2Vec 和聚类模型

```powershell
python scripts/train_ner.py
python scripts/train_vectors.py
```

训练完成后会继续生成：

```text
models/
  thyroid_sequence_ner.pkl
  mammary_sequence_ner.pkl
  liver_sequence_ner.pkl
  ner_metrics.json
  ultrasound_word2vec.model
  thyroid_kmeans.pkl
  mammary_kmeans.pkl
  liver_kmeans.pkl
  vector_metrics.json
```

网页中的“CRF 序列实体识别”“聚类结果”“向量与聚类模型”会自动读取这些模型。

## 目录

```text
NLP/
  app.py                  # 网页服务入口
  nlp_pipeline.py         # 兼容旧模型和旧导入的薄入口
  ultrasound_nlp/         # NLP 核心算法包
  scripts/                # 数据预处理和模型训练脚本
  docs/                   # 数据集和项目说明文档
  models/                 # 训练后的分类、NER、向量和聚类模型
  USData/                 # 原始数据
  processed_data/         # 预处理结果
  static/                 # 前端页面
  tests/                  # 单元测试
```
