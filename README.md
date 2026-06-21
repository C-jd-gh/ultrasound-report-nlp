# 超声报告中文分词系统

这是一个面向自然语言处理课程的领域分词项目。系统使用甲状腺、乳腺、肝胆胰脾三类超声报告，完成文本清洗、医学词典构建、停用词过滤、Jieba 分词、正向最大匹配、逆向最大匹配、词性标注、人工金标准评测和单条报告词云。

项目只研究中文分词，不进行疾病分类、标签预测、实体识别或报告诊断。

## 技术路线

```text
原始超声报告
→ 文本清洗
→ 医学短语保护
→ Jieba / FMM / RMM 分词
→ 停用词过滤
→ 词性标注
→ 算法边界对比
→ 人工金标准评测
```

系统包含三种分词方法：

- Jieba + 超声医学自定义词典
- 正向最大匹配 FMM
- 逆向最大匹配 RMM

## 安装与运行

```powershell
python -m pip install -r requirements.txt
python scripts/preprocess_data.py
python scripts/evaluate_segmentation.py
python app.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

## 页面功能

### 报告分词

- 选择甲状腺、乳腺或肝胆胰脾报告
- 输入自定义报告或载入数据集样本
- 同时展示 Jieba、FMM、RMM 三种结果
- 展示词边界一致率和不同切分位置
- 展示词性、医学词、占位符和未登录词候选
- 根据当前报告的医学分词结果动态生成词云

### 分词评测

- 使用 30 条人工标注超声报告
- 按词边界计算 Precision、Recall、F1
- 展示总体指标、分器官指标和典型错误

当前人工金标准评测结果：

| 算法 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Jieba + 医学词典 | 0.8246 | 0.9691 | 0.8910 |
| 正向最大匹配 FMM | 0.7311 | 0.8969 | 0.8056 |
| 逆向最大匹配 RMM | 0.7607 | 0.9175 | 0.8318 |

### 词典构建

- 人工分类医学词典
- `USData/key_technical_words.txt` 技术词表
- 从数据集中扩展的高频医学短语
- `resources/medical_phrases.txt` 强制保护短语
- `resources/stopwords.txt` 停用词表

## API

- `GET /api/report/sample?organ=thyroid&index=0`
- `POST /api/analyze`
- `POST /api/wordcloud`
- `GET /api/stats`
- `GET /api/dictionary`

`POST /api/analyze` 请求示例：

```json
{
  "organ": "thyroid",
  "text": "甲状腺右叶可见低回声结节，大小约_2DS_。"
}
```

`POST /api/wordcloud` 使用相同的请求结构，直接返回 `image/png`。词云会过滤停用词、纯数字和测量占位符，并优先显示医学词典词。词云只用于分词结果可视化，不代表医学重要性或诊断结论。

## 目录结构

```text
NLP/
  app.py
  requirements.txt
  ultrasound_nlp/
    nlp_pipeline.py
  scripts/
    preprocess_data.py
    evaluate_segmentation.py
  resources/
    stopwords.txt
    medical_phrases.txt
    segmentation_gold.json
  USData/
  processed_data/
  static/
  tests/
  docs/
```

## 测试

```powershell
python -m unittest discover -s tests
```

该项目只用于课程学习和分词算法研究，不用于临床诊断。
