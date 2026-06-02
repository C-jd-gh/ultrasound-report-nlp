# 超声报告 NLP 小型网页系统

这是一个面向自然语言处理课程展示的小型系统，基于 `USData` 中的甲状腺、乳腺、肝胆胰脾超声报告，完成数据清洗、分词、词典构建、关键词抽取、正则匹配、实体识别、文本规范化、模板报告生成、质量检查和相似报告检索。

## 运行

第一次运行建议先生成预处理数据：

```powershell
python preprocess_data.py
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
- 医学词典 + 最大匹配分词
- TF-IDF 关键词和词频关键词
- 正则抽取尺寸、部位、回声、边界、形态、血流、淋巴结、术后状态
- 字典和规则混合实体识别
- 同义表达和占位符规范化
- 模板化结构摘要生成
- 报告质量检查
- TF-IDF 相似报告检索
- 数据集统计和词典分类展示

## API

- `GET /api/report/sample?organ=thyroid&index=0`
- `POST /api/analyze`
- `POST /api/similar`
- `GET /api/stats`
- `GET /api/dictionary`

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

## 目录

```text
NLP/
  app.py                  # 网页服务入口
  preprocess_data.py      # 数据预处理脚本
  nlp_pipeline.py         # NLP 核心流程
  USData/                 # 原始数据
  processed_data/         # 预处理结果
  static/                 # 前端页面
  tests/                  # 单元测试
```
