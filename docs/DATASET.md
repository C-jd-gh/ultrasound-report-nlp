# 超声报告数据集说明

## 数据规模

| 报告类型 | 总数 | 训练集 | 测试集 | 验证集 |
| --- | ---: | ---: | ---: | ---: |
| 甲状腺 | 2457 | 1719 | 492 | 246 |
| 乳腺 | 3513 | 2458 | 703 | 352 |
| 肝胆胰脾 | 1394 | 975 | 279 | 140 |
| 合计 | 7364 | 5152 | 1474 | 738 |

原始数据位于：

```text
USData/
  new_Thyroid2.json
  new_Mammary2.json
  new_Liver2.json
  key_technical_words.txt
```

## 字段

每条数据包含：

| 字段 | 含义 | 本项目是否使用 |
| --- | --- | --- |
| `uid` | 报告编号 | 是 |
| `finding` | 超声报告文本 | 是，分词核心数据 |
| `image_path` | 图像文件名 | 否 |
| `labels` | 原始数字标签 | 否 |
| `split` | train/test/val 划分 | 是，仅用于统计 |

本项目不进行分类任务，因此不读取或展示 `labels`。

## 文本特点

- 医学复合词多：`低回声结节`、`囊实混合回声结节`
- 部位表达复杂：`右叶近峡部`、`左叶中下部`
- 否定表达多：`未见`、`未探及`
- 包含 CDFI 血流描述
- 包含大量固定模板和同义表达
- 包含测量占位符

## 测量占位符

| 占位符 | 含义 |
| --- | --- |
| `_2DS_` | 二维尺寸 |
| `_3DS_` | 三维尺寸 |
| `_SCM_` | 厘米数值 |
| `_SMM_` | 毫米数值 |
| `_Loc_` | 钟点或方位 |
| `_LocR_` | 方位范围 |
| `_r_` | 纵横比 |

分词时这些占位符必须保持为完整词元。

## 预处理输出

运行：

```powershell
python scripts/preprocess_data.py
```

生成：

```text
processed_data/
  thyroid_clean.json
  mammary_clean.json
  liver_clean.json
  dataset_stats.json
  medical_dictionary.json
  medical_jieba_userdict.txt
  preprocess_summary.json
```

原始 `USData/` 不会被修改。
