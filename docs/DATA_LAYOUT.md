# 本地数据目录 | Local Data Layout

本仓库不包含比赛数据。数据目录已被 `.gitignore` 排除，使用者必须从官方渠道取得授权数据并只保存在本地。

Competition data is not included in this repository. Dataset directories are
excluded by `.gitignore`. Users must obtain authorized data from the official
source and keep it local.

## A 榜 | A Track

默认目录结构 | Default layout:

```text
project-root/
  public_dataset_a/
    public_dataset_upload/
      questions/
        group_a/
          *.json
      raw/
        ... source PDF, HTML, and TXT files ...
```

预处理后会在本地生成 | Generated locally after preprocessing:

```text
processed_data/
  questions.jsonl
  documents.jsonl
  chunks.jsonl
  preprocess_report.json
```

```powershell
python script\inspect_dataset.py
python train.py --mode preprocess
python train.py --mode dry-run
```

`inspect_dataset.py` 检查题目数量、文档映射和缺失文件。`dry-run` 只检查检索证据，不调用 API。

`inspect_dataset.py` checks question counts, document mappings, and missing
files. `dry-run` checks retrieval evidence without making an API call.

## B 榜 | B Track

默认目录结构 | Default layout:

```text
project-root/
  upload_b/
    question_b/
      *.json
    submit.csv
  public_dataset_a/
    public_dataset_upload/
      raw/
        ... source documents shared by the task ...
```

B 榜处理文件写入 | B-track processed output:

```text
processed_data_b/
  questions.jsonl
  documents.jsonl
  chunks.jsonl
  preprocess_report.json
```

不同发布批次的数据布局可能变化，应以官方压缩包和提交模板为准。如果字段变化，先更新 `agent/b_preprocess.py` 和提交校验测试。

Official package layouts may vary. Treat the official archive and submission
template as authoritative. If fields change, update `agent/b_preprocess.py` and
the submission-validation tests first.

## 必须保留在本地 | Keep These Files Local

- 原始题目和文档 / raw questions and documents
- 处理后的 chunks / processed chunks
- `answer.csv` 和候选提交 / answers and submission candidates
- `evidence.json`、checkpoint 和缓存 / evidence, checkpoints, and caches
- `local_config.py`、`.env` 和 API 日志 / credentials and API logs

检查忽略规则 | Verify ignore rules:

```powershell
git check-ignore -v local_config.py
git check-ignore -v public_dataset_a
git check-ignore -v runs
```

## Standard Local Project Layout

Source code, private datasets, generated indexes, and protected results should
remain separate:

```text
project-root/
  agent/                    # reusable QA, retrieval, and model code
  script/                   # command-line entry points
  tests/                    # deterministic validation tests
  docs/                     # public documentation
  public_dataset_a/         # local A-track dataset; ignored by Git
  upload_b/                 # local B-track dataset; ignored by Git
  processed_data/           # generated A-track index; ignored by Git
  processed_data_b/         # generated B-track index; ignored by Git
  runs/                     # temporary experiments; ignored by Git
  artifacts/
    best_experiments/       # protected best answers/evidence; ignored by Git
  local_config.py           # local API credentials; ignored by Git
```

Routine outputs belong in `runs/`. Promote a result into
`artifacts/best_experiments/` only after recording its score and evidence. Never
overwrite a protected baseline.
