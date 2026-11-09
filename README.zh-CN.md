# rag-framework

[English](README.md) | [简体中文](README.zh-CN.md)

用受控实验评估 RAG：不仅看某个组件改对了哪些答案，也看它改坏了哪些答案。

rag-framework 在同一份数据切片上运行闭卷、基础 RAG、增强 RAG 和验证修订四种方案，并分别统计答案正确性、检索相关性、证据支持和引用质量。验证修订总是与它所基于的初稿配对比较，因此"改对的"和"改坏的"都会被计入。

- 内置 KILT（NQ、HotpotQA）、ALCE（ASQA、QAMPARI）、SciFact、GaRAGe、RAGTruth 和 RGB 的数据适配器
- 可替换或移除的组件：查询改写、BM25、稠密检索、排名融合、交叉编码器重排、句子筛选、带引用生成和验证修订
- 支持逐项移除消融和逐步叠加消融，也可以在已保存的初稿上用不同修订策略重放验证
- 输入缺失、任务不支持、调用失败都记录为明确的状态，而不是记作 0 分
- 每次运行都会保存解析后的配置、预测、检索与各阶段轨迹、指标、token 用量和耗时

**项目状态：** 研究原型，版本 0.1.0。命令行参数、配置项和记录格式在后续版本中可能变化。

## 快速上手

第一次运行使用一个只有三个问题的合成数据集，不需要 API 密钥，也不会下载任何东西。

```sh
python3.11 -m venv .venv
. .venv/bin/activate              # Windows PowerShell：.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m scripts.init_local

python -m scripts.run_pipeline --dataset toy --pipeline vanilla_rag --execution-profile smoke --generator extractive --limit 3 --run-id toy-vanilla
```

Windows 上用 `py -3.11 -m venv .venv` 创建环境。所有命令都在仓库根目录下执行。

命令会打印一段 JSON 摘要，并把结果写入 `outputs/formal/toy-vanilla/`。主要指标位于 `metrics.track_results.vanilla_rag.primary_metrics`：

```json
"primary_metrics": {
  "all_gold_hit@10": 1.0,
  "em": 0.0,
  "f1": 0.2738095238095238,
  "recall@10": 1.0
}
```

每个问题都检索到了标准答案所在的段落。精确匹配（EM）为 0，是因为 `extractive` 生成器直接返回整句证据（"The Harbor Observatory roof is made of copper."），而参考答案只是 "copper"；token 级 F1 会给出部分分数。

也可以脱离这次运行，单独对保存下来的预测打分：

```sh
python -m scripts.evaluate_predictions --predictions outputs/formal/toy-vanilla/predictions.jsonl --processed-root examples/toy/processed --output-dir outputs/evaluation/toy-vanilla --judge none
```

每次运行都要用新的 `--run-id`，每次评估都要用新的 `--output-dir`；命令不会覆盖已有结果，而是直接报错。

## 流水线

| 流水线 | 做什么 | 用途 |
| --- | --- | --- |
| `closed_book` | 只用生成器作答，不做检索。 | 无检索基线。 |
| `vanilla_rag` | 用原始问题做稠密检索，把原始段落交给生成器。 | 标准 RAG 基线。 |
| `best_practice_rag` | 查询改写、BM25 + 稠密混合检索、交叉编码器重排、句子筛选和带引用生成。 | 增强组合。名称只是这套配置的标签，不代表评测结果。 |
| `rag_verification` | 对照证据检查 `best_practice_rag` 初稿中的论断，只在证据支持时修改。 | 被测试的干预手段。 |
| `oracle_context_rag` | 根据标准相关性标注选取上下文。 | 诊断用条件，使用了真实系统拿不到的信息。 |

实验套件会在同一份切片上依次运行前四种流水线，并把 `best_practice_rag` 的初稿直接交给 `rag_verification`，保证两者基于同一批初始答案比较。答案格式、引用要求和 token 预算会按各数据集的任务自动调整。

## 支持的数据集

| 流水线中的名称 | 数据准备用的键 | 任务 | 上游 | 目录中记录的许可证 |
| --- | --- | --- | --- | --- |
| `kilt_nq` | `kilt_nq` | 单跳开放域问答 | [KILT](https://github.com/facebookresearch/KILT) | 代码为 MIT；源数据集和维基百科内容沿用各自条款 |
| `kilt_hotpotqa` | `kilt_hotpotqa` | 多跳开放域问答 | [KILT](https://github.com/facebookresearch/KILT) | 同上 |
| `alce_asqa` | `alce` | 歧义问题，带引用的长答案 | [ALCE](https://github.com/princeton-nlp/ALCE) | 代码为 MIT；附带数据集沿用各自条款 |
| `alce_qampari` | `alce` | 带引用的列表答案 | [ALCE](https://github.com/princeton-nlp/ALCE) | 同上 |
| `scifact` | `scifact` | 科学论断核查 | [SciFact](https://github.com/allenai/scifact) | 论断与标注 CC-BY-4.0；摘要 ODC-By-1.0；代码 Apache-2.0 |
| `garage` | `garage` | 依据证据作答与选择性拒答 | [GaRAGe](https://github.com/amazon-science/GaRAGe) | CC-BY-NC-4.0 |
| `ragtruth_qa` | `ragtruth` | 对给定回答做幻觉检测 | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | MIT |
| `rgb_en` | `rgb` | 对噪声、反事实证据和多源信息整合的鲁棒性 | [RGB](https://github.com/chen700564/RGB) | CC-BY-NC-SA-4.0 |

本框架不分发任何数据集。使用或分享数据前，请查看各上游的许可证，其中有几个禁止商用。

## 在真实基准上运行

下面以 ALCE ASQA 为例走完整个流程。这一部分需要下载模型并提供 OpenAI API 密钥。建议先用 `--limit 20` 测试，根据 `cost_latency.json` 里的用量决定如何扩大规模。

**1. 安装模型相关依赖。**

```sh
python -m pip install -e ".[models]"
```

请按自己的 CPU 或 GPU 选择 PyTorch 版本。默认模型为 `BAAI/bge-large-en-v1.5`（向量检索）、`BAAI/bge-reranker-large`（重排）和 `cross-encoder/nli-deberta-v3-base`（NLI），版本固定在 [`configs/models/default_retrieval.example.yaml`](configs/models/default_retrieval.example.yaml) 中。

**2. 配置生成模型。** 在环境变量中设置 `OPENAI_API_KEY`，程序只从这里读取密钥。然后保存模型 ID；如果希望运行时报告费用，再填上服务商的单价（美元 / 百万 token）：

```sh
python -m scripts.configure_openai_local --model YOUR_MODEL_ID --input-price INPUT_PRICE --output-price OUTPUT_PRICE
```

设置保存在 `configs/openai.local.json`，该文件不会被 git 跟踪。TLS 证书、模型选择顺序等细节见[进阶用法](docs/advanced-usage.md#api-generation)。

**3. 准备数据。** 第一条命令只打印下载计划，列出来源、许可证、版本、目标路径和所需文件。

```sh
python -m scripts.prepare_benchmark_data --datasets alce
python -m scripts.prepare_benchmark_data --datasets alce --download
python -m scripts.prepare_benchmark_data --datasets alce --profile --process
python -m scripts.prepare_benchmark_data --datasets alce --validate
```

`alce` 这个键会同时生成 `alce_asqa` 和 `alce_qampari`。如果手动下载，把上游文件放到计划中列出的位置，再执行处理和校验两步。

**4. 为要运行的切片构建稠密索引：**

```sh
python -m scripts.build_index --dataset alce_asqa --pipeline vanilla_rag --split-profile mvp --limit 20 --embedder config --search-backend numpy --output-dir models/indexes/asqa-example
```

NumPy 检索不需要额外安装。使用 `--search-backend faiss` 时，需要安装与你的平台匹配的 FAISS。

**5. 比较四种方案。**

```sh
python -m scripts.run_experiment_suite --dataset alce_asqa --split-profile mvp --limit 20 --execution-profile quality_max --generator openai --query-rewriter openai --dense-index-path models/indexes/asqa-example --suite-id asqa-example
```

**6. 衡量各组件的贡献。** 逐项移除（leave-one-out）每次去掉 `best_practice_rag` 五个组件中的一个，再与完整组合比较。改用 `--design sequential` 则从 `vanilla_rag` 出发，每一步加入一个组件，直到加上验证修订。

```sh
python -m scripts.run_ablation --dataset alce_asqa --split-profile mvp --limit 20 --execution-profile quality_max --generator openai --query-rewriter openai --dense-index-path models/indexes/asqa-example --design leave-one-out --ablation-id asqa-components
```

**7. 评估已保存的预测。** 用法与快速上手中的 `scripts.evaluate_predictions` 相同。基于参考答案的指标不需要模型。需要通过 API 抽取论断时设置 `--extractor-model`；做支持判断时使用 `--judge local-nli`，或 `--judge openai` 加 `--judge-model`。凡是调用 API 的步骤都还需要 `--allow-network`。互相比较的预测文件必须来自同一数据集、同一批样例、同一上下文和同一随机种子。

## 术语

| 术语 | 含义 |
| --- | --- |
| `mvp`、`dev_small`、`formal` | 适配器生成的数据划分。`dev_small` 是开发集（多数数据集最多 300 条），`mvp` 是其中前 100 条，用于快速测试；`formal` 是与开发集不重叠的评估集。RAGTruth 使用 `judge_calibration` 和 `judge_eval`。每次运行读取 `--split-profile` 对应的划分文件，文件不存在时直接报错。 |
| `outputs/formal/` | 保存流水线运行结果的目录。目录名是固定的，与所用划分无关。 |
| `smoke` 组件 | 确定性的替身组件，只用来检查数据流是否跑通。它们的分数不反映模型质量。 |
| 初稿 | `best_practice_rag` 在验证修订之前给出的答案。验证结果总是和它所基于的初稿配对。 |
| 上下文集 | 数据集为某个样例在某种条件下提供的一组固定段落，例如 RGB 的某个噪声比例。用 `--context-condition` 和 `--context-metadata` 选择，详见[进阶用法](docs/advanced-usage.md#context-conditions)。 |
| 候选缓存 | 预先为每个 KILT 问题冻结的一批维基百科候选，运行时直接使用，不再检索全量语料。 |

## 配置

```text
configs/
  paths.example.json             数据、模型和输出的存放位置
  datasets/*.example.yaml        数据集与任务定义
  pipelines/*.example.yaml       流水线各阶段与预算
  profiles/*.example.yaml        执行配置的覆盖项
  models/*.example.yaml          检索、重排和 NLI 模型
```

`scripts.init_local` 会在 `configs/paths.local.json` 不存在时创建它，已有文件不会被覆盖。路径默认相对于仓库根目录，也可以写成绝对路径。所有 `*.local.json` 文件都不会被 git 跟踪。

| 执行配置 | 用途 |
| --- | --- |
| `smoke` | 用确定性替身组件检查数据流。 |
| `quality_max` | 使用真实模型组件；缺少组件时直接报错，不会退回替身。 |
| `budget_matched` | 在相同的 token 和检索上限下做比较。 |

超出预算、请求失败和模型输出格式错误都会保留在运行记录中。wheel 安装、缓存位置、TLS 设置和模型快照见[进阶用法](docs/advanced-usage.md)。

## 输出

| 文件 | 内容 |
| --- | --- |
| `config.snapshot.yaml`、`run_manifest.json` | 解析后的设置、标识和运行元数据 |
| `predictions.jsonl` | 答案、任务相关的结构化字段和样例元数据 |
| `retrieval_traces.jsonl`、`stage_traces.jsonl` | 检索结果，以及每个阶段的输入和输出 |
| `metrics.json`、`all_metrics.csv`、`all_metrics.html` | 带可用状态的指标，分别为 JSON、CSV 和 HTML 格式 |
| `cost_latency.json` | token 用量、耗时，以及在已知单价时的费用 |

`scripts.evaluate_predictions` 会把 `evaluations.jsonl`、评估清单以及调用和缓存记录写到自己的输出目录，不会修改原预测文件。`scripts.run_diagnostics --run-dir <运行目录>` 会在运行目录下新建一个带时间戳的 `diagnostics-v1-*` 文件夹，写入 `diagnostics.jsonl`。输出中可能包含原文片段和本机绝对路径，分享前请检查。`outputs/` 目录不会被 git 跟踪。

指标定义、配对方式和统计方法见[评估说明](docs/evaluation.md)，记录格式见[统一数据格式](docs/data-format.md)（均为英文）。

## 其他工具

| 命令 | 用途 |
| --- | --- |
| `scripts.run_verification_replay` | 在已保存的初稿上，用指定策略（`detect_only`、`constrained_revision`、`abstain_only`）重新运行验证 |
| `scripts.run_rgb_stress` | 在 RGB 的不同噪声比例、信息整合和反事实条件下做比较 |
| `scripts.run_ragtruth_benchmark` | 对 RAGTruth QA 中的回答做基于证据的幻觉检测 |
| `scripts.run_judge` | 对已保存的预测做证据与引用判断 |
| `scripts.run_diagnostics` | 为已保存的运行生成检索和生成错误诊断 |
| `scripts.build_kilt_fullwiki_fts`、`build_kilt_candidate_cache`、`embed_kilt_candidate_cache` | 为 KILT 构建全量维基百科检索索引和冻结的候选池 |
| `scripts.freeze_model_assets` | 下载固定版本的模型快照并生成清单 |

各命令的参数见其 `--help`，更完整的示例见[进阶用法](docs/advanced-usage.md)。

## 扩展

| 方向 | 代码位置（`src/rag_benchmark/`） |
| --- | --- |
| 配置与执行 | `config.py`、`component_factory.py`、`runner.py` |
| 记录与数据集适配 | `data.py`、`schemas.py`、`dataset_adapters.py`、`dataset_catalog.py` |
| 查询与上下文准备 | `query_formulation/`、`retrieval/`、`reranking/`、`compression/` |
| 生成与修订 | `generation/`、`verification/` |
| 评估与诊断 | `evaluation/`、`analysis/` |
| 比较与产物 | `ablation.py`、`sampling.py`、`run_summary.py`、`paired_artifacts.py`、`reporting/` |

新增数据集时，需要提供统一格式的记录、数据集配置、路径条目和明确的划分文件。新增组件时，实现对应接口并在 `component_factory.py` 中注册。新指标在跨运行比较之前，应先定义它的输入、分析单位和缺失状态。

## 开发

```sh
python -m pip install -e ".[dev]"
python -m scripts.check_environment
python -m pytest -q
```

测试完全离线，只使用小型合成数据（说明见 [`tests/fixtures/README.md`](tests/fixtures/README.md)）。`check_environment` 默认按纯 CPU 环境检查；加上 `--require-models` 或 `--require-cuda` 可以检查可选依赖。

欢迎提交问题报告和 pull request。提交 pull request 前请先跑一遍测试。

## 请注意

- toy 数据集和 smoke 组件只用来跑通接口，它们的分数不反映模型质量。
- 基于模型的评判需要针对具体任务校准，并经过人工验证。测试通过不一定是评估器准确，也未必说明修订有益。
- 证据支持是相对于所选证据衡量的，可能与事实正确性不一致。KILT 使用候选池检索时，报告结果应注明检索范围。
- 按 token 估算的费用可能与实际账单不同。缓存、重试和外部服务都会影响结果的可重复性。
- 本仓库只提供工具，不包含实验结果。

## 引用

如果你在研究中使用了 rag-framework，请引用本仓库：

```bibtex
@software{rag_framework_2026,
  author = {SilentScreaming},
  title  = {rag-framework: controlled experiments and evaluation for retrieval-augmented generation},
  year   = {2026},
  version = {0.1.0},
  url    = {https://github.com/SilentScreaming/Rag-framework}
}
```

## 许可证

源码采用 [MIT 许可证](LICENSE)。模型和数据集需另行获取，并沿用各自的许可证。
