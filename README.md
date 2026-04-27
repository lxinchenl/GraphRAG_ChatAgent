


## 技术方案

- 文档解析：`PyMuPDF`、`python-docx`、`python-pptx`、`RapidOCR`
- 向量检索：`Chroma`
- 本地向量模型：`sentence-transformers + BAAI/bge-small-zh-v1.5`
- 图数据库：`Neo4j`
- 聊天大模型：兼容 OpenAI 风格接口
- 界面：`Streamlit`


## 准备数据

把你的资料放到 `data/` 目录下，例如：

- `pdf`
- `docx`
- `doc`
- `pptx`
- `png`
- `jpg`


## 环境变量

复制 `.env.example` 为 `.env`，填写：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_CHAT_MODEL`
- `EMBEDDING_PROVIDER`
- `LOCAL_EMBED_MODEL`
- `EMBEDDING_DEVICE`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## 运行方式

先安装项目：

```bash
pip install -e .
```

说明：

- 首次运行会自动下载本地 embedding 模型
- 如果你有 NVIDIA GPU，可把 `.env` 里的 `EMBEDDING_DEVICE=cpu` 改成 `cuda`
- 运行前请先确保本地 `Neo4j` 已启动，且 `.env` 中密码正确


先启动neo4j数据库
neo4j.bat console


构建索引：

```bash
kg-rag ingest
```

只重建向量库，不构建知识图谱：

```bash
kg-rag ingest-vector --reset
```

这个命令会在写入完成后自动启动一个新的 Python 进程做自检：重新打开本地 Chroma 并执行一次检索，尽早发现 HNSW 索引损坏问题。

如果数据不在默认 `data/` 目录，也可以指定：

```bash
kg-rag ingest-vector --reset --data-dir add_data
```

命令行提问：

```bash
kg-rag ask "什么是事务？"
```

查看调试上下文：

```bash
kg-rag debug "数据库范式有哪些？"
```

启动界面：

```bash
streamlit run app.py
```

## 本地 Qwen 测试

如果你想把知识图谱关系抽取改成本地模型，可以先测试本地 `Qwen` 推理。

当前项目提供了一个测试脚本，默认使用官方模型：

- `Qwen/Qwen3-4B-Instruct-2507`

模型权重会下载到：

- `kg_rag_demo/model/Qwen3-4B-Instruct-2507/`

先重新安装项目依赖：

```bash
pip install -e .
```

仅下载模型：

```bash
kg-rag-local-qwen-test --download-only
```

下载并执行一次本地推理：

```bash
kg-rag-local-qwen-test
```

自定义提示词：

```bash
kg-rag-local-qwen-test --prompt "请简要说明数据库系统的组成。"
```

如果你有 NVIDIA GPU，可以显式指定：

```bash
kg-rag-local-qwen-test --device cuda
```

如果只想在 CPU 上验证流程：

```bash
kg-rag-local-qwen-test --device cpu
```

## 当前实现内容

- 解析 `PDF / DOCX / DOC / PPTX / 图片`
- 对扫描 PDF 页面自动尝试 OCR
- 将文本切块后写入 `Chroma`
- 用大模型从 chunk 中抽取实体关系
- 将关系写入 `Neo4j`
- 查询时同时参考向量检索结果和图谱关系


加载数据
neo4j-admin database load neo4j --from-path=C:\Users\29026\neo4j_export --overwrite-destination=true

查询节点
match(n) return n

查询关系
match()