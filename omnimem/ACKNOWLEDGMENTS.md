# 致谢 Acknowledgments

OmniMem 的设计与实现受益于众多优秀的开源项目、学术研究和技术社区。在此致以诚挚的感谢。

## 直接依赖的开源项目

| 项目 | 许可证 | 用途 |
|:---|:---|:---|
| [ChromaDB](https://github.com/chroma-core/chroma) | Apache-2.0 | 向量数据库存储与语义检索 |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | MIT | BM25 关键词检索算法实现 |
| [tiktoken](https://github.com/openai/tiktoken) | MIT | Token 计数与文本分词 |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | YAML 配置解析 |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Apache-2.0 | 语义嵌入与 Cross-Encoder 重排序 |
| [cryptography](https://github.com/pyca/cryptography) | Apache-2.0 / BSD | 隐私数据加密 |
| [PEFT](https://github.com/huggingface/peft) | Apache-2.0 | LoRA 参数高效微调 |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | 大语言模型推理与训练 |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | 深度学习框架 |

## 架构与算法灵感

### 记忆系统架构

| 概念/项目 | 来源 | 在 OmniMem 中的体现 |
|:---|:---|:---|
| **五层记忆架构** | 人类认知神经科学（感觉记忆→工作记忆→长期记忆） | L0感知→L1工作→L2结构化→L3深层→L4内化 |
| **Hindsight** | 开源记忆框架 | Reflect 工具循环、Consolidation 四阶段升华、Disposition 性格系统 |
| **MemPalace** | 记忆宫殿方法论 | Wing/Room/Hall 三级空间组织结构 |
| **MemOS / ActMemory** | KV Cache 记忆研究 | L4 KV Cache 预填充机制、知识图谱时序三元组 |
| **ReMe** | 会话记忆系统 | 6字段结构化摘要设计、会话监听与信号检测 |
| **memU** | 主动式记忆系统 | 主动感知引擎、意图预测与预加载 |

### 检索与排序算法

| 算法 | 来源 | 应用场景 |
|:---|:---|:---|
| **BM25** | Robertson, S. et al. (1994). Okapi at TREC-3. *NIST Special Publication*. | 关键词精确匹配检索 |
| **Reciprocal Rank Fusion (RRF)** | Cormack, G.V. et al. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR*. | 多路检索结果融合排序 |
| **Cross-Encoder Re-ranking** | Nogueira, R. & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*. | 检索结果精排优化 |
| **Vector Similarity Search** | Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. *IEEE TPAMI*. | 语义向量检索 |

### 计算机科学基础

| 概念/技术 | 来源 | 应用场景 |
|:---|:---|:---|
| **Saga 模式** | Garcia-Molina, H. & Salem, K. (1987). Sagas. *ACM SIGMOD*. | 多数据源事务协调与最终一致性 |
| **向量时钟 (Vector Clock)** | Mattern, F. (1988). Virtual Time and Global States of Distributed Systems. | 多实例同步与因果排序 |
| **艾宾浩斯遗忘曲线** | Ebbinghaus, H. (1885). *Über das Gedächtnis* | 四阶段记忆归档策略 |
| **记忆宫殿/Method of Loci** | 古罗马修辞学传统 | 空间化记忆组织结构 |

### 软件工程实践

| 概念 | 来源 | 应用场景 |
|:---|:---|:---|
| **三层分离架构** | Anthropic managed-agents 设计哲学 | 存储层/上下文管理层/上下文窗口分离 |
| **LoRA (Low-Rank Adaptation)** | Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. | L4 内化记忆参数微调 |
| **KV Cache 优化** | 大语言模型推理优化实践 | 高频记忆模式缓存加速 |

## 开发工具与生态

| 工具 | 用途 |
|:---|:---|
| [Black](https://github.com/psf/black) | 代码格式化 |
| [Ruff](https://github.com/astral-sh/ruff) | 快速 Python 代码检查 |
| [pytest](https://github.com/pytest-dev/pytest) | 测试框架 |
| [mypy](https://github.com/python/mypy) | 静态类型检查 |
| [pre-commit](https://github.com/pre-commit/pre-commit) | Git 提交前检查 |

## 相关开源记忆系统

以下项目为 OmniMem 提供了宝贵的参考与启发：

- **[MemGPT](https://github.com/cpacker/MemGPT)** — LLM 操作系统级记忆管理
- **[mem0](https://github.com/mem0ai/mem0)** — 个性化 AI 记忆层
- **[Letta](https://github.com/letta-ai/letta)** (原 MemGPT) — 有状态 LLM 应用框架
- **[LangChain Memory](https://github.com/langchain-ai/langchain)** — LLM 应用记忆模块
- **[Zep](https://github.com/getzep/zep)** — 长期记忆服务
- **[CoALA](https://github.com/lingo-mit/coala)** — 认知架构与语言 Agent
- **[Generative Agents](https://github.com/joonspk-research/generative_agents)** — 生成式智能体模拟

## 学术参考

1. **记忆与认知架构**
   - Atkinson, R.C. & Shiffrin, R.M. (1968). Human memory: A proposed system and its control processes. *Psychology of Learning and Motivation*.
   - Baddeley, A.D. (2000). The episodic buffer: a new component of working memory? *Trends in Cognitive Sciences*.

2. **信息检索**
   - Manning, C.D. et al. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
   - Lin, J. et al. (2021). Pretrained Transformers for Text Ranking: BERT and Beyond. *Foundations and Trends in IR*.

3. **知识表示**
   - Hogan, A. et al. (2021). Knowledge Graphs. *ACM Computing Surveys*.

4. **LLM 记忆优化**
   - Wu, Y. et al. (2024). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.
   - Zhong, Y. et al. (2024). ReMe: A Memory-Enhanced Framework for LLM Agents. *arXiv*.

## 特别感谢

- **Anthropic** — 通过 Claude 系列模型和 AI 安全研究，为 Agent 记忆系统设计提供了重要参考
- **OpenAI** — tiktoken 等工具链为 Token 级记忆管理奠定基础
- **Hugging Face** — Transformers 生态推动了开源 LLM 记忆技术的发展
- **开源社区** — 所有为上述项目贡献代码、文档和想法的开发者

---

> 如有遗漏或错误，欢迎提交 Issue 指正。
