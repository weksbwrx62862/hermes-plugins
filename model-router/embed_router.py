"""Embedding 语义任务分类器。

使用本地轻量嵌入模型对查询进行向量化，通过 k-NN 与示例向量库匹配，
输出任务类型、复杂度区间和置信度。模型与示例库在后台线程懒加载，
加载失败时自动降级到关键词启发式，避免阻塞 Gateway 启动。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 默认模型配置：优先使用 multilingual MiniLM，失败则回退到 bge-small-zh
_PRIMARY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_FALLBACK_MODEL = "BAAI/bge-small-zh"

# 默认 k-NN 参数：经验值，在测试集上表现较优
_K_NEIGHBORS = 5
_SIM_POWER = 2

# 有效任务类型与复杂度范围
_TASK_TYPES = [
    "simple_qa",
    "classify",
    "extract",
    "long_doc",
    "code",
    "math",
    "complex_reasoning",
    "agent",
]
_MIN_COMPLEXITY = 1
_MAX_COMPLEXITY = 5


class _KeywordFallback:
    """最小化关键词启发式兜底分类器，避免循环引用 plugins/model-router/__init__.py。"""

    _CLASSIFY_KW = ["分类", "归类", "判断是否", "classify", "categorize", "是真是假", "类别"]
    _EXTRACT_CORE_KW = ["提取", "抽取", "摘录", "extract", "parse"]
    _EXTRACT_WEAK_KW = ["摘要", "summarize", "总结"]
    _SIMPLE_QA_KW = ["多少钱", "价格", "天气", "时间", "翻译", "什么是", "定义", "hello", "hi", "ping", "是谁", "哪里", "怎么样"]
    _LONG_DOC_KW = ["文档", "论文", "长文", "报告", "document", "paper", "report", "阅读理解", "白皮书", "年报", "手册"]
    _CODE_KW = [
        "代码", "code", "编程", "函数", "class", "def", "import", "bug", "修复", "重构",
        "refactor", "debug", "测试", "test", "python", "java", "go", "rust", "js", "ts",
        "react", "vue", "api", "接口", "算法", "algorithm", "sql", "数据库", "编译", "部署",
        "deploy", "docker", "git", "script", "bash",
    ]
    _MATH_KW = ["计算", "数学", "方程", "公式", "calculate", "math", "equation", "证明", "积分", "微分", "矩阵", "概率", "导数"]
    _COMPLEX_REASONING_KW = [
        "分析", "优化", "设计", "架构", "安全", "性能", "诊断", "排查", "漏洞",
        "explain", "analyze", "比较", "对比", "区别", "差异", "优缺点", "评估", "系统",
        "分布式", "事务", "一致性", "高可用", "容灾", "选型", "方案", "影响",
    ]
    _AGENT_KW = ["帮我", "执行", "操作", "调用工具", "agent", "工具", "搜索", "联网", "api", "运行", "发送", "预订", "安排"]

    @classmethod
    def detect_task_type(cls, query: str) -> str:
        """关键词启发式任务类型检测。"""
        lower = query.lower()

        extract_score = sum(1 for kw in cls._EXTRACT_CORE_KW if kw in lower)
        if extract_score == 0 and any(kw in lower for kw in cls._EXTRACT_WEAK_KW):
            extract_score = 0
        elif any(kw in lower for kw in cls._EXTRACT_WEAK_KW):
            extract_score += 1

        scores = {
            "classify": sum(1 for kw in cls._CLASSIFY_KW if kw in lower),
            "extract": extract_score,
            "simple_qa": sum(1 for kw in cls._SIMPLE_QA_KW if kw in lower),
            "long_doc": sum(1 for kw in cls._LONG_DOC_KW if kw in lower),
            "code": sum(1 for kw in cls._CODE_KW if kw in lower),
            "math": sum(1 for kw in cls._MATH_KW if kw in lower),
            "complex_reasoning": sum(1 for kw in cls._COMPLEX_REASONING_KW if kw in lower),
            "agent": sum(1 for kw in cls._AGENT_KW if kw in lower),
        }

        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "simple_qa"
        return best_type

    @classmethod
    def estimate_complexity(cls, query: str) -> int:
        """关键词启发式复杂度估计。"""
        score = 3.0
        high_kw = [
            "分析", "优化", "设计", "架构", "重构", "review", "refactor", "debug", "调试",
            "安全", "security", "性能", "performance", "多步骤", "复杂", "系统", "部署",
            "deploy", "explain", "实现", "诊断", "排查", "漏洞", "攻击", "渗透", "加密",
            "认证", "论文", "长文", "报告", "文档", "paper", "report", "阅读理解", "总结",
            "对比", "比较", "区别", "优缺点", "评估", "研究", "协议", "分布式", "事务",
            "一致性", "高可用", "容错", "并发", "异步", "微服务", "容器", "编排", "调度",
            "算法", "algorithm", "推导", "建模", "仿真", "论证", "规划", "策略", "方案",
            "改进", "建议", "容灾", "多活", "选型", "根因", "整改", "影响",
        ]
        low_kw = [
            "多少钱", "价格", "天气", "时间", "翻译", "translate", "什么是", "定义",
            "简单", "快捷", "hello", "hi", "你好", "echo", "重复", "ping", "是谁", "哪里",
            "多少", "多久",
        ]
        for kw in high_kw:
            if kw in query.lower():
                score += 0.3
        for kw in low_kw:
            if kw in query.lower():
                score -= 0.3
        return max(_MIN_COMPLEXITY, min(_MAX_COMPLEXITY, round(score)))


class EmbedTaskClassifier:
    """使用本地嵌入模型的任务类型与复杂度分类器。

    参数：
        model_name: 指定嵌入模型名，默认优先 multilingual MiniLM。
        device: 运行设备，默认自动选择（有 GPU 则用 cuda，否则 cpu）。
        examples_path: 示例向量库 JSON 路径，默认使用插件目录下的 embed_examples.json。
        lazy: 是否后台线程懒加载模型与示例库，默认 True。
        k_neighbors: k-NN 近邻数量。
        sim_power: 相似度加权幂次，越大高相似近邻权重越高。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        examples_path: Optional[str] = None,
        lazy: bool = True,
        k_neighbors: int = _K_NEIGHBORS,
        sim_power: int = _SIM_POWER,
    ) -> None:
        self.model_name = model_name or _PRIMARY_MODEL
        self.fallback_model = _FALLBACK_MODEL
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        if examples_path is None:
            self.examples_path = Path(__file__).resolve().parent / "embed_examples.json"
        else:
            self.examples_path = Path(examples_path)

        self.k_neighbors = max(1, k_neighbors)
        self.sim_power = max(1, sim_power)

        # 运行时状态
        self._model: Any = None
        self._examples: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._error: Optional[Exception] = None
        self._ready_event = threading.Event()
        self._load_lock = threading.Lock()
        self._loaded = False

        if lazy:
            self._load_thread = threading.Thread(target=self._load, daemon=True)
            self._load_thread.start()
        else:
            self._load()

    @staticmethod
    def _has_cuda() -> bool:
        """检查是否有可用 CUDA 设备，且 compute capability 与当前 PyTorch 兼容。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            # 检查 GPU compute capability 是否被当前 PyTorch 版本支持
            try:
                device_cc = torch.cuda.get_device_capability()
                # PyTorch 2.x+ 要求 CC >= 7.5，CC 6.x 的 GTX 1050 等旧卡不兼容
                if device_cc[0] < 7 or (device_cc[0] == 7 and device_cc[1] < 5):
                    logger.info(
                        "GPU compute capability %s.%s 低于 PyTorch 最低要求 7.5，回退到 CPU",
                        device_cc[0], device_cc[1],
                    )
                    return False
            except Exception:
                pass  # 无法检测 CC 时保守放行
            return True
        except Exception:
            return False

    def _load_examples(self) -> list[dict]:
        """从 JSON 文件加载示例向量库。"""
        try:
            with open(self.examples_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            examples = data.get("examples", [])
            if not examples:
                raise ValueError("示例向量库为空")

            # 校验字段并过滤无效项
            valid = []
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                text = ex.get("text")
                task_type = ex.get("task_type")
                complexity = ex.get("complexity")
                if (
                    text
                    and task_type in _TASK_TYPES
                    and isinstance(complexity, (int, float))
                    and _MIN_COMPLEXITY <= complexity <= _MAX_COMPLEXITY
                ):
                    valid.append({
                        "text": str(text),
                        "task_type": task_type,
                        "complexity": int(complexity),
                    })

            if not valid:
                raise ValueError("示例向量库中没有有效条目")

            # 统计各任务类型数量
            counts: dict[str, int] = {}
            for ex in valid:
                counts[ex["task_type"]] = counts.get(ex["task_type"], 0) + 1
            for t in _TASK_TYPES:
                if counts.get(t, 0) < 20:
                    logger.warning("任务类型 %s 示例数量不足 20：%d", t, counts.get(t, 0))

            return valid
        except Exception as exc:
            logger.error("加载示例向量库失败 %s: %s", self.examples_path, exc)
            raise

    def _load_model(self) -> Any:
        """加载本地嵌入模型，主模型失败则回退。

        优先使用本地缓存（local_files_only=True），避免首次启动时因网络探测而阻塞；
        本地无缓存时允许回退模型从 Hugging Face 下载，超时仍失败则抛出异常。
        """
        from sentence_transformers import SentenceTransformer

        models_to_try = [
            (self.model_name, True),   # 主模型：优先本地缓存
            (self.fallback_model, False),  # 回退模型：允许下载
        ]
        last_error: Optional[Exception] = None

        for model_name, local_only in models_to_try:
            try:
                logger.info("正在加载嵌入模型：%s (local_files_only=%s)", model_name, local_only)
                model = SentenceTransformer(
                    model_name,
                    device=self.device,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                logger.info("嵌入模型加载成功：%s", model_name)
                return model
            except Exception as exc:
                last_error = exc
                logger.warning("加载嵌入模型 %s 失败：%s", model_name, exc)

        raise last_error or RuntimeError("无法加载任何嵌入模型")

    def _load(self) -> None:
        """后台加载模型和示例向量库。"""
        try:
            examples = self._load_examples()
            model = self._load_model()

            texts = [ex["text"] for ex in examples]
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            with self._load_lock:
                self._examples = examples
                self._embeddings = np.asarray(embeddings, dtype=np.float32)
                self._model = model
                self._loaded = True

            logger.info(
                "EmbedTaskClassifier 就绪：%d 条示例，维度 %s",
                len(examples),
                self._embeddings.shape,
            )
        except Exception as exc:
            self._error = exc
            logger.error("EmbedTaskClassifier 加载失败，将使用关键词启发式兜底：%s", exc)
        finally:
            self._ready_event.set()

    def wait_ready(self, timeout: Optional[float] = 30.0) -> bool:
        """等待模型加载完成。"""
        return self._ready_event.wait(timeout=timeout or 0.0)

    def is_ready(self) -> bool:
        """检查模型是否已加载成功。"""
        return self._ready_event.is_set() and self._loaded

    def reload_examples(self) -> bool:
        """热更新示例向量库，重新编码示例文本。"""
        with self._load_lock:
            if self._model is None:
                logger.warning("模型未加载，无法热更新示例向量库")
                return False
            try:
                examples = self._load_examples()
                texts = [ex["text"] for ex in examples]
                embeddings = self._model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                self._examples = examples
                self._embeddings = np.asarray(embeddings, dtype=np.float32)
                logger.info("示例向量库热更新完成：%d 条", len(examples))
                return True
            except Exception as exc:
                logger.error("热更新示例向量库失败：%s", exc)
                return False

    def _encode(self, query: str) -> np.ndarray:
        """编码单条查询为归一化向量。"""
        with self._load_lock:
            model = self._model
            if model is None:
                raise RuntimeError("嵌入模型未加载")

        embedding = model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    def _fallback_classify(self, query: str) -> tuple[str, int, float]:
        """关键词启发式兜底分类。"""
        task_type = _KeywordFallback.detect_task_type(query)
        complexity = _KeywordFallback.estimate_complexity(query)
        return task_type, complexity, 0.0

    def classify(self, query: str, timeout: Optional[float] = 30.0) -> tuple[str, int, float]:
        """对单条查询进行分类。

        返回：
            (task_type, complexity_bucket, confidence)
            - task_type: 任务类型字符串
            - complexity_bucket: 1-5 的整数复杂度
            - confidence: 0-1 之间的置信度
        """
        if not query or not query.strip():
            return "simple_qa", 1, 0.0

        # 等待模型加载，超时或失败则降级
        ready = self.wait_ready(timeout=timeout)
        if not ready or not self._loaded or self._embeddings is None:
            logger.debug("嵌入模型未就绪，使用关键词启发式兜底")
            return self._fallback_classify(query)

        try:
            query_vec = self._encode(query)
            with self._load_lock:
                embeddings = self._embeddings
                examples = self._examples

            # 计算余弦相似度（向量已归一化，点积即余弦相似度）
            similarities = (embeddings @ query_vec.T).flatten()

            # 取 top k 近邻
            k = min(self.k_neighbors, len(examples))
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]

            # 加权投票计算任务类型
            type_weights: dict[str, float] = {}
            complexity_weighted = 0.0
            total_weight = 0.0
            for idx in top_k_idx:
                sim = float(similarities[idx])
                weight = sim ** self.sim_power
                ex = examples[int(idx)]
                task_type = ex["task_type"]
                type_weights[task_type] = type_weights.get(task_type, 0.0) + weight
                complexity_weighted += ex["complexity"] * weight
                total_weight += weight

            if total_weight <= 0:
                return self._fallback_classify(query)

            predicted_type = max(type_weights, key=type_weights.get)
            raw_complexity = complexity_weighted / total_weight
            complexity_bucket = int(max(_MIN_COMPLEXITY, min(_MAX_COMPLEXITY, round(raw_complexity))))

            # 置信度：获胜类型的加权得分占比
            confidence = type_weights[predicted_type] / total_weight if total_weight else 0.0
            confidence = float(max(0.0, min(1.0, confidence)))

            return predicted_type, complexity_bucket, confidence
        except Exception as exc:
            logger.warning("语义分类失败，使用关键词启发式兜底：%s", exc)
            return self._fallback_classify(query)
