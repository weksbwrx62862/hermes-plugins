"""统一复杂度评估服务。

综合 model-router 的关键词启发式与 AMA 的多维特征工程，
输出统一的复杂度评分与任务类型，并保留 LLM 二次精修接口。
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional


class ComplexityAssessor:
    """统一任务复杂度评估器。

    输入为单条查询与可选上下文，输出统一格式：
    {
        "score_10": float,      # 1-10 浮点
        "score_5": int,         # 1-5 整数，ceil(score_10 / 2)
        "task_type": str,       # 任务类型
        "confidence": float,    # 0-1
        "features": dict,       # 提取到的特征
    }
    """

    # 高复杂度关键词（与 model-router 对齐）
    HIGH_COMPLEXITY_KEYWORDS: List[str] = [
        "分析", "优化", "设计", "架构", "重构", "review", "refactor",
        "debug", "调试", "安全", "security", "性能", "performance",
        "多步骤", "复杂", "系统", "部署", "deploy", "explain", "实现",
        "诊断", "排查", "漏洞", "攻击", "渗透", "加密", "认证",
        "论文", "长文", "报告", "文档", "paper", "report", "阅读理解",
        "总结", "对比", "比较", "区别", "优缺点", "评估", "研究",
        "协议", "分布式", "事务", "一致性", "高可用", "容错",
        "并发", "异步", "微服务", "容器", "编排", "调度",
        "算法", "algorithm", "推导", "建模", "仿真", "论证",
        "规划", "策略", "方案", "改进", "建议",
    ]

    # 低复杂度关键词（与 model-router 对齐）
    LOW_COMPLEXITY_KEYWORDS: List[str] = [
        "多少钱", "价格", "天气", "时间", "翻译", "translate",
        "什么是", "定义", "简单", "快捷", "hello", "hi", "你好",
        "echo", "重复", "ping",
    ]

    # 任务类型关键词（与 model-router 对齐）
    TASK_TYPE_PATTERNS: Dict[str, List[str]] = {
        "classify": ["分类", "归类", "判断是否", "classify", "categorize", "是真是假"],
        "extract": ["提取", "抽取", "摘录", "extract"],
        "simple_qa": ["多少钱", "价格", "天气", "时间", "翻译", "什么是", "定义", "hello", "hi", "ping"],
        "long_doc": ["文档", "论文", "长文", "报告", "document", "paper", "report", "阅读理解"],
        "code": [
            "代码", "code", "编程", "函数", "class", "def", "import",
            "bug", "修复", "重构", "refactor", "debug", "测试", "test",
            "python", "java", "go", "rust", "js", "ts", "react", "vue",
            "api", "接口", "算法", "algorithm", "sql", "数据库",
            "编译", "部署", "deploy", "docker", "git",
        ],
        "math": ["计算", "数学", "方程", "公式", "calculate", "math", "equation", "证明", "积分", "微分"],
        "complex_reasoning": [
            "分析", "优化", "设计", "架构", "安全", "性能",
            "诊断", "排查", "漏洞", "explain", "analyze",
            "比较", "对比", "区别", "差异", "优缺点",
        ],
        "agent": ["帮我", "执行", "操作", "调用工具", "agent", "工具", "搜索", "联网"],
    }

    # 多维特征关键词（参考 AMA）
    FEATURE_KEYWORDS: Dict[str, List[str]] = {
        "has_explicit_verification": [
            "验证", "检查", "测试", "标准", "规范",
            "verify", "check", "test", "standard", "validate",
        ],
        "needs_parallelism": [
            "同时", "并行", "多个", "分别", "批量",
            "parallel", "concurrent", "multiple", "batch", "simultaneously",
        ],
        "has_roles": [
            "角色", "分工", "负责", "团队",
            "role", "assign", "responsible", "team",
        ],
        "is_event_driven": [
            "事件", "监控", "警报", "实时", "监听",
            "event", "monitor", "alert", "realtime", "listen",
        ],
        "needs_collaboration": [
            "协作", "一起", "共同", "互相",
            "collaborate", "together", "joint", "cooperative",
        ],
        "iterative_potential": [
            "迭代", "改进", "循环", "多次",
            "iterate", "improve", "loop", "refine",
        ],
        "requires_shared_knowledge": [
            "共享", "知识库", "整合", "综合",
            "shared", "knowledge base", "integrate", "synthesize",
        ],
        "reasoning_depth": [
            "为什么", "原因", "推导", "证明", "逻辑", "因果",
            "假设", "权衡", "取舍", "利弊",
            "why", "reason", "derive", "prove", "logic", "tradeoff",
        ],
        "cross_reference": [
            "结合", "整合", "融合", "综合", "兼顾",
            "跨", "对比", "映射", "关联",
            "combine", "integrate", "cross", "relate",
        ],
        "multi_perspective": [
            "多角度", "多维度", "全面分析", "深入分析", "综合分析",
            "利弊", "优缺点", "对比分析", "可行性", "风险评估",
            "multi-angle", "holistic", "comprehensive analysis", "pros and cons",
            "feasibility", "risk assessment", "trade-off",
        ],
    }

    # 输出格式要求关键词
    OUTPUT_FORMAT_KEYWORDS: Dict[str, List[str]] = {
        "requires_table": ["表格", "table", "矩阵", "matrix", "对照表"],
        "requires_report": ["报告", "report", "文档", "document", "设计文档"],
        "requires_code": ["代码", "code", "脚本", "script", "函数", "function"],
        "requires_diagram": ["图", "diagram", "流程图", "架构图", "时序图"],
        "requires_comparison": ["对比", "比较", "compare", "comparison", "优缺点"],
    }

    # 否定前缀（用于过滤被否定的关键词）
    NEGATION_PREFIXES: List[str] = [
        "不", "没", "无", "非", "未", "别", "勿", "莫",
        "not", "no", "non", "un", "dis", "never", "without",
    ]

    # 简单连接词（否定与关键词之间出现这些词时仍视为否定）
    NEGATION_BRIDGE_WORDS: List[str] = [
        "需", "需要", "要", "用", "必", "会", "能", "得", "是",
        "须", "该", "应", "可", "经", "经过", "被", "to", "a", "the",
    ]

    def __init__(
        self,
        enable_llm: bool = False,
        llm_refinement_threshold: float = 0.6,
        llm_refine_fn: Optional[Callable[[Dict[str, Any], str, Optional[Dict[str, Any]]], Dict[str, Any]]] = None,
    ):
        """初始化评估器。

        参数:
            enable_llm: 是否启用 LLM 二次精修（默认关闭，仅保留接口）。
            llm_refinement_threshold: 置信度低于此阈值时可能触发精修。
            llm_refine_fn: 外部注入的 LLM 精修函数，签名 refine_fn(result, query, context)。
        """
        self.enable_llm = enable_llm
        self.llm_refinement_threshold = llm_refinement_threshold
        self.llm_refine_fn = llm_refine_fn

    def assess(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        estimated_tokens: Optional[int] = None,
        enable_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """评估任务复杂度，返回统一格式结果。

        参数:
            query: 用户查询文本。
            context: 可选上下文信息（如历史消息、附加说明）。
            estimated_tokens: 可选的已估算 token 数；未提供时自动估算。
            enable_llm: 是否在本次调用中启用 LLM 精修，默认使用实例配置。

        返回:
            包含 score_10、score_5、task_type、confidence、features 的字典。
        """
        query = query or ""

        features = self._extract_features(query, context, estimated_tokens)
        score_10 = self._calculate_score_10(features, query, context)
        score_5 = math.ceil(score_10 / 2)
        task_type = self._detect_task_type(query)
        confidence = self._calculate_confidence(features, score_10, query)

        result: Dict[str, Any] = {
            "score_10": score_10,
            "score_5": score_5,
            "task_type": task_type,
            "confidence": confidence,
            "features": features,
        }

        # LLM 二次精修接口：默认不调用
        use_llm = self.enable_llm if enable_llm is None else enable_llm
        if use_llm and confidence < self.llm_refinement_threshold:
            result = self._refine_with_llm(result, query, context)

        return result

    def _extract_features(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        estimated_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """从查询与上下文中抽取多维特征。"""
        combined = (query + " " + str(context or "")).lower()

        features: Dict[str, Any] = {}

        # 关键词特征（带否定过滤）
        for feature_name, keywords in self.FEATURE_KEYWORDS.items():
            features[feature_name] = self._keyword_match(combined, keywords)

        # 复杂度关键词命中数（避免与输出格式要求重复计数）
        output_format_keywords = {
            kw for keywords in self.OUTPUT_FORMAT_KEYWORDS.values() for kw in keywords
        }
        high_hits = sum(
            1
            for kw in self.HIGH_COMPLEXITY_KEYWORDS
            if kw not in output_format_keywords and kw in query.lower()
        )
        low_hits = sum(1 for kw in self.LOW_COMPLEXITY_KEYWORDS if kw in query.lower())
        features["high_complexity_keyword_hits"] = high_hits
        features["low_complexity_keyword_hits"] = low_hits

        # 输出格式特征
        for feature_name, keywords in self.OUTPUT_FORMAT_KEYWORDS.items():
            features[feature_name] = self._keyword_match(combined, keywords)

        # 长度与上下文特征
        features["task_length"] = len(query)
        features["context_size"] = len(str(context or ""))
        features["estimated_tokens"] = estimated_tokens or self._estimate_text_tokens(query)
        features["subtask_count"] = self._count_subtasks(query)

        return features

    def _keyword_match(self, text: str, keywords: List[str]) -> bool:
        """关键词匹配，带否定前缀过滤。"""
        for kw in keywords:
            idx = text.find(kw)
            while idx != -1:
                if idx == 0 or not self._is_negated(text, idx):
                    return True
                idx = text.find(kw, idx + len(kw))
        return False

    def _is_negated(self, text: str, idx: int) -> bool:
        """检查关键词是否被前方否定前缀修饰。"""
        prefix_start = max(0, idx - 8)
        prefix = text[prefix_start:idx]
        for neg in self.NEGATION_PREFIXES:
            neg_idx = prefix.find(neg)
            if neg_idx == -1:
                continue
            gap = prefix[neg_idx + len(neg):].strip()
            if not gap:
                return True
            if gap in self.NEGATION_BRIDGE_WORDS:
                return True
        return False

    def _count_subtasks(self, text: str) -> int:
        """检测明确子任务数量（列表项或中文顿号编号）。"""
        numbered_items = len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]\s|[•\-*]\s)", text))
        cn_numbered = len(re.findall(r"\d+、", text))
        return max(numbered_items, cn_numbered)

    def _estimate_text_tokens(self, text: str) -> int:
        """估算文本 token 数（无 tiktoken 时按字符混合估算）。"""
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
        cn_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other_chars = len(text) - cn_chars
        return int(cn_chars * 1.5 + other_chars * 0.5)

    def _calculate_score_10(
        self,
        features: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """计算 1-10 浮点复杂度评分。"""
        # 基础分：与 model-router 默认复杂度 3 对应，使简单查询落在 1-3 区间
        score = 2.5

        # 高/低复杂度关键词（model-router 风格）
        score += min(features.get("high_complexity_keyword_hits", 0) * 0.3, 1.5)
        score -= min(features.get("low_complexity_keyword_hits", 0) * 0.3, 1.0)

        # AMA 风格显性特征
        if features.get("needs_parallelism"):
            score += 2.0
        if features.get("has_roles"):
            score += 1.5
        if features.get("is_event_driven"):
            score += 1.0
        if features.get("needs_collaboration"):
            score += 1.5
        if features.get("requires_shared_knowledge"):
            score += 1.0
        if features.get("has_explicit_verification"):
            score += 0.5
        if features.get("iterative_potential"):
            score += 1.0
        if features.get("reasoning_depth"):
            score += 0.8
        if features.get("cross_reference"):
            score += 1.2
        if features.get("multi_perspective"):
            score += 1.5

        # 输出格式要求
        if features.get("requires_report"):
            score += 1.0
        if features.get("requires_table") or features.get("requires_comparison"):
            score += 0.5
        if features.get("requires_diagram"):
            score += 0.5

        # 子任务数量
        subtask_count = features.get("subtask_count", 0)
        if subtask_count >= 3:
            score += 1.5
        elif subtask_count >= 1:
            score += 0.5

        # 任务长度
        task_len = features.get("task_length", 0)
        if task_len > 100:
            score += 0.5
        if task_len > 300:
            score += 1.0
        if task_len > 600:
            score += 1.0
        if task_len > 1000:
            score += 1.0

        # 历史上下文 / token 规模
        estimated_tokens = features.get("estimated_tokens", 0)
        if estimated_tokens > 200_000:
            score += 2.0
        elif estimated_tokens > 50_000:
            score += 1.0
        elif estimated_tokens > 10_000:
            score += 0.5

        return max(1.0, min(10.0, score))

    def _detect_task_type(self, query: str) -> str:
        """检测任务类型（与 model-router 对齐）。"""
        lower = query.lower()
        scores = {
            task_type: sum(1 for kw in keywords if kw in lower)
            for task_type, keywords in self.TASK_TYPE_PATTERNS.items()
        }
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "simple_qa"
        return best_type

    def _calculate_confidence(
        self,
        features: Dict[str, Any],
        score_10: float,
        query: str,
    ) -> float:
        """计算置信度：信号越强、越一致，置信度越高。"""
        confidence = 0.85

        high_hits = features.get("high_complexity_keyword_hits", 0)
        low_hits = features.get("low_complexity_keyword_hits", 0)

        # 同时存在高低复杂度信号，置信度降低
        if high_hits > 0 and low_hits > 0:
            confidence -= 0.2

        # 过短输入信息不足
        if len(query) < 10:
            confidence -= 0.15

        # 活跃布尔特征数量
        bool_features = [
            k for k, v in features.items()
            if isinstance(v, bool) and v and not k.startswith("requires_")
        ]
        if len(bool_features) < 2:
            confidence -= 0.1
        if len(bool_features) >= 5:
            confidence += 0.1

        # 超大上下文增加确定性
        if features.get("estimated_tokens", 0) > 50_000:
            confidence += 0.1

        return max(0.0, min(1.0, round(confidence, 2)))

    def _refine_with_llm(
        self,
        result: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """LLM 二次精修接口，默认不执行实际调用。

        若外部注入 llm_refine_fn，则使用之；否则返回原结果并标记 llm_refined=false。
        """
        if self.llm_refine_fn is not None:
            try:
                refined = self.llm_refine_fn(result, query, context)
                if isinstance(refined, dict) and "score_10" in refined:
                    refined["score_5"] = math.ceil(refined["score_10"] / 2)
                    refined["llm_refined"] = True
                    return refined
            except Exception:
                pass

        refined = dict(result)
        refined["llm_refined"] = False
        return refined
