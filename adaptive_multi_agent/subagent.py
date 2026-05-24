from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class SubagentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class AgentMode(Enum):
    GENERATOR_VERIFIER = "generator_verifier"
    ORCHESTRATOR_SUBAGENT = "orchestrator_subagent"
    AGENT_TEAMS = "agent_teams"
    MESSAGE_BUS = "message_bus"
    SHARED_STATE = "shared_state"

    @property
    def cn(self) -> str:
        return _MODE_CN.get(self, self.value)


_MODE_CN: Dict[AgentMode, str] = {
    AgentMode.GENERATOR_VERIFIER: "生成-验证",
    AgentMode.ORCHESTRATOR_SUBAGENT: "编排-子代理",
    AgentMode.AGENT_TEAMS: "团队协作",
    AgentMode.MESSAGE_BUS: "事件驱动",
    AgentMode.SHARED_STATE: "共享状态",
}

# 模式名短标签（TS 日志用，取前2字）
MODE_CN_SHORT: Dict[str, str] = {
    "orchestrator_subagent": "编排",
    "generator_verifier": "生成",
    "agent_teams": "团队",
    "message_bus": "事件",
    "shared_state": "共享",
}

# 任务类型中文名
TASK_TYPE_CN: Dict[str, str] = {
    "simple": "简单任务",
    "moderate": "中等任务",
    "complex": "复杂任务",
    "creative": "创意任务",
    "analysis": "分析任务",
    "coding": "编码任务",
    "research": "研究任务",
    "multi_step": "多步骤任务",
    "default": "默认任务",
}


@dataclass
class SubagentConfig:
    name: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)
    model: str = "default"
    max_turns: int = 10
    timeout_seconds: int = 300
    priority: int = 0
    max_retries: int = 3


_MODE_PRESETS: Dict[AgentMode, SubagentConfig] = {
    AgentMode.GENERATOR_VERIFIER: SubagentConfig(
        name="generator_verifier",
        description="生成-验证模式：先生成结果，再独立验证，迭代改进直至通过",
        system_prompt=(
            "你是一个生成-验证代理。你的职责是："
            "1. 生成高质量的初始结果；"
            "2. 对生成结果进行严格验证，检查正确性、完整性和一致性；"
            "3. 根据验证反馈迭代改进，直到结果满足要求。"
            "生成和验证必须独立进行，确保验证的客观性。"
        ),
        timeout_seconds=300,
        max_turns=10,
        max_retries=3,
    ),
    AgentMode.ORCHESTRATOR_SUBAGENT: SubagentConfig(
        name="orchestrator_subagent",
        description="协调-子代理模式：将复杂任务分解为子任务，委派给子代理执行并综合结果",
        system_prompt=(
            "你是一个协调-子代理。你的职责是："
            "1. 分析复杂任务，将其分解为可独立执行的子任务；"
            "2. 为每个子任务明确输入、输出和验收标准；"
            "3. 按依赖关系有序委派子任务给子代理执行；"
            "4. 收集子任务结果，综合生成最终输出。"
            "分解要合理，子任务粒度适中，避免过细或过粗。"
        ),
        timeout_seconds=600,
        max_turns=15,
        max_retries=3,
    ),
    AgentMode.AGENT_TEAMS: SubagentConfig(
        name="agent_teams",
        description="团队协作模式：多个角色分工协作，各司其职完成复杂任务",
        system_prompt=(
            "你是一个团队协作代理。你的职责是："
            "1. 明确团队中各角色的职责和分工；"
            "2. 按角色依次执行任务，每个角色专注于自己的领域；"
            "3. 角色之间传递工作成果，形成协作流水线；"
            "4. 确保各角色输出衔接一致，最终产出高质量结果。"
            "角色分工要清晰，协作流程要高效。"
        ),
        timeout_seconds=300,
        max_turns=10,
        max_retries=3,
    ),
    AgentMode.MESSAGE_BUS: SubagentConfig(
        name="message_bus",
        description="事件驱动模式：通过事件总线协调订阅者，实现松耦合的异步处理",
        system_prompt=(
            "你是一个事件驱动代理。你的职责是："
            "1. 监听事件总线上的事件消息；"
            "2. 根据事件类型触发相应的处理逻辑；"
            "3. 处理完成后发布新事件，驱动下游流程；"
            "4. 确保事件处理的顺序性和幂等性。"
            "事件处理要快速响应，避免阻塞事件流。"
        ),
        timeout_seconds=300,
        max_turns=10,
        max_retries=3,
    ),
    AgentMode.SHARED_STATE: SubagentConfig(
        name="shared_state",
        description="共享状态模式：多个代理通过共享状态协作，逐步推进任务直至收敛",
        system_prompt=(
            "你是一个共享状态协作代理。你的职责是："
            "1. 读取并理解当前共享状态中的所有信息；"
            "2. 根据自身角色更新共享状态中的对应部分；"
            "3. 确保状态更新的一致性和完整性；"
            "4. 持续迭代直到任务目标达成并验证通过。"
            "状态更新要原子化，避免覆盖他人的工作成果。"
        ),
        timeout_seconds=300,
        max_turns=10,
        max_retries=3,
    ),
}


@dataclass
class SubagentResult:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: SubagentStatus = SubagentStatus.PENDING
    error_category: Optional[str] = None
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    elapsed_seconds: float = 0.0
    retries_attempted: int = 0
    created_at: float = field(default_factory=time.time)


class RetryPolicy:
    # 可重试的错误类别
    RETRYABLE_ERRORS = {"rate_limit", "network", "context_overflow", "json_parse_error", "internal_error", "timeout", "unknown"}

    # 各错误类别的最大重试次数
    max_retries_by_category: Dict[str, int] = {
        "rate_limit": 3,
        "network": 3,
        "context_overflow": 1,
        "json_parse_error": 2,
        "internal_error": 1,
        "timeout": 2,
        "unknown": 1,
    }

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter_range: float = 0.5,
        max_retries: int = 3,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter_range = jitter_range
        self.max_retries = max_retries

    def should_retry(self, error_category: str, current_retry: int = 0) -> bool:
        if error_category not in self.RETRYABLE_ERRORS:
            return False
        category_max = self.max_retries_by_category.get(error_category, self.max_retries)
        return current_retry < category_max

    def get_wait_time(self, attempt: int) -> float:
        # 指数退避
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        # 添加 jitter
        jitter = random.uniform(0, self.jitter_range * delay)
        return delay + jitter

    def get_retry_hint(self, error_category: str) -> str:
        hints = {
            "json_parse_error": "请确保输出为严格的 JSON 格式，不要包含其他文本",
            "timeout": "请简化输出，减少不必要的详细描述",
        }
        return hints.get(error_category, "")

    def classify_error(self, exception: Exception) -> str:
        msg = str(exception).lower()
        exc_type = type(exception).__name__.lower()

        # 速率限制类错误
        rate_limit_keywords = ["rate", "limit", "429", "throttl", "too many requests"]
        if any(kw in msg for kw in rate_limit_keywords):
            return "rate_limit"

        # 网络类错误
        network_keywords = ["timeout", "connection", "network", "unreachable", "refused", "dns"]
        network_exceptions = ["timeouterror", "connectionerror", "connectionreseterror"]
        if any(kw in msg for kw in network_keywords) or exc_type in network_exceptions:
            return "network"

        # 上下文溢出类错误
        overflow_keywords = ["context", "token", "overflow", "too long", "maximum length", "exceeds"]
        if any(kw in msg for kw in overflow_keywords):
            return "context_overflow"

        # JSON 解析类错误
        json_keywords = ["json", "decode", "parse", "invalid json", "unexpected token", "malformed"]
        if any(kw in msg for kw in json_keywords):
            return "json_parse_error"

        # 内部错误（Python 异常标记）
        internal_exceptions = ["typeerror", "attributeerror", "valueerror", "keyerror", "indexerror"]
        if exc_type in internal_exceptions:
            return "internal_error"

        # 业务超时类错误
        timeout_keywords = ["timed out", "deadline", "execution expired"]
        if any(kw in msg for kw in timeout_keywords):
            return "timeout"

        return "unknown"


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_permitted = True
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN:
                if self._last_failure_time is None:
                    return False
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_permitted = True
                    return True
                return False
            # HALF_OPEN 状态仅允许一次探测请求
            if self._half_open_permitted:
                self._half_open_permitted = False
                return True
            return False

    def _check_half_open(self) -> bool:
        """检查是否可以从 OPEN 转换到 HALF_OPEN（仅内部使用）"""
        with self._lock:
            if self._state == self.OPEN and self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_permitted = True
                    return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED
            self._half_open_permitted = True

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = self.OPEN

    def get_state(self) -> str:
        with self._lock:
            # 检查是否应从 OPEN 转为 HALF_OPEN
            if self._state == self.OPEN and self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state


class TaskResultStore:
    def __init__(self):
        self._store: Dict[str, SubagentResult] = {}
        self._cancelled: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def put(self, task_id: str, result: SubagentResult) -> None:
        with self._lock:
            self._store[task_id] = result

    def get(self, task_id: str) -> Optional[SubagentResult]:
        with self._lock:
            return self._store.get(task_id)

    def cleanup(self, max_age_seconds: float) -> None:
        now = time.time()
        with self._lock:
            expired_ids = [
                tid for tid, result in self._store.items()
                if now - result.created_at > max_age_seconds
            ]
            for tid in expired_ids:
                del self._store[tid]
                self._cancelled.pop(tid, None)

    def request_cancel(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._store:
                return False
            if self._cancelled.get(task_id, False):
                return False
            result = self._store[task_id]
            if result.status in (SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED):
                return False
            self._cancelled[task_id] = True
            result.cancel_event.set()
            return True

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            return self._cancelled.get(task_id, False)


@dataclass
class SubtaskItem:
    """结构化子任务定义"""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    expected_output: str = ""

    def validate(self) -> List[str]:
        """校验子任务字段完整性"""
        errors = []
        if not self.id:
            errors.append("子任务缺少 id")
        if not self.description:
            errors.append(f"子任务 {self.id} 缺少 description")
        return errors


def validate_subtask_dag(subtasks: List[SubtaskItem]) -> List[str]:
    """校验子任务 DAG 合法性，返回错误列表"""
    errors = []
    ids = {st.id for st in subtasks}

    for st in subtasks:
        for dep in st.dependencies:
            if dep not in ids:
                errors.append(f"子任务 {st.id} 依赖不存在的子任务 {dep}")

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {st.id: WHITE for st in subtasks}
    dep_map = {st.id: st.dependencies for st in subtasks}

    def has_cycle(node):
        color[node] = GRAY
        for dep in dep_map.get(node, []):
            if color.get(dep) == GRAY:
                return True
            if color.get(dep, WHITE) == WHITE and has_cycle(dep):
                return True
        color[node] = BLACK
        return False

    for st in subtasks:
        if color[st.id] == WHITE:
            if has_cycle(st.id):
                errors.append("子任务存在循环依赖")
                break

    has_root = any(len(st.dependencies) == 0 for st in subtasks)
    if not has_root and subtasks:
        errors.append("子任务 DAG 没有入度为 0 的起点")

    return errors


_TEMPLATE_SUBTASKS = {
    "code_generation": [
        SubtaskItem(id="1", description="分析需求并设计技术方案", dependencies=[], expected_output="技术方案文档"),
        SubtaskItem(id="2", description="实现核心功能代码", dependencies=["1"], expected_output="可运行的代码"),
        SubtaskItem(id="3", description="编写测试并验证功能正确性", dependencies=["2"], expected_output="测试报告"),
    ],
    "research": [
        SubtaskItem(id="1", description="收集和整理相关资料", dependencies=[], expected_output="资料摘要"),
        SubtaskItem(id="2", description="分析资料并提取关键信息", dependencies=["1"], expected_output="分析报告"),
        SubtaskItem(id="3", description="综合分析结果形成结论", dependencies=["2"], expected_output="结论文档"),
    ],
    "fact_checking": [
        SubtaskItem(id="1", description="查找原始信息来源", dependencies=[], expected_output="信息来源列表"),
        SubtaskItem(id="2", description="交叉验证信息准确性", dependencies=["1"], expected_output="验证结果"),
    ],
    "software_dev": [
        SubtaskItem(id="1", description="需求分析与架构设计", dependencies=[], expected_output="设计文档"),
        SubtaskItem(id="2", description="编码实现", dependencies=["1"], expected_output="源代码"),
        SubtaskItem(id="3", description="测试与部署", dependencies=["2"], expected_output="部署结果"),
    ],
    "event_driven": [
        SubtaskItem(id="1", description="定义事件流和处理流程", dependencies=[], expected_output="流程定义"),
        SubtaskItem(id="2", description="实现事件处理器", dependencies=["1"], expected_output="处理器代码"),
    ],
}

_DEFAULT_SUBTASKS = [
    SubtaskItem(id="1", description="调研任务背景和相关资料", dependencies=[]),
    SubtaskItem(id="2", description="分析任务需求并制定方案", dependencies=["1"]),
    SubtaskItem(id="3", description="执行方案并输出结果", dependencies=["2"]),
]


def get_template_subtasks(task_type: str) -> List[SubtaskItem]:
    """根据任务类型获取模板化子任务"""
    return list(_TEMPLATE_SUBTASKS.get(task_type, _DEFAULT_SUBTASKS))


class SubagentRegistry:
    def __init__(self):
        self._configs: Dict[AgentMode, SubagentConfig] = dict(_MODE_PRESETS)
        self._plugins: Dict[AgentMode, ModePlugin] = {}
        self._lock = threading.Lock()

    def register(self, mode: AgentMode, config: SubagentConfig) -> None:
        with self._lock:
            self._configs[mode] = config

    def get(self, mode: AgentMode) -> SubagentConfig:
        with self._lock:
            if mode not in self._configs:
                raise KeyError(f"未注册的 AgentMode: {mode}")
            return self._configs[mode]

    def get_all(self) -> Dict[AgentMode, SubagentConfig]:
        with self._lock:
            return dict(self._configs)

    def register_plugin(self, plugin: ModePlugin) -> None:
        """注册模式插件"""
        if not isinstance(plugin, ModePlugin):
            raise TypeError(f"{type(plugin).__name__} 未实现 ModePlugin 协议")
        if not plugin.validate_config():
            raise ValueError(f"插件 {plugin.mode.value} 配置校验失败")
        self._plugins[plugin.mode] = plugin

    def get_plugin(self, mode: AgentMode) -> Optional[ModePlugin]:
        """获取已注册的模式插件"""
        return self._plugins.get(mode)

    def list_plugins(self) -> List[Dict[str, str]]:
        """列出所有已注册的模式插件"""
        return [{"mode": m.value, "description": p.describe()} for m, p in self._plugins.items()]


@runtime_checkable
class ModePlugin(Protocol):
    """模式插件协议，新模式只需实现此协议即可注册到引擎"""

    @property
    def mode(self) -> "AgentMode":
        """返回此插件对应的执行模式"""
        ...

    def register_graph(self, graph: "ModeGraph") -> None:
        """将模式的节点和边注册到 ModeGraph 中"""
        ...

    def validate_config(self) -> bool:
        """校验插件配置是否合法"""
        ...

    def describe(self) -> str:
        """返回模式描述"""
        ...


class PluginRegistry:
    """模式插件注册表"""
    def __init__(self):
        self._plugins: Dict[AgentMode, ModePlugin] = {}
        self._lock = threading.Lock()

    def register_plugin(self, plugin: ModePlugin) -> None:
        """注册模式插件"""
        with self._lock:
            self._plugins[plugin.mode] = plugin

    def get_plugin(self, mode: AgentMode) -> Optional[ModePlugin]:
        """获取模式插件"""
        with self._lock:
            return self._plugins.get(mode)

    def has_plugin(self, mode: AgentMode) -> bool:
        """检查模式是否有插件"""
        with self._lock:
            return mode in self._plugins

    def get_all_plugins(self) -> Dict[AgentMode, ModePlugin]:
        """获取所有已注册插件"""
        with self._lock:
            return dict(self._plugins)

    def unregister_plugin(self, mode: AgentMode) -> None:
        """注销模式插件"""
        with self._lock:
            self._plugins.pop(mode, None)
