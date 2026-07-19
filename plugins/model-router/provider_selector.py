"""ProviderSelector：模型与 Provider 双层路由中的 Provider 选择器。

核心职责：
- 针对同一模型，在多个 Provider deployment 之间做综合评分；
- 输出选中的 deployment 与有序 Provider 级降级链；
- 兼容当前 config.yaml 的 flat 模型配置（deployment 缺 provider 时从模型名推断）。
"""

from __future__ import annotations

from typing import Any, Optional


# 默认评分权重： latency / cost / quality / quota，数值越大越重要。
_DEFAULT_WEIGHTS = {
    "latency": 1.0,
    "cost": 1.0,
    "quality": 1.0,
    "quota": 1.0,
}

# Region 偏好默认加分值，需足以抵消小幅延迟/成本劣势。
_DEFAULT_REGION_BONUS = 0.5

# 模型名到 provider 的推断映射，用于兼容 flat 配置。
_PROVIDER_BY_MODEL_PREFIX: dict[tuple[str, ...], str] = {
    ("qwen", "glm", "kimi", "minimax"): "dashscope",
    ("step",): "stepfun",
    ("sensenova",): "sensenova",
    ("deepseek",): "deepseek",
    ("agnes",): "custom:agnes",
    ("llama", "nemotron", "gpt-oss", "nvidia"): "nvidia-nim",
}


def _infer_provider(model: str) -> str:
    """从模型名推断 provider，用于 deployment 信息不完整时的兜底。"""
    lower = (model or "").lower()
    for prefixes, provider in _PROVIDER_BY_MODEL_PREFIX.items():
        if any(prefix in lower for prefix in prefixes):
            return provider
    return "unknown"


def _normalize(value: float, max_value: float) -> float:
    """归一化到 [0, 1]，避免除零。"""
    if max_value <= 0:
        return 1.0
    return max(0.0, min(1.0, value / max_value))


class ProviderSelector:
    """Provider 选择器。

    评分维度：
      - latency（越低越好）
      - cost（越低越好）
      - quality（越高越好）
      - quota remaining（越高越好）
      - region 偏好（可配置加分）
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        region_bonus: float = _DEFAULT_REGION_BONUS,
    ) -> None:
        """初始化选择器。

        Args:
            weights: 各维度权重，未指定维度使用默认值。
            region_bonus: Region 偏好加分值。
        """
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.region_bonus = region_bonus

    def select(
        self,
        model: str,
        deployments: list[dict[str, Any]],
        *,
        region_preferences: Optional[list[str]] = None,
    ) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
        """为指定模型选择一个最优 Provider 并生成降级链。

        Args:
            model: 模型名称。
            deployments: 候选 deployment 列表，每项至少包含 provider/key/base_url
                等字段；缺少 provider 时会根据 model 推断。
            region_preferences: 偏好的 region 列表，命中时会获得额外加分。

        Returns:
            (selected_deployment, fallback_chain)
            - selected_deployment: 选中的 deployment；无可选时返回 None。
            - fallback_chain: 按综合评分排序的降级链（不含已选中项），
              配额耗尽的项会带上 exhausted=True 标记。
        """
        if not deployments:
            return None, []

        region_preferences = region_preferences or []

        # 1. 补齐 deployment 信息并推断 provider（兼容 flat 配置）
        normalized = []
        for dep in deployments:
            d = dict(dep)
            d.setdefault("model", model)
            if not d.get("provider"):
                d["provider"] = _infer_provider(d.get("model", model))
            d.setdefault("latency_ms", 0)
            d.setdefault("cost", 0)
            d.setdefault("quality", 0)
            d.setdefault("rpm_limit", 0)
            d.setdefault("tpm_limit", 0)
            d.setdefault("rpm_used", 0)
            d.setdefault("tpm_used", 0)
            d.setdefault("region", "")
            normalized.append(d)

        # 2. 计算用于归一化的最大值
        max_latency = max(d["latency_ms"] for d in normalized) or 1.0
        max_cost = max(d["cost"] for d in normalized) or 1.0
        max_quality = max(d["quality"] for d in normalized) or 1.0
        max_rpm = max(d["rpm_limit"] for d in normalized) or 1.0
        max_tpm = max(d["tpm_limit"] for d in normalized) or 1.0

        # 3. 逐项评分
        scored: list[tuple[dict[str, Any], float]] = []
        for d in normalized:
            latency_score = 1.0 - _normalize(d["latency_ms"], max_latency)
            cost_score = 1.0 - _normalize(d["cost"], max_cost)
            quality_score = _normalize(d["quality"], max_quality)

            remaining_rpm = max(0, d["rpm_limit"] - d["rpm_used"])
            remaining_tpm = max(0, d["tpm_limit"] - d["tpm_used"])
            quota_rpm = _normalize(remaining_rpm, max_rpm)
            quota_tpm = _normalize(remaining_tpm, max_tpm)
            quota_score = min(quota_rpm, quota_tpm)

            # 任意一种配额耗尽即视为该 deployment 不可用
            exhausted = remaining_rpm <= 0 or remaining_tpm <= 0

            region_bonus = (
                self.region_bonus
                if d["region"] and d["region"] in region_preferences
                else 0.0
            )

            score = (
                self.weights["latency"] * latency_score
                + self.weights["cost"] * cost_score
                + self.weights["quality"] * quality_score
                + self.weights["quota"] * quota_score
                + region_bonus
            )

            d["_score"] = score
            d["exhausted"] = exhausted
            scored.append((d, score))

        # 4. 按评分排序
        scored.sort(key=lambda item: item[1], reverse=True)

        # 5. 选择：优先使用未耗尽的 deployment
        viable = [d for d, _ in scored if not d["exhausted"]]
        selected = viable[0] if viable else scored[0][0]

        # 6. 构建降级链（排除已选中项，保留评分排序）
        fallback_chain = [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for d, _ in scored
            if d is not selected
        ]

        selected_clean = {k: v for k, v in selected.items() if not k.startswith("_")}
        return selected_clean, fallback_chain
