"""
协同进化验证 — Surrogate Verifier

借鉴 EvoSkills (arxiv 2604.01687)，引入轻量级评测代理模型，
与 Ground Truth（完整评测）协同进化校准，减少评测偏差。

核心思路：
  1. Surrogate 用轻量级模型快速预测技能分数
  2. 定期与 GT（完整评测）对比，校准偏差
  3. 当偏差超过阈值时自动重训练（调整 prompt / 清空校准）
  4. predict() 应用校准偏移，predict_raw() 不应用
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from self_evolution.core.evolution_provider import EvalExample
from self_evolution.core.fitness import evaluate_skill, _get_active_model

logger = logging.getLogger(__name__)

_CALIBRATION_STORE = Path.home() / ".hermes" / ".surrogate_calibration.json"


@dataclass
class CalibrationRecord:
    """校准记录：surrogate 预测 vs GT 实际。"""
    surrogate_score: float
    gt_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class VerifierStats:
    """验证器统计信息。"""
    total_predictions: int = 0
    total_calibrations: int = 0
    mean_gap: float = 0.0
    last_gap: float = 0.0
    calibration_history: list[CalibrationRecord] = field(default_factory=list)

    @property
    def fidelity(self) -> float:
        """评测保真度：1 - mean_gap。"""
        return max(0.0, 1.0 - self.mean_gap)


class SurrogateVerifier:
    """轻量级评测代理模型，与 GT 协同进化校准。"""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        gap_threshold: float = 0.15,
        calibration_interval: int = 5,
    ):
        self._client = client or OpenAI()
        self._model = model or _get_active_model()
        self._gap_threshold = gap_threshold
        self._calibration_interval = calibration_interval
        self._stats = VerifierStats()
        self._calibration_offset = 0.0
        self._load_calibration()

    def predict(self, skill_text: str, example: EvalExample) -> float:
        """用 surrogate 模型预测技能分数（快速评估，应用校准偏移）。"""
        fitness = evaluate_skill(skill_text, example, self._client, self._model)
        raw = fitness.composite
        calibrated = max(0.0, min(1.0, raw + self._calibration_offset))
        self._stats.total_predictions += 1
        logger.info(
            f"[surrogate.predict] raw={raw:.3f} | offset={self._calibration_offset:.3f} | "
            f"calibrated={calibrated:.3f} | total_predictions={self._stats.total_predictions}"
        )
        return calibrated

    def predict_raw(self, skill_text: str, example: EvalExample) -> float:
        """原始 surrogate 预测（不应用校准偏移）。"""
        fitness = evaluate_skill(skill_text, example, self._client, self._model)
        return fitness.composite

    def calibrate(self, skill_text: str, example: EvalExample, gt_score: float) -> CalibrationRecord:
        """用 GT 分数校准 surrogate。"""
        surrogate_score = self.predict_raw(skill_text, example)
        gap = abs(surrogate_score - gt_score)
        record = CalibrationRecord(surrogate_score=surrogate_score, gt_score=gt_score)

        self._stats.calibration_history.append(record)
        self._stats.total_calibrations += 1
        self._stats.last_gap = gap

        gaps = [abs(r.surrogate_score - r.gt_score) for r in self._stats.calibration_history[-20:]]
        self._stats.mean_gap = sum(gaps) / len(gaps)

        self._calibration_offset = gt_score - surrogate_score

        if gap > self._gap_threshold:
            logger.info(f"[surrogate] gap={gap:.3f} > threshold={self._gap_threshold}, 需要重训练")
        else:
            logger.info(f"[surrogate] 校准完成: gap={gap:.3f}, offset={self._calibration_offset:.3f}")

        self._save_calibration()
        return record

    def needs_retrain(self) -> bool:
        """判断是否需要重训练。"""
        return self._stats.last_gap > self._gap_threshold

    def retrain(self) -> None:
        """重训练 surrogate：清空校准历史，重置偏移。"""
        logger.info("[surrogate] 开始重训练，清空校准历史")
        self._stats.calibration_history.clear()
        self._calibration_offset = 0.0
        self._stats.mean_gap = 0.0
        self._stats.last_gap = 0.0
        self._save_calibration()

    @property
    def stats(self) -> VerifierStats:
        return self._stats

    def _load_calibration(self) -> None:
        """加载持久化的校准数据。"""
        if not _CALIBRATION_STORE.exists():
            logger.info("[surrogate._load_calibration] 无持久化校准数据，使用默认值")
            return
        try:
            data = json.loads(_CALIBRATION_STORE.read_text())
            self._calibration_offset = data.get("calibration_offset", 0.0)
            self._stats.mean_gap = data.get("mean_gap", 0.0)
            self._stats.total_calibrations = data.get("total_calibrations", 0)
            self._stats.total_predictions = data.get("total_predictions", 0)
            logger.info(
                f"[surrogate._load_calibration] 加载校准数据 | "
                f"offset={self._calibration_offset:.3f} | mean_gap={self._stats.mean_gap:.3f} | "
                f"calibrations={self._stats.total_calibrations} | predictions={self._stats.total_predictions}"
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("[surrogate._load_calibration] 校准数据 JSON 解析失败，使用默认值")

    def _save_calibration(self) -> None:
        """持久化校准数据。"""
        try:
            _CALIBRATION_STORE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "calibration_offset": self._calibration_offset,
                "mean_gap": self._stats.mean_gap,
                "total_calibrations": self._stats.total_calibrations,
                "total_predictions": self._stats.total_predictions,
            }
            _CALIBRATION_STORE.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug(f"[surrogate] 保存校准数据失败: {exc}")
