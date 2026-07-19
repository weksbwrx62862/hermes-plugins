#!/usr/bin/env python3
"""
技能路由器评估脚本

功能：
1. 评估当前模型准确率（纯向量检索）
2. 混合检索评估：纯向量 / 纯 BM25 / 混合检索对比
3. 置信度分布统计
4. 错误案例分析与改进建议

注：BM25 与混合检索的实现均复用插件根目录下的 bm25_searcher / hybrid_searcher 模块，
    确保离线评估与线上完全一致。
"""

import importlib.util
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import sqlite3
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.path.expanduser(
    os.environ.get(
        "SKILL_ROUTER_MODEL_PATH",
        "~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v7",
    )
)
DB_PATH = os.path.expanduser("~/.hermes/skill_index.db")
TRAINING_DATA = os.path.expanduser("~/.hermes/skills/devops/skill-router-scalable/training_data.json")

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# 混合检索分数校准策略：minmax / sigmoid / zscore
HYBRID_CALIBRATION = "minmax"


def _load_local_module(name: str, rel_path: str) -> Any:
    """从插件目录动态加载本地子模块"""
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(plugin_dir, rel_path)
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {name}，spec 或 loader 为 None")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 复用线上 BM25Searcher 与 HybridSearcher
_bm25_searcher_mod = _load_local_module("bm25_searcher", "bm25_searcher.py")
_hybrid_searcher_mod = _load_local_module("hybrid_searcher", "hybrid_searcher.py")
BM25Searcher = _bm25_searcher_mod.BM25Searcher
HybridSearcher = _hybrid_searcher_mod.HybridSearcher


def load_model() -> SentenceTransformer:
    """加载嵌入模型"""
    return SentenceTransformer(MODEL_PATH, device="cpu")


def load_skills() -> Dict[str, Dict[str, str]]:
    """从数据库加载技能索引

    evaluate.py 使用数据库中的 body 列，但 BM25Searcher 期望 body_text 键，
    因此在这里做字段映射。
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, description, body FROM skills")
    skills: Dict[str, Dict[str, str]] = {}
    for name, desc, body in cursor.fetchall():
        skills[name] = {"description": desc or "", "body_text": body or ""}
    conn.close()
    return skills


def evaluate(model: SentenceTransformer, skills: Dict[str, Dict[str, str]], test_data: List[Dict[str, Any]]):
    """纯向量检索评估"""
    skill_names = list(skills.keys())
    skill_texts = [f"{n} {skills[n]['description']} {skills[n]['body_text']}" for n in skill_names]
    skill_emb = model.encode(skill_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)

    queries = [item['query'] for item in test_data]
    q_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    sim = np.dot(q_emb, skill_emb.T)

    results = []
    for i, item in enumerate(test_data):
        top_idx = np.argsort(sim[i])[::-1][:5]
        top_names = [skill_names[idx] for idx in top_idx]
        top_scores = [float(sim[i][idx]) for idx in top_idx]

        results.append({
            "query": item['query'],
            "expected": item['positive'],
            "predicted": top_names[0],
            "top3": top_names[:3],
            "scores": top_scores[:3],
            "correct": item['positive'] == top_names[0],
            "in_top3": item['positive'] in top_names[:3],
        })

    return results, skill_names, skill_emb, sim


def evaluate_hybrid(model: SentenceTransformer, skills: Dict[str, Dict[str, str]], test_data: List[Dict[str, Any]]):
    """混合检索评估：同时评估纯向量、纯 BM25、混合检索三种模式

    返回三种模式各自的评估结果列表，以及向量检索的中间数据供后续分析使用。
    """
    skill_names = list(skills.keys())
    skill_texts = [f"{n} {skills[n]['description']} {skills[n]['body_text']}" for n in skill_names]
    skill_emb = model.encode(skill_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)

    queries = [item['query'] for item in test_data]
    q_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    sim = np.dot(q_emb, skill_emb.T)

    bm25 = BM25Searcher(skills)
    hybrid = HybridSearcher(config={
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "use_reranker": False,
        "hybrid_calibration": HYBRID_CALIBRATION,
    })

    vector_results = []
    bm25_results = []
    hybrid_results = []

    for i, item in enumerate(test_data):
        query = item['query']
        expected = item['positive']

        # ── 纯向量检索 ──
        vec_top_idx = np.argsort(sim[i])[::-1][:5]
        vec_top_names = [skill_names[idx] for idx in vec_top_idx]
        vec_top_scores = [float(sim[i][idx]) for idx in vec_top_idx]

        vector_results.append({
            "query": query,
            "expected": expected,
            "predicted": vec_top_names[0],
            "top3": vec_top_names[:3],
            "scores": vec_top_scores[:3],
            "correct": expected == vec_top_names[0],
            "in_top3": expected in vec_top_names[:3],
        })

        # ── 纯 BM25 检索 ──
        bm25_hits = bm25.search(query, top_k=5)
        bm25_top_names = [name for name, _ in bm25_hits[:5]]
        bm25_top_scores = [score for _, score in bm25_hits[:5]]

        bm25_results.append({
            "query": query,
            "expected": expected,
            "predicted": bm25_top_names[0] if bm25_top_names else "",
            "top3": bm25_top_names[:3],
            "scores": bm25_top_scores[:3],
            "correct": expected in bm25_top_names[:1],
            "in_top3": expected in bm25_top_names[:3],
        })

        # ── 混合检索：向量 + BM25 加权融合 ──
        vector_input = [{"name": name, "score": float(sim[i][skill_names.index(name)])} for name in vec_top_names]
        bm25_input = bm25_hits

        fused = hybrid.search(
            query=query,
            top_k=5,
            vector_results=vector_input,
            bm25_results=bm25_input,
            skill_names=skill_names,
        )

        hybrid_top_names = [r["name"] for r in fused]
        hybrid_top_scores = [r["score"] for r in fused]

        hybrid_results.append({
            "query": query,
            "expected": expected,
            "predicted": hybrid_top_names[0] if hybrid_top_names else "",
            "top3": hybrid_top_names[:3],
            "scores": hybrid_top_scores[:3],
            "correct": expected in hybrid_top_names[:1],
            "in_top3": expected in hybrid_top_names[:3],
        })

    return vector_results, bm25_results, hybrid_results


def print_confidence_distribution(results: List[Dict[str, Any]]) -> None:
    """输出置信度分布统计

    按置信度等级分组：high (>=0.4), medium (0.3-0.4), low (<0.3)
    """
    top1_scores = [r['scores'][0] for r in results if r['scores']]

    high = [s for s in top1_scores if s >= 0.4]
    medium = [s for s in top1_scores if 0.3 <= s < 0.4]
    low = [s for s in top1_scores if s < 0.3]
    total = len(top1_scores)

    print("\n置信度分布 (Top-1 分数):")
    print(f"  high   (>=0.4): {len(high):3d} 个 ({len(high)/total*100:5.1f}%)")
    print(f"  medium (0.3-0.4): {len(medium):3d} 个 ({len(medium)/total*100:5.1f}%)")
    print(f"  low    (<0.3):  {len(low):3d} 个 ({len(low)/total*100:5.1f}%)")

    if top1_scores:
        print(f"  平均分数: {np.mean(top1_scores):.3f}")
        print(f"  中位分数: {np.median(top1_scores):.3f}")


def print_accuracy(label: str, results: List[Dict[str, Any]]) -> None:
    """输出单模式的 Top-1 / Top-3 准确率"""
    correct = sum(1 for r in results if r['correct'])
    in_top3 = sum(1 for r in results if r['in_top3'])
    total = len(results)
    print(f"  {label}:")
    print(f"    Top-1: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"    Top-3: {in_top3}/{total} ({in_top3/total*100:.1f}%)")


def main() -> None:
    print("📊 技能路由器评估")
    print("=" * 60)

    model = load_model()
    skills = load_skills()

    with open(TRAINING_DATA, 'r') as f:
        test_data = json.load(f)

    # ── 混合检索评估 ──
    print(f"\n{'─' * 60}")
    print("🔍 混合检索模式对比")
    print(f"{'─' * 60}")

    vector_results, bm25_results, hybrid_results = evaluate_hybrid(model, skills, test_data)

    print_accuracy("纯向量检索", vector_results)
    print_accuracy("纯 BM25 检索", bm25_results)
    print_accuracy(f"混合检索 (向量 {VECTOR_WEIGHT} + BM25 {BM25_WEIGHT}, 校准={HYBRID_CALIBRATION})", hybrid_results)

    # ── 置信度分布统计（基于纯向量检索） ──
    print(f"\n{'─' * 60}")
    print("📈 置信度分布统计（纯向量检索）")
    print(f"{'─' * 60}")
    print_confidence_distribution(vector_results)

    # ── 混合检索置信度分布 ──
    print(f"\n📈 置信度分布统计（混合检索，校准={HYBRID_CALIBRATION}）")
    print_confidence_distribution(hybrid_results)

    # ── 基础准确率（纯向量） ──
    results = vector_results
    correct = sum(1 for r in results if r['correct'])
    in_top3 = sum(1 for r in results if r['in_top3'])
    total = len(results)

    print(f"\n{'=' * 60}")
    print("📋 纯向量检索详细分析")
    print("=" * 60)
    print("\n准确率:")
    print(f"  Top-1: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Top-3: {in_top3}/{total} ({in_top3/total*100:.1f}%)")

    # ── 错误分析 ──
    errors = [r for r in results if not r['correct']]
    print(f"\n错误案例 ({len(errors)} 个):")

    error_pairs: Dict[str, List[str]] = {}
    for e in errors:
        key = f"{e['expected']} → {e['predicted']}"
        if key not in error_pairs:
            error_pairs[key] = []
        error_pairs[key].append(e['query'])

    for pair, queries in sorted(error_pairs.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"\n  {pair} ({len(queries)} 次):")
        for q in queries[:3]:
            print(f"    - {q}")

    # ── 改进建议 ──
    print(f"\n{'=' * 60}")
    print("💡 改进建议")
    print("=" * 60)

    low_conf = [r for r in results if r['correct'] and r['scores'][0] < 0.5]
    if low_conf:
        print(f"\n1. 低置信度正确预测 ({len(low_conf)} 个):")
        print("   需要添加更多变体查询强化这些技能")
        for r in sorted(low_conf, key=lambda x: x['scores'][0])[:5]:
            print(f"   - '{r['query']}' → {r['expected']} ({r['scores'][0]:.3f})")

    if error_pairs:
        print("\n2. 高频错误对 (需添加困难负样本):")
        for pair, queries in sorted(error_pairs.items(), key=lambda x: -len(x[1]))[:5]:
            expected, predicted = pair.split(" → ")
            print(f"   - {pair}: {len(queries)} 次")
            print(f"     建议: 添加 {{'query': '...', 'positive': '{expected}', 'negatives': ['{predicted}']}}")

    # ── 混合检索改进分析 ──
    hybrid_improved = 0
    hybrid_degraded = 0
    for vr, hr in zip(vector_results, hybrid_results):
        if hr['correct'] and not vr['correct']:
            hybrid_improved += 1
        elif vr['correct'] and not hr['correct']:
            hybrid_degraded += 1

    if hybrid_improved > 0 or hybrid_degraded > 0:
        print("\n3. 混合检索对比纯向量:")
        print(f"   混合检索修复: {hybrid_improved} 个错误")
        print(f"   混合检索退化: {hybrid_degraded} 个正确")
        if hybrid_improved > hybrid_degraded:
            print("   ✅ 混合检索整体优于纯向量检索")
        else:
            print("   ⚠️ 混合检索整体不如纯向量检索，建议调整权重")


if __name__ == "__main__":
    main()
