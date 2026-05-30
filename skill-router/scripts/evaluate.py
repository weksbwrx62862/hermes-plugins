#!/usr/bin/env python3
"""
技能路由器评估脚本

功能：
1. 评估当前模型准确率（纯向量检索）
2. 混合检索评估：纯向量 / 纯 BM25 / 混合检索对比
3. 置信度分布统计
4. 错误案例分析与改进建议
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import logging
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.path.expanduser("~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v7")
DB_PATH = os.path.expanduser("~/.hermes/skill_index.db")
TRAINING_DATA = os.path.expanduser("~/.hermes/skills/devops/skill-router-scalable/training_data.json")

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


class BM25Evaluator:
    """BM25 评估器，与 __init__.py 中 BM25Searcher 算法一致

    使用 jieba 分词（不可用时降级为字符级分词），预计算文档分词和 IDF 值。
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self, skills: Dict[str, Dict]):
        self._use_jieba = False
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            self._use_jieba = True
        except ImportError:
            pass

        self._doc_tokens: Dict[str, List[str]] = {}
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._doc_len: Dict[str, int] = {}

        if not skills:
            return

        doc_freq: Dict[str, int] = {}
        total_dl = 0

        for name, info in skills.items():
            text = f"{name} {info.get('description', '')} {info.get('body', '')[:500]}"
            tokens = self._tokenize(text)
            self._doc_tokens[name] = tokens
            self._doc_len[name] = len(tokens)
            total_dl += len(tokens)

            seen = set()
            for t in tokens:
                if t not in seen:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                    seen.add(t)

        n_docs = len(skills)
        self._avg_dl = total_dl / n_docs if n_docs > 0 else 1.0

        for term, df in doc_freq.items():
            self._idf[term] = max(0.0, ((n_docs - df + 0.5) / (df + 0.5) + 1.0))

    def _tokenize(self, text: str) -> List[str]:
        """分词：优先 jieba，降级为字符级"""
        if self._use_jieba:
            import jieba
            return [w for w in jieba.cut(text) if w.strip()]
        return [c for c in text if c.strip()]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """检索与查询最相关的技能，返回 (skill_name, score) 列表"""
        if not self._doc_tokens:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for name, doc_tokens in self._doc_tokens.items():
            tf_map: Dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            dl = self._doc_len[name]
            score = 0.0
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = self._idf.get(qt, 0.0)
                numerator = tf * (self._K1 + 1)
                denominator = tf + self._K1 * (1 - self._B + self._B * dl / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores[name] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def load_model():
    """加载嵌入模型"""
    return SentenceTransformer(MODEL_PATH, device="cpu")


def load_skills():
    """从数据库加载技能索引"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, description, body FROM skills")
    skills = {}
    for name, desc, body in cursor.fetchall():
        skills[name] = {"description": (desc or "")[:500], "body": (body or "")[:500]}
    conn.close()
    return skills


def evaluate(model, skills, test_data):
    """纯向量检索评估"""
    skill_names = list(skills.keys())
    skill_texts = [f"{n} {skills[n]['description']} {skills[n]['body'][:200]}" for n in skill_names]
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


def evaluate_hybrid(model, skills, test_data):
    """混合检索评估：同时评估纯向量、纯 BM25、混合检索三种模式

    返回三种模式各自的评估结果列表，以及向量检索的中间数据供后续分析使用。
    """
    skill_names = list(skills.keys())
    skill_texts = [f"{n} {skills[n]['description']} {skills[n]['body'][:200]}" for n in skill_names]
    skill_emb = model.encode(skill_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)

    queries = [item['query'] for item in test_data]
    q_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    sim = np.dot(q_emb, skill_emb.T)

    bm25 = BM25Evaluator(skills)

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

        # ── 混合检索：向量 0.7 + BM25 0.3 加权融合 ──
        vector_score_map = {name: float(sim[i][skill_names.index(name)]) for name in vec_top_names}
        bm25_score_map = dict(bm25_hits)

        norm_vector = _normalize_scores(vector_score_map)
        norm_bm25 = _normalize_scores(bm25_score_map)

        all_candidates = set(norm_vector.keys()) | set(norm_bm25.keys())
        fused = []
        for name in all_candidates:
            v_score = norm_vector.get(name, 0.0)
            b_score = norm_bm25.get(name, 0.0)
            weighted = VECTOR_WEIGHT * v_score + BM25_WEIGHT * b_score
            fused.append((name, weighted))

        fused.sort(key=lambda x: x[1], reverse=True)
        hybrid_top = fused[:5]
        hybrid_top_names = [name for name, _ in hybrid_top]
        hybrid_top_scores = [score for _, score in hybrid_top]

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


def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    """将分数归一化到 [0, 1] 范围（min-max 归一化）"""
    if not score_map:
        return {}
    values = list(score_map.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return {k: 1.0 for k in score_map}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}


def print_confidence_distribution(results):
    """输出置信度分布统计

    按置信度等级分组：high (>=0.4), medium (0.3-0.4), low (<0.3)
    """
    top1_scores = [r['scores'][0] for r in results if r['scores']]

    high = [s for s in top1_scores if s >= 0.4]
    medium = [s for s in top1_scores if 0.3 <= s < 0.4]
    low = [s for s in top1_scores if s < 0.3]
    total = len(top1_scores)

    print(f"\n置信度分布 (Top-1 分数):")
    print(f"  high   (>=0.4): {len(high):3d} 个 ({len(high)/total*100:5.1f}%)")
    print(f"  medium (0.3-0.4): {len(medium):3d} 个 ({len(medium)/total*100:5.1f}%)")
    print(f"  low    (<0.3):  {len(low):3d} 个 ({len(low)/total*100:5.1f}%)")

    if top1_scores:
        print(f"  平均分数: {np.mean(top1_scores):.3f}")
        print(f"  中位分数: {np.median(top1_scores):.3f}")


def print_accuracy(label, results):
    """输出单模式的 Top-1 / Top-3 准确率"""
    correct = sum(1 for r in results if r['correct'])
    in_top3 = sum(1 for r in results if r['in_top3'])
    total = len(results)
    print(f"  {label}:")
    print(f"    Top-1: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"    Top-3: {in_top3}/{total} ({in_top3/total*100:.1f}%)")


def main():
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
    print_accuracy(f"混合检索 (向量 {VECTOR_WEIGHT} + BM25 {BM25_WEIGHT})", hybrid_results)

    # ── 置信度分布统计（基于纯向量检索） ──
    print(f"\n{'─' * 60}")
    print("📈 置信度分布统计（纯向量检索）")
    print(f"{'─' * 60}")
    print_confidence_distribution(vector_results)

    # ── 混合检索置信度分布 ──
    print(f"\n📈 置信度分布统计（混合检索）")
    print_confidence_distribution(hybrid_results)

    # ── 基础准确率（纯向量） ──
    results = vector_results
    correct = sum(1 for r in results if r['correct'])
    in_top3 = sum(1 for r in results if r['in_top3'])
    total = len(results)

    print(f"\n{'=' * 60}")
    print("📋 纯向量检索详细分析")
    print("=" * 60)
    print(f"\n准确率:")
    print(f"  Top-1: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Top-3: {in_top3}/{total} ({in_top3/total*100:.1f}%)")

    # ── 错误分析 ──
    errors = [r for r in results if not r['correct']]
    print(f"\n错误案例 ({len(errors)} 个):")

    error_pairs = {}
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
        print(f"\n2. 高频错误对 (需添加困难负样本):")
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
        print(f"\n3. 混合检索对比纯向量:")
        print(f"   混合检索修复: {hybrid_improved} 个错误")
        print(f"   混合检索退化: {hybrid_degraded} 个正确")
        if hybrid_improved > hybrid_degraded:
            print(f"   ✅ 混合检索整体优于纯向量检索")
        else:
            print(f"   ⚠️ 混合检索整体不如纯向量检索，建议调整权重")


if __name__ == "__main__":
    main()
