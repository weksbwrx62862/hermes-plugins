import time
import json
import statistics
from pathlib import Path
from typing import Any

from omnimem.deep.consolidation import ConsolidationEngine
from omnimem.deep.reflect import ReflectEngine
from omnimem.internalize.kv_cache import KVCacheManager
from omnimem.retrieval.engine import HybridRetriever


_TEST_FACTS = [
    "用户喜欢使用Python进行数据分析",
    "用户每天早上喝咖啡",
    "用户养了一只橘猫叫小橘",
    "用户偏好深色主题的编辑器",
    "用户习惯使用Git进行版本控制",
    "用户正在学习深度学习技术",
    "用户喜欢在周末阅读技术书籍",
    "用户使用VS Code作为主要IDE",
    "用户对函数式编程感兴趣",
    "用户每天工作8小时",
]

_REFLECT_QUERIES = [
    "用户的技术偏好",
    "用户的生活习惯",
    "用户的学习方向",
]

_RETRIEVAL_QUERIES = [
    "用户喜欢什么编程语言",
    "用户的宠物",
    "用户的工作习惯",
    "用户的编辑器偏好",
    "用户的学习兴趣",
    "用户每天喝什么",
    "用户的版本控制工具",
    "用户的编程风格",
    "用户的周末活动",
    "用户的IDE选择",
]


class L3L4Benchmark:
    def __init__(self, storage_dir: str | Path):
        self._storage_dir = Path(storage_dir)
        self._results: dict[str, Any] = {}

    def run_all(self) -> dict:
        self._results = {
            "timestamp": time.time(),
            "consolidation": self._benchmark_consolidation(),
            "reflection": self._benchmark_reflection(),
            "kv_cache": self._benchmark_kv_cache(),
            "retrieval": self._benchmark_retrieval(),
        }
        return self._results

    def _benchmark_consolidation(self) -> dict:
        data_dir = self._storage_dir / "bench_consolidation"
        engine = ConsolidationEngine(data_dir=data_dir, fact_threshold=5)

        for i, fact in enumerate(_TEST_FACTS):
            engine.submit(memory_id=f"bench-fact-{i:03d}", content=fact)

        start = time.perf_counter()
        processed = engine.process_pending()
        latency_ms = (time.perf_counter() - start) * 1000

        stats = engine.get_stats()
        observations = engine.get_observations(limit=50)
        models = engine.get_mental_models(limit=50)

        engine.close()

        return {
            "latency_ms": round(latency_ms, 3),
            "facts_processed": processed,
            "observations_generated": len(observations),
            "models_generated": len(models),
            "stats": stats,
        }

    def _benchmark_reflection(self) -> dict:
        data_dir = self._storage_dir / "bench_reflect"
        consolidation = ConsolidationEngine(
            data_dir=data_dir / "consolidation", fact_threshold=3
        )

        for i, fact in enumerate(_TEST_FACTS):
            consolidation.submit(memory_id=f"bench-ref-{i:03d}", content=fact)
        consolidation.process_pending()

        engine = ReflectEngine(
            data_dir=data_dir / "reflect",
            consolidation_engine=consolidation,
        )

        total_latency = 0.0
        steps_completed = 0
        all_keywords: list[str] = []

        for query in _REFLECT_QUERIES:
            start = time.perf_counter()
            result = engine.reflect(query=query)
            latency = (time.perf_counter() - start) * 1000
            total_latency += latency
            steps_completed += 1

            if result.observation:
                from omnimem.deep.consolidation import _extract_keywords

                kws = _extract_keywords([result.observation], top_k=5)
                all_keywords.extend(kws)

        engine.close()
        consolidation.close()

        unique_keywords = list(dict.fromkeys(all_keywords))

        return {
            "latency_ms": round(total_latency, 3),
            "steps_completed": steps_completed,
            "keywords_extracted": len(unique_keywords),
            "keywords_sample": unique_keywords[:10],
        }

    def _benchmark_kv_cache(self) -> dict:
        data_dir = self._storage_dir / "bench_kv_cache"
        cache = KVCacheManager(data_dir=data_dir, auto_preload_threshold=5, max_cache_size=50)

        n_entries = 30
        n_cached = 15
        n_lookups = 100

        for i in range(n_entries):
            key = f"pattern-{i:03d}"
            cache.check_and_auto_preload(
                key=key,
                content=f"缓存测试内容 #{i}: 关于主题{i % 5}的频繁访问记忆",
                metadata={"index": i},
                source_memory_ids=[f"mem-{i:03d}"],
            )

        cache.preload([
            {
                "key": f"pattern-{i:03d}",
                "content": f"缓存测试内容 #{i}: 关于主题{i % 5}的频繁访问记忆",
                "metadata": {"index": i},
                "source_memory_ids": [f"mem-{i:03d}"],
            }
            for i in range(n_cached)
        ])

        hits = 0
        latencies: list[float] = []

        for lookup_idx in range(n_lookups):
            if lookup_idx % 2 == 0:
                idx = lookup_idx % n_cached
            else:
                idx = n_cached + (lookup_idx % (n_entries - n_cached))
            key = f"pattern-{idx:03d}"

            start = time.perf_counter()
            result = cache.get(key)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            if result is not None:
                hits += 1

        cache.close()

        hit_rate = hits / n_lookups if n_lookups > 0 else 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0

        return {
            "entries": n_entries,
            "lookups": n_lookups,
            "hits": hits,
            "hit_rate": round(hit_rate, 4),
            "avg_latency_ms": round(avg_latency, 3),
        }

    def _benchmark_retrieval(self) -> dict:
        data_dir = self._storage_dir / "bench_retrieval"
        retriever = HybridRetriever(vector_backend="chromadb", data_dir=data_dir)

        n_entries = 100
        for i in range(n_entries):
            topic = i % 10
            content = (
                f"记忆条目 #{i}: 这是关于主题{topic}的测试内容。"
                f"用户对主题{topic}有持续的兴趣和关注。"
            )
            retriever.add(
                content=content,
                memory_id=f"bench-mem-{i:03d}",
                metadata={"topic": str(topic), "index": i},
            )

        latencies: list[float] = []
        result_counts: list[int] = []

        for query in _RETRIEVAL_QUERIES:
            start = time.perf_counter()
            results = retriever.search(query=query, top_k=10, max_tokens=1500)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            result_counts.append(len(results))

        retriever.flush()

        avg_latency = statistics.mean(latencies) if latencies else 0.0
        avg_results = statistics.mean(result_counts) if result_counts else 0.0
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p50_ms = sorted_latencies[p50_idx] if sorted_latencies else 0.0
        p95_ms = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0

        return {
            "entries": n_entries,
            "queries": len(_RETRIEVAL_QUERIES),
            "avg_latency_ms": round(avg_latency, 3),
            "avg_results": round(avg_results, 2),
            "p50_ms": round(p50_ms, 3),
            "p95_ms": round(p95_ms, 3),
        }

    def save_results(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(self._results, f, ensure_ascii=False, indent=2)
