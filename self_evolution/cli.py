#!/usr/bin/env python3
"""
自进化 CLI 入口

用法:
  # 进化一个 skill
  python -m plugins.self_evolution.cli --evolve github-actions-python-ci

  # 进化并自动部署
  python -m plugins.self_evolution.cli --evolve my-skill --deploy

  # 指定迭代次数和模型
  python -m plugins.self_evolution.cli --evolve my-skill --iterations 10 --model gpt-4o

  # 快速模式（启发式评分，省 API 费用）
  python -m plugins.self_evolution.cli --evolve my-skill --fast
"""

import argparse
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="自进化 - 让 Skill 自己训练自己",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --evolve github-actions-python-ci
  %(prog)s --evolve my-skill --deploy --iterations 10
  %(prog)s --list-providers
  %(prog)s --rollback my-skill
        """,
    )

    # 主要命令
    parser.add_argument(
        "--evolve", metavar="SKILL_NAME",
        help="对指定 skill 执行自进化管线",
    )
    parser.add_argument(
        "--auto-scan", action="store_true",
        help="扫描所有 skill，对达到阈值的生成审批请求",
    )
    parser.add_argument(
        "--approve", metavar="SKILL_NAME",
        help="批准指定 skill 的进化请求",
    )
    parser.add_argument(
        "--reject", metavar="SKILL_NAME",
        help="拒绝指定 skill 的进化请求",
    )
    parser.add_argument(
        "--execute-approved", action="store_true",
        help="执行所有已批准的进化任务",
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="全联动：执行已批准进化 → 重建索引 → 调优常驻 → 发起下次扫描",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="查看所有 skill 的进化状态",
    )
    parser.add_argument(
        "--rollback", metavar="SKILL_NAME",
        help="回滚指定 skill 到最近备份",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="列出可用的进化提供者",
    )

    # 配置选项
    parser.add_argument(
        "--deploy", action="store_true",
        help="自动部署通过验证的进化版本",
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="优化迭代次数 (默认: 5)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="用于评估和变异的模型 (默认: gpt-4o-mini)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="快速模式：用启发式评分替代 LLM-as-judge，加快速度减少 API 费用",
    )
    parser.add_argument(
        "--optimizer", default="diversify",
        choices=["gepa", "diversify"],
        help="优化策略选择 (默认: diversify)",
    )
    parser.add_argument(
        "--target-dir", default=None,
        help="Hermes Agent 仓库根目录 (默认: ~/.hermes/)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI 兼容 API key (默认从 OPENAI_API_KEY 环境变量读取)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI 兼容 API base URL",
    )

    args = parser.parse_args()

    # 导入核心模块
    from self_evolution import create_manager
    from self_evolution.default_provider import DefaultEvolutionProvider
    from self_evolution.core.evolution_provider import EvolutionPhase

    # 创建 manager
    manager = create_manager(optimizer_name=args.optimizer)

    if args.list_providers:
        print("Available evolution providers:")
        manager.add_provider(DefaultEvolutionProvider())  # force add
        for p in manager.list_providers():
            print(f"  - {p}")
        return 0

    # 处理 status
    if args.status:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        states = aem.list_all()
        if not states:
            print("📊 暂无进化记录")
            return 0
        print(f"{'Skill':<30} {'状态':<12} {'使用次数':<10} {'进化次数':<8} {'上次分数':<8}")
        print("-" * 75)
        for s in states:
            icon = {"monitor": "👀", "pending": "⏳", "approved": "✅",
                    "rejected": "❌", "executing": "🔄", "done": "🎉"}.get(s.status, "?")
            print(f"{s.skill_name:<30} {icon} {s.status:<10} {s.usage_count:<10} "
                  f"{s.evolution_count:<8} {s.evolved_score:<8.3f}")
        return 0

    # 处理 auto-scan
    if args.auto_scan:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        candidates = aem.scan_candidates()

        if not candidates:
            print("📊 扫描完成：没有需要进化的 skill")
            return 0

        print(f"📊 扫描完成：发现 {len(candidates)} 个候选 skill\n")
        for c in candidates:
            print(aem.format_approval_request(c))
            print()
        print("使用 --approve <name> 或 --reject <name> 处理")
        return 0

    # 处理 approve
    if args.approve:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        if aem.approve(args.approve):
            print(f"✅ 已批准 {args.approve} 的进化请求")
            print("使用 --execute-approved 执行进化")
        else:
            print(f"❌ {args.approve} 不在待审批状态")
        return 0

    # 处理 reject
    if args.reject:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        if aem.reject(args.reject):
            print(f"❌ 已拒绝 {args.reject}，进入冷却期")
        else:
            print(f"❌ {args.reject} 不在待审批状态")
        return 0

    # 处理 execute-approved
    if args.execute_approved:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        approved = aem.get_approved()

        if not approved:
            print("没有已批准的进化任务")
            return 0

        print(f"🚀 执行 {len(approved)} 个已批准的进化任务\n")

        # 初始化 API
        target_dir = args.target_dir or str(Path.home() / ".hermes")
        manager.initialize_all(
            target_dir,
            model=args.model,
            iterations=args.iterations,
            use_llm_eval=not args.fast,
            api_key=args.api_key,
            base_url=args.base_url,
        )

        for state in approved:
            print(f"▶ 正在进化: {state.skill_name}")
            aem.mark_executing(state.skill_name)

            result = manager.evolve_skill(
                state.skill_name,
                phase=EvolutionPhase.SKILL,
                iterations=args.iterations,
                auto_deploy=args.deploy,
            )

            if result.error:
                print(f"  ✗ 失败: {result.error}\n")
                # 回退到 pending 以便重试
                aem._states[state.skill_name].status = "pending"
            else:
                improvement_str = f"{result.improvement:+.3f}" if result.improvement != 0 else "无变化"
                print(f"  ✓ 完成: {result.baseline_score:.3f} → {result.evolved_score:.3f} ({improvement_str})")
                print(f"    约束: {'通过' if result.constraint_passed else '失败'}")
                print(f"    部署: {'已部署' if result.deployed else '未部署'}\n")
                aem.mark_done(state.skill_name, result.evolved_score)

        aem._save()
        print("✅ 全部完成")
        return 0

    # 处理 rollback
    if args.rollback:
        provider = manager.get_provider(EvolutionPhase.SKILL)
        if not provider:
            print("ERROR: no SKILL provider available")
            return 1

        target_dir = args.target_dir or str(Path.home() / ".hermes")
        provider.initialize(target_dir)
        skill_path = Path(target_dir) / "skills" / args.rollback / "SKILL.md"

        if provider.handle_rollback(str(skill_path)):
            print(f"✓ rolled back {args.rollback}")
            return 0
        else:
            print(f"✗ rollback failed for {args.rollback}")
            return 1

    # ── 全联动 ──────────────────────────────────────────────
    if args.sync:
        from self_evolution.core.orchestrator import EvolutionOrchestrator

        target_dir = args.target_dir or str(Path.home() / ".hermes")
        orch = EvolutionOrchestrator(
            target_dir=target_dir,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )

        print("🔄 全联动同步中...\n")
        report = orch.sync_all(fast=args.fast, iterations=args.iterations)
        print(orch.format_report(report))
        return 0

    # 处理 evolve
    if args.evolve:
        # 确定 target_dir
        target_dir = args.target_dir or str(Path.home() / ".hermes")
        target = Path(target_dir)
        if not target.exists():
            print(f"ERROR: target directory not found: {target_dir}")
            print("Use --target-dir to specify your hermes-agent root")
            return 1

        # 初始化
        print("=" * 60)
        print(f" 自进化引擎 v1.0")
        print(f" 8阶段 Loop | 3层评测 | 5维 AND 门控")
        print("=" * 60)
        print()

        manager.initialize_all(
            str(target),
            model=args.model,
            iterations=args.iterations,
            use_llm_eval=not args.fast,
            api_key=args.api_key,
            base_url=args.base_url,
        )

        if args.fast:
            print("⚡ 快速模式 (启发式评分)")
        print(f"🤖 评估模型: {args.model}")
        print(f"🔄 迭代次数: {args.iterations}")
        print(f"🧬 优化策略: {args.optimizer}")
        print(f"🎯 目标 skill: {args.evolve}")
        print()

        # 执行进化
        result = manager.evolve_skill(
            args.evolve,
            phase=EvolutionPhase.SKILL,
            iterations=args.iterations,
            auto_deploy=args.deploy,
        )

        # 输出结果
        print()
        print("─" * 60)
        print(" 进化结果")
        print("─" * 60)

        if result.error:
            print(f"✗ 失败: {result.error}")
            return 1

        print(f"  Skill:        {result.target_name}")
        print(f"  Phase:        {result.phase.name}")
        print(f"  Baseline:     {result.baseline_score:.3f}")
        print(f"  Evolved:      {result.evolved_score:.3f}")
        print(f"  Improvement:  {result.improvement:+.3f}")
        print(f"  Iterations:   {result.iterations_used}")
        print(f"  Holdout:      {result.holdout_score:.3f}")

        # 门控结果
        print(f"\n  5维 AND 门控:")
        for dim, passed in result.constraint_details.checks.items():
            icon = "✓" if passed else "✗"
            print(f"    {icon} {dim}")

        # 部署状态
        if result.deployed:
            print(f"\n  ✓ 已自动部署到 ~/.hermes/skills/{result.target_name}/")
        elif result.improvement > 0 and not args.deploy:
            print(f"\n  ⚠ 有改进但未部署 (使用 --deploy 自动部署)")
        elif result.improvement <= 0:
            print(f"\n  - 无改进，不部署")

        print()
        return 0

    # 无命令
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
