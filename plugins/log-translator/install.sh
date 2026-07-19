#!/bin/bash
# Hermes 日志翻译器插件集成脚本

set -e

PLUGIN_DIR="$HOME/.hermes/plugins/log-translator"
CONFIG_FILE="$HOME/.hermes/config.yaml"

echo "=== Hermes 日志翻译器插件集成 ==="
echo ""

# 检查插件目录是否存在
if [ ! -d "$PLUGIN_DIR" ]; then
    echo "错误: 插件目录不存在: $PLUGIN_DIR"
    exit 1
fi

# 检查插件文件是否存在
if [ ! -f "$PLUGIN_DIR/__init__.py" ]; then
    echo "错误: 插件文件不存在: $PLUGIN_DIR/__init__.py"
    exit 1
fi

echo "插件目录: $PLUGIN_DIR"

# 测试插件
echo ""
echo "=== 测试插件 ==="
cd "$PLUGIN_DIR"
python3 __init__.py --test

echo ""
echo "=== 集成说明 ==="
echo ""
echo "插件已准备就绪！请按照以下步骤手动集成："
echo ""
echo "1. 编辑配置文件:"
echo "   $CONFIG_FILE"
echo ""
echo "2. 在 plugins.enabled 列表中添加 'log-translator':"
echo ""
echo "   plugins:"
echo "     disabled: []"
echo "     enabled:"
echo "     - log-translator          # <-- 添加这一行"
echo "     - repo-chinese-names"
echo "     - adaptive_multi_agent"
echo "     - deepseek-cache-optimizer"
echo "     - dev-lifecycle"
echo "     - disk-cleanup"
echo "     - model-router"
echo "     - self-evolution"
echo "     - skill_pool"
echo ""
echo "3. 重启 Hermes:"
echo "   hermes restart"
echo ""
echo "4. 验证插件是否加载:"
echo "   hermes plugins list"
echo ""
echo "=== 快速命令 ==="
echo ""
echo "测试插件:"
echo "  python3 $PLUGIN_DIR/__init__.py --test"
echo ""
echo "列出所有翻译规则:"
echo "  python3 $PLUGIN_DIR/__init__.py --list"
echo ""
echo "安装日志翻译器 (手动):"
echo "  python3 $PLUGIN_DIR/__init__.py --install"
echo ""
echo "=== 完成 ==="
