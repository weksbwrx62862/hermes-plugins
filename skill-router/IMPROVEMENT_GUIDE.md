# 技能路由器持续改进指南

## 快速改进流程

### 1. 评估当前状态
```bash
cd ~/.hermes/plugins/skill-router
python3 scripts/evaluate.py
```

### 2. 添加训练数据

编辑 `~/.hermes/skills/devops/skill-router-scalable/training_data.json`：

```json
{
  "query": "用户实际查询",
  "positive": "正确技能名",
  "negatives": ["容易混淆的技能1", "容易混淆的技能2"]
}
```

**技巧**：
- 添加**口语化**表达（"帮我看看行情" 而非 "查询股票价格"）
- 添加**困难负样本**（容易混淆的技能）
- 添加**变体查询**（同义词、不同表述）

### 3. 重新微调
```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=""
cd ~/.hermes/skills/devops/skill-router-scalable

# 快速训练（1 epoch，~2分钟）
python3 -c "
import os, json, sqlite3
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

with open('training_data.json') as f:
    data = json.load(f)
conn = sqlite3.connect(os.path.expanduser('~/.hermes/skill_index.db'))
descs = {n: (d or '')[:500] for n, d in conn.execute('SELECT name, description FROM skills')}
conn.close()

examples = [InputExample(texts=[d['query'], descs.get(d['positive'], d['positive'])]) for d in data]
model = SentenceTransformer('shibing624/text2vec-base-chinese', device='cpu')
loader = DataLoader(examples, shuffle=True, batch_size=64)
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=int(len(loader)*0.1), 
          optimizer_params={'lr': 2e-5}, show_progress_bar=True,
          output_path='fine-tuned-model-v5')
print('✅ 完成')
"
```

### 4. 更新插件配置

编辑 `~/.hermes/plugins/skill-router/__init__.py`，更新 `model_path`：
```python
"model_path": "~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v5",
```

### 5. 清除缓存重启

```bash
# 清除插件缓存
rm -f ~/.hermes/skill_index.db-journal

# 重启 Hermes（如果在运行）
hermes gateway restart  # 或重新启动 CLI
```

## 常见改进场景

### 场景1: 口语化查询未命中
**问题**: "帮我看看行情" 未命中 finance
**解决**: 添加口语化变体
```json
{"query": "帮我看看行情", "positive": "finance", "negatives": ["stock-analyzer", "market-data"]}
```

### 场景2: 相似技能混淆
**问题**: python-debugpy 和 jupyter-live-kernel 混淆
**解决**: 添加困难负样本
```json
{"query": "设置断点调试", "positive": "python-debugpy", "negatives": ["jupyter-live-kernel"]}
```

### 场景3: 新技能需要训练
**问题**: 新添加的技能未被识别
**解决**: 
1. 在 `training_data.json` 添加 3-5 个查询
2. 重新微调模型

## 评估指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Top-1 准确率 | >65% | 首选技能正确 |
| Top-3 准确率 | >85% | 前三包含正确技能 |
| 低置信度比例 | <30% | score < 0.5 的比例 |

## 文件位置

- 训练数据: `~/.hermes/skills/devops/skill-router-scalable/training_data.json`
- 模型目录: `~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v*`
- 评估脚本: `~/.hermes/plugins/skill-router/scripts/evaluate.py`
- 插件配置: `~/.hermes/plugins/skill-router/__init__.py`
