# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

如果你发现了安全漏洞，请通过以下方式报告：

1. **不要** 在公开的 Issue 中披露漏洞
2. 发送邮件至 [security@example.com]（替换为实际邮箱）
3. 邮件主题：`[SECURITY] OmniMem Vulnerability Report`
4. 包含以下信息：
   - 漏洞描述
   - 复现步骤
   - 影响范围
   - 建议修复方案（如有）

我们会在 48 小时内确认收到报告，并在 7 天内提供初步评估。

## 安全最佳实践

### 使用 OmniMem 时

- 不要在代码中硬编码 API 密钥或密码
- 定期更新依赖包以获取安全补丁
- 对于 `secret` 级别的隐私数据，建议安装 `cryptography` 依赖启用加密
- 配置文件（`config.yaml`）已包含在 `.gitignore` 中，避免意外提交敏感信息

### 开发时

- 运行 `pre-commit install` 确保代码提交前通过安全检查
- 定期运行 `pip-audit` 检查依赖漏洞

## 已知安全问题

目前无已知安全问题。历史安全问题将记录在此处。
