from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from omnimem.utils.logging import sanitize_for_log
from omnimem.utils.metrics import (
    dec_active_connections,
    get_metrics_collector,
    inc_active_connections,
    record_memorize_duration,
    record_recall_duration,
    record_reflect_duration,
)

logger = logging.getLogger(__name__)

# 请求体大小限制（10 MB）
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024


class AuthMiddleware:
    """基于 Bearer Token 的 API Key 认证中间件。

    默认启用：未配置 api_key 时仍要求认证（由 run_api 生成默认密钥）。
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def validate(self, request_headers: dict[str, str]) -> bool | tuple[int, dict]:
        """验证请求头。

        当 api_key 为空字符串时关闭认证，直接返回 True（仅用于测试显式关闭）。
        当请求头 Authorization 为 Bearer <api_key> 时返回 True。
        其它情况返回 (401, {"error": "Unauthorized"})。
        """
        if self._api_key == "":
            return True
        auth_header = request_headers.get("Authorization", "")
        if auth_header == f"Bearer {self._api_key}":
            return True
        return (401, {"error": "Unauthorized"})


class AdminAuthMiddleware:
    """敏感操作（导出/导入）的额外权限令牌校验中间件。"""

    def __init__(self, admin_token: str = ""):
        self._admin_token = admin_token

    def validate(self, request_headers: dict[str, str]) -> bool | tuple[int, dict]:
        """验证管理令牌。

        当 admin_token 为空字符串时跳过检查。
        当请求头 X-Admin-Token 匹配时返回 True。
        其它情况返回 (403, {"error": "Forbidden"})。
        """
        if self._admin_token == "":
            return True
        admin_header = request_headers.get("X-Admin-Token", "")
        if admin_header == self._admin_token:
            return True
        return (403, {"error": "Forbidden"})


class RateLimiter:
    """基于内存字典的滑动窗口速率限制器，按 (client_ip, path) 计数。"""

    def __init__(self, limit_per_minute: int = 60):
        self._limit = limit_per_minute
        self._window_seconds = 60
        self._lock = threading.Lock()
        # 记录每个 (client_ip, path) 的请求时间戳列表
        self._records: dict[tuple[str, str], list[float]] = {}

    def is_allowed(self, client_ip: str, path: str) -> bool:
        """判断请求是否未超过速率限制。"""
        key = (client_ip, path)
        now = time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            timestamps = self._records.get(key, [])
            # 仅保留滑动窗口内的记录
            timestamps = [ts for ts in timestamps if ts > cutoff]
            if len(timestamps) >= self._limit:
                self._records[key] = timestamps
                return False
            timestamps.append(now)
            self._records[key] = timestamps
            return True


class OmniMemAPIHandler(BaseHTTPRequestHandler):
    # 使用 HTTP/1.1 以支持更稳定的连接管理（如大请求体拒绝后的优雅关闭）
    protocol_version = "HTTP/1.1"

    _sdk = None
    _auth_middleware: AuthMiddleware | None = None
    _admin_auth_middleware: AdminAuthMiddleware | None = None
    _rate_limiter: RateLimiter | None = None
    _cors_allowed_origins: list[str] = []

    def _get_client_ip(self) -> str:
        """获取客户端 IP 地址。"""
        return self.client_address[0]

    def _drain_request_body(self, content_length: int) -> None:
        """读取并丢弃请求体，避免客户端因连接提前关闭而收到 Broken Pipe。

        为防御超大恶意请求，最多排空 2 倍大小限制。
        """
        drain_limit = min(content_length, _MAX_REQUEST_BODY_BYTES * 2)
        remaining = drain_limit
        chunk_size = 64 * 1024
        while remaining > 0:
            chunk = self.rfile.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _get_origin(self) -> str | None:
        """获取请求 Origin 头。"""
        origin = self.headers.get("Origin")
        return origin.strip() if origin else None

    def _add_cors_headers(self) -> None:
        """根据 cors_allowed_origins 配置添加 CORS 响应头。"""
        if not self._cors_allowed_origins:
            return
        origin = self._get_origin()
        if origin and origin in self._cors_allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)

    def _check_security(self, path: str) -> bool:
        """执行认证与速率限制检查，通过返回 True，否则已发送错误响应。"""
        auth = self._auth_middleware.validate(dict(self.headers))
        if auth is not True:
            code, body = auth
            self._send_json(code, body)
            return False

        if not self._rate_limiter.is_allowed(self._get_client_ip(), path):
            self._send_json(429, {"error": "Rate limit exceeded"})
            return False

        return True

    def _check_admin(self) -> bool:
        """校验敏感操作的管理令牌，通过返回 True，否则已发送错误响应。"""
        admin = self._admin_auth_middleware.validate(dict(self.headers))
        if admin is not True:
            code, body = admin
            self._send_json(code, body)
            return False
        return True

    def _audit_log(self, operation: str, details: dict | None = None) -> None:
        """记录敏感操作审计日志，并对详情进行脱敏。"""
        safe_details = sanitize_for_log(json.dumps(details, ensure_ascii=False, default=str)) if details else None
        logger.info("AUDIT operation=%s details=%s", operation, safe_details or "")
        sdk = getattr(self, "_sdk", None)
        if sdk is None:
            return
        governance = getattr(sdk, "_governance", None)
        audit_logger = getattr(governance, "audit_logger", None) if governance else None
        if audit_logger is not None:
            try:
                audit_logger.log(
                    operation,
                    details=details,
                    result="success",
                    instance_id=getattr(sdk, "_instance_id", None),
                )
            except Exception as e:
                logger.warning("审计日志写入失败: %s", e)

    def _send_json(self, code: int, data: Any):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self._add_cors_headers()
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if not self._check_security(path):
            return

        # 请求体大小限制
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > _MAX_REQUEST_BODY_BYTES:
            # 先排空请求体，避免客户端在发送过程中收到 Broken Pipe
            self._drain_request_body(content_length)
            self.close_connection = True
            self._send_json(413, {"error": "请求体超过 10MB 限制"})
            return

        # JSON 请求体解析
        try:
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        # 敏感操作需要额外管理令牌
        if path in ("/api/export", "/api/import") and not self._check_admin():
            return

        routes = {
            "/api/memorize": self._handle_memorize,
            "/api/recall": self._handle_recall,
            "/api/reflect": self._handle_reflect,
            "/api/govern": self._handle_govern,
            "/api/compact": self._handle_compact,
            "/api/detail": self._handle_detail,
            "/api/export": self._handle_export,
            "/api/import": self._handle_import,
            "/api/health": self._handle_health,
        }

        handler = routes.get(path)
        if handler:
            try:
                result = handler(body)
                self._send_json(200, result)
            except Exception as e:
                logger.exception("REST API 处理失败: path=%s", path)
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": f"Not found: {path}"})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/metrics":
            # Prometheus 指标导出端点
            self._send_metrics()
            return

        if not self._check_security(path):
            return

        if path == "/api/health":
            self._send_json(200, self._sdk.health_check() if self._sdk else {"status": "not initialized"})
        elif path == "/api/tools":
            self._send_json(200, {"tools": ["memorize", "recall", "reflect", "govern", "compact", "detail", "export", "import", "health"]})
        else:
            self._send_json(404, {"error": f"Not found: {path}"})

    def _handle_memorize(self, body):
        start = time.time()
        try:
            result = self._sdk.memorize(**body)
            return result
        finally:
            record_memorize_duration(time.time() - start)

    def _handle_recall(self, body):
        start = time.time()
        try:
            result = self._sdk.recall(**body)
            return result
        finally:
            record_recall_duration(time.time() - start)

    def _handle_reflect(self, body):
        start = time.time()
        try:
            result = self._sdk.reflect(**body)
            return result
        finally:
            record_reflect_duration(time.time() - start)

    def _handle_govern(self, body):
        return self._sdk.govern(**body)

    def _handle_compact(self, body):
        return self._sdk.compact(**body)

    def _handle_detail(self, body):
        action = body.get("action", "list")
        if action == "list":
            return self._sdk.detail_list(**body)
        elif action == "events":
            return self._sdk.detail_events(**body)
        else:
            return self._sdk.detail(**body)

    def _handle_export(self, body):
        self._audit_log("export", details={"path": body.get("output_path"), "format": body.get("format", "json")})
        return self._sdk.export_memories(**body)

    def _handle_import(self, body):
        self._audit_log("import", details={"path": body.get("input_path")})
        return self._sdk.import_memories(**body)

    def _handle_health(self, body):
        return self._sdk.health_check()

    def _send_metrics(self):
        """输出 Prometheus 格式指标文本。

        Content-Type 遵循 Prometheus 文本格式规范 version=0.0.4。
        """
        try:
            body = get_metrics_collector().collect_all().encode("utf-8")
        except Exception as e:
            body = f"# 指标收集失败: {e}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self._add_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _generate_default_key() -> str:
    """生成默认随机 API Key（32 字节十六进制）。"""
    return secrets.token_hex(32)


def run_api(host: str = "0.0.0.0", port: int = 8765, storage_dir: str | None = None, config: dict | None = None):
    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK(storage_dir=storage_dir, config=config)
    OmniMemAPIHandler._sdk = sdk

    # 从 SDK 配置初始化安全中间件；未配置时默认生成随机密钥并启用认证
    api_key = sdk._config.get("api_key", "")
    if not api_key:
        api_key = _generate_default_key()
        logger.warning("REST API 未配置 api_key，已自动生成默认密钥（请查看控制台输出）")
    admin_token = sdk._config.get("admin_token", "") or os.environ.get("OMNIMEM_ADMIN_TOKEN", "")
    if not admin_token:
        admin_token = _generate_default_key()
        logger.warning("REST API 未配置 admin_token，已自动生成默认管理令牌（请查看控制台输出）")

    rate_limit = sdk._config.get("api_rate_limit_per_minute", 60)
    cors_origins = sdk._config.get("cors_allowed_origins", [])
    OmniMemAPIHandler._auth_middleware = AuthMiddleware(api_key=api_key)
    OmniMemAPIHandler._admin_auth_middleware = AdminAuthMiddleware(admin_token=admin_token)
    OmniMemAPIHandler._rate_limiter = RateLimiter(limit_per_minute=rate_limit)
    OmniMemAPIHandler._cors_allowed_origins = cors_origins

    # 记录活跃连接（服务启动时 +1）
    inc_active_connections()
    server = HTTPServer((host, port), OmniMemAPIHandler)
    print(f"OmniMem REST API running on http://{host}:{port}")
    print(f"API Key: {api_key}")
    print(f"Admin Token: {admin_token}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sdk.close()
        server.server_close()
        # 服务关闭时 -1
        dec_active_connections()


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_api(port=port)
