"""tools.registry 的测试 mock，避免真实 Hermes 运行时依赖。"""


def tool_error(msg):
    return f"ERROR: {msg}"


def tool_result(data):
    return str(data)
