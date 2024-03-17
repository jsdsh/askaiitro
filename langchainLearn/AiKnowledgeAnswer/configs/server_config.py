import sys
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 7099,
}
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 7861,
}
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}
# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 300.0