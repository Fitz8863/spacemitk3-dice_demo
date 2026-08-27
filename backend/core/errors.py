"""统一错误类型与响应格式。

所有业务错误继承 DiceArenaError，携带 HTTP 状态码和机器可读的错误码。
前端可根据 code 做针对性处理（如 GAME_DISABLED 显示"即将开放"）。
"""


class DiceArenaError(Exception):
    """Dice Arena 业务错误基类。"""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status

    def to_dict(self) -> dict:
        """返回 JSON 可序列化的字典。"""
        return {"error": self.message, "code": self.code}


# ---- 游戏相关错误 ----


class GameNotFoundError(DiceArenaError):
    """游戏未注册。"""

    def __init__(self, game_id: str):
        super().__init__(
            message=f"游戏未找到: {game_id}",
            code="GAME_NOT_FOUND",
            status=404
        )


class GameDisabledError(DiceArenaError):
    """游戏已禁用（enabled=false）。"""

    def __init__(self, game_id: str):
        super().__init__(
            message=f"游戏 {game_id} 暂未开放",
            code="GAME_DISABLED",
            status=403
        )


class GameConfigError(DiceArenaError):
    """游戏配置格式错误（manifest.json 无效）。"""

    def __init__(self, game_id: str, detail: str):
        super().__init__(
            message=f"游戏 {game_id} 配置错误: {detail}",
            code="GAME_CONFIG_ERROR",
            status=500
        )


# ---- 组件相关错误 ----


class ComponentNotFoundError(DiceArenaError):
    """组件未注册。"""

    def __init__(self, component_id: str):
        super().__init__(
            message=f"组件未找到: {component_id}",
            code="COMPONENT_NOT_FOUND",
            status=500
        )


class ComponentNotReadyError(DiceArenaError):
    """组件健康检查失败（如 TTS 服务未启动）。"""

    def __init__(self, component_id: str, detail: str):
        super().__init__(
            message=f"组件 {component_id} 未就绪: {detail}",
            code="COMPONENT_NOT_READY",
            status=503
        )


class VisionError(DiceArenaError):
    """视觉识别失败（YOLOv8 进程错误）。"""

    def __init__(self, detail: str):
        super().__init__(
            message=f"视觉识别失败: {detail}",
            code="VISION_ERROR",
            status=500
        )


# ---- 任务相关错误 ----


class JobNotFoundError(DiceArenaError):
    """分析任务 ID 不存在。"""

    def __init__(self, job_id: str):
        super().__init__(
            message=f"任务未找到: {job_id}",
            code="JOB_NOT_FOUND",
            status=404
        )


class JobAlreadyExistsError(DiceArenaError):
    """单任务锁：已有任务在运行。"""

    def __init__(self, active_job_id: str):
        super().__init__(
            message=f"已有任务 {active_job_id} 正在运行，同一时间只允许一个分析任务",
            code="JOB_ALREADY_EXISTS",
            status=409
        )


class JobTimeoutError(DiceArenaError):
    """任务超时。"""

    def __init__(self, job_id: str, timeout_seconds: int):
        super().__init__(
            message=f"任务 {job_id} 超时（{timeout_seconds} 秒）",
            code="JOB_TIMEOUT",
            status=504
        )


class JobCancelledError(DiceArenaError):
    """任务被用户取消。"""

    def __init__(self, job_id: str):
        super().__init__(
            message=f"任务 {job_id} 已取消",
            code="JOB_CANCELLED",
            status=499  # Client Closed Request (nginx convention)
        )


# ---- TTS 相关错误 ----


class TtsValidationError(DiceArenaError):
    """TTS 请求参数校验失败。"""

    def __init__(self, detail: str):
        super().__init__(
            message=f"TTS 参数错误: {detail}",
            code="TTS_VALIDATION_ERROR",
            status=400
        )


class TtsServiceError(DiceArenaError):
    """TTS 服务调用失败（llama-server 错误）。"""

    def __init__(self, detail: str):
        super().__init__(
            message=f"TTS 服务错误: {detail}",
            code="TTS_SERVICE_ERROR",
            status=502
        )


# ---- 请求相关错误 ----


class InvalidRequestError(DiceArenaError):
    """请求格式错误（JSON 解析失败、缺少必需字段等）。"""

    def __init__(self, detail: str):
        super().__init__(
            message=f"请求格式错误: {detail}",
            code="INVALID_REQUEST",
            status=400
        )
