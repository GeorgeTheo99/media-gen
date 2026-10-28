"""Errors carried to clients in the OpenAI error envelope."""

from __future__ import annotations


class MediaError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        param: str | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param
        self.retry_after = retry_after

    @property
    def type(self) -> str:
        if self.status == 401:
            return "authentication_error"
        if self.status == 404:
            return "not_found_error"
        if self.status < 500:
            return "invalid_request_error"
        return "server_error"

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "code": self.code, "param": self.param}}


def invalid(message: str, param: str | None = None, code: str = "invalid_value") -> MediaError:
    return MediaError(400, code, message, param=param)


class InsufficientMemory(MediaError):
    def __init__(self, message: str):
        super().__init__(503, "insufficient_memory", message, retry_after=60)


class GenerationFailed(MediaError):
    def __init__(self, code: str, message: str):
        super().__init__(500, code, message)
