"""A small exception type that carries a user-friendly message."""


class AppError(Exception):
    def __init__(self, message, code="error", status=400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status