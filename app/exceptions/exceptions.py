class RAGException(Exception):
    """Base exception for application-specific errors."""


class DocumentNotFoundError(RAGException):
    pass


class UnsupportedDocumentTypeError(RAGException):
    pass


class InvalidChunkConfigurationError(RAGException):
    pass


class LLMUnavailableError(RAGException):
    pass