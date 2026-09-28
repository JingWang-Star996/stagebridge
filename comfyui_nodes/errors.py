"""Public errors deliberately exclude request/response bodies and credentials."""


class RemoteTEError(RuntimeError):
    pass


class RetryableRemoteError(RemoteTEError):
    """Only transport failures, HTTP 429 and HTTP 5xx permit local fallback."""


class ProtocolError(RemoteTEError):
    pass


class VersionMismatch(ProtocolError):
    pass


class ConfigurationError(RemoteTEError):
    pass
