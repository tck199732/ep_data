"""Framework-specific exception hierarchy."""


class HEPDataRetrievalError(Exception):
    """Base exception for expected framework failures."""


class ConfigurationError(HEPDataRetrievalError):
    """The dataset registry is missing or invalid."""


class UnknownDatasetError(ConfigurationError):
    """A requested dataset key is not registered."""


class DownloadError(HEPDataRetrievalError):
    """A DOI could not be resolved or its archive could not be downloaded."""


class CacheError(HEPDataRetrievalError):
    """Cached raw data is unavailable or incomplete."""


class UnsafeArchiveError(DownloadError):
    """An archive contains unsafe or ambiguous members."""


class ParserLoadError(HEPDataRetrievalError):
    """A configured parser cannot be imported or does not meet the contract."""


class ProcessingError(HEPDataRetrievalError):
    """A parser failed or its output could not be committed."""


class DiagnosticsLoadError(HEPDataRetrievalError):
    """A diagnostics module cannot be imported or does not meet the contract."""


class DiagnosticsError(HEPDataRetrievalError):
    """Diagnostic plotting failed or its output could not be committed."""
