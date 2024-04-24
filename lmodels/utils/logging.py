from lmodels.protocols import Logger


class NullLogger:
    """A logger that does nothing."""

    info = Logger.info
    warn = Logger.warn
    error = Logger.error
    debug = Logger.debug
