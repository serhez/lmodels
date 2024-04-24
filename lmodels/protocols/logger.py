from typing import Any, Protocol


class Logger(Protocol):
    """An interface for a logger."""

    def info(
        self,
        *messages: str | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Records a log message with the `INFO` level.

        ### Parameters
        ----------
        `messages`: the messages to be logged.
        """

        ...

    def warn(
        self,
        *messages: str | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Records a log message with the `WARN` level.

        ### Parameters
        ----------
        `messages`: the messages to be logged.
        """

        ...

    def error(
        self,
        *messages: str | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Records a log message with the `ERROR` level.

        ### Parameters
        ----------
        `messages`: the messages to be logged.
        """

        ...

    def debug(
        self,
        *messages: str | dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Records a log message with the `DEBUG` level.

        ### Parameters
        ----------
        `messages`: the messages to be logged.
        """

        ...
