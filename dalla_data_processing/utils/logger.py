"""
Structured logging configuration using structlog.

This module provides a centralized logging setup for the application,
with support for both JSON and console output formats.
"""

import logging
import sys

import structlog
from structlog.types import EventDict, Processor


def rename_event_key(_, __, event_dict: EventDict) -> EventDict:
    """
    Log entries keep the text message in the `event` field, but some systems
    use the `message` field. This processor moves the value from one field to
    the other.
    See https://github.com/hynek/structlog/issues/35#issuecomment-591321744
    """
    event_dict["message"] = event_dict.pop("event")
    return event_dict


def drop_color_message_key(_, __, event_dict: EventDict) -> EventDict:
    """
    Drop the `color_message` key from the event dict if it exists.
    Some loggers add this redundant field which we don't need.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(log_format: str = "console", log_level: str = "INFO"):
    """
    Configure structured logging for the application.

    Args:
        log_format: Output format - either "json" or "console" (default: "console")
        log_level: Minimum log level - DEBUG, INFO, WARNING, ERROR, or CRITICAL (default: "INFO")
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        drop_color_message_key,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        # We rename the `event` key to `message` only in JSON logs
        # The pretty ConsoleRenderer looks for `event`
        shared_processors.append(rename_event_key)
        # Format the exception only for JSON logs, as we want to pretty-print them when
        # using the ConsoleRenderer
        shared_processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=shared_processors
        + [
            # Prepare event dict for `ProcessorFormatter`.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log_renderer: structlog.types.Processor
    if log_format == "json":
        log_renderer = structlog.processors.JSONRenderer()
    else:
        log_renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within
        # structlog.
        foreign_pre_chain=shared_processors,
        # These run on ALL entries after the pre_chain is done.
        processors=[
            # Remove _record & _from_structlog.
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            log_renderer,
        ],
    )

    handler = logging.StreamHandler()
    # Use OUR `ProcessorFormatter` to format all `logging` entries.
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        Log any uncaught exception instead of letting it be printed by Python
        (but leave KeyboardInterrupt untouched to allow users to Ctrl+C to stop)
        See https://stackoverflow.com/a/16993115/3641865
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def get_logger(name: str = "dalla") -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (default: "dalla")

    Returns:
        A configured structlog logger instance
    """
    return structlog.stdlib.get_logger(name)


# Default logger instance for the application
logger = get_logger("dalla")
