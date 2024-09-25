import atexit
import json
import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import pathlib
import datetime as dt
from typing import override, Any


def setup_logging() -> None:
    """
    Configures logging
    If user defines 'logging_config.json' in the directory
    where the main file exists than it is loaded as logging
    config, otherwise default photon_weave logging is used
    """
    config_file = pathlib.Path("logs/config.json")
    user_config_file = pathlib.Path("logging_config.json")

    if user_config_file.is_file():
        config_file = user_config_file
    else:
        config_file = pathlib.Path(__file__).parent.resolve() / "config.json"
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)

    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


class PhotonWeaveJSONFormatter(logging.Formatter):
    """
    Custom Photon Weave JSON Formatter
    Attributes:
        fmt_keys (dict): keys with values that will get logged

    Methods:
        format: Formats the records into a dict
        _prepare_log_dict: Prepares the actual log dict
    """

    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }

        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)
        for key, value in record.__dict__.items():
            if key not in message.keys():
                message[key] = value
        return message


class RotatingFileHandlerWithDir(RotatingFileHandler):
    """
    Custom implementation of RotatingFileHandler, which
    creates log directory and file if it does not exist
    already
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Ensure the directory exists
        log_file_path = kwargs.get("filename")
        if log_file_path:
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

        super().__init__(*args, **kwargs)
