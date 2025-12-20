import logging

app_logger = logging.getLogger("sketch-search")
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
app_logger.addHandler(_console_handler)
