import logging

train_logger = logging.getLogger("train")
_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
train_logger.addHandler(_console_handler)

dataset_logger = logging.getLogger("dataset")
_dataset_handler = logging.StreamHandler()
_dataset_handler.setFormatter(_formatter)
dataset_logger.addHandler(_dataset_handler)
