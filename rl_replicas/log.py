import logging
import sys

FORMATTER = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

def get_logger(name):
  """
  return the logger

  :param name: (str) name of a logger
  :return:
  """
  logger = logging.getLogger(name)

  # if (logger.hasHandlers()):
  #   logger.handlers.clear()

  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(FORMATTER)

  logger.addHandler(handler)
  logger.setLevel(logging.DEBUG)
  logger.propagate = False

  return logger
