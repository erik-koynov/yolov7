# init
from .yolo import Model
import logging
logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(name)s: %(filename)s: %(funcName)s]- %(levelname)s - %(message)s'))
logger.addHandler(handler)