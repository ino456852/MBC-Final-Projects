import logging
import sys

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    if not logger.handlers:
        logger.addHandler(stream_handler)
    return logger

# settings 모듈을 임포트하여 DEBUG 값에 따라 로그 레벨을 동적으로 설정
from .config import settings 

log = get_logger(__name__)