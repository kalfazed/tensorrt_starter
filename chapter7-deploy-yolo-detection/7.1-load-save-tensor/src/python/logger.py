import logging
import colorlog
import os

log_colors_config = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'bold_red',
}
current_path = os.path.dirname(__file__)
log_path     = current_path + "/../../logs/info.log"

def set_logger(log_path):
    logger           = logging.getLogger('logger_name')
    console_handler  = logging.StreamHandler()
    file_handler     = logging.FileHandler(filename=log_path, mode='a', encoding='utf8')

    # 日志输出的格式(文件)
    file_formatter   = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d::%H:%M:%S'
    )

    # 日志输出的格式(控制台)
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(levelname)s]:%(reset)s %(message)s',
        log_colors=log_colors_config
    )

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    file_handler.close()
    console_handler.close()

    return logger, file_handler, console_handler

def init_logger(log_path=log_path):
    logger, _, console_handler = set_logger(log_path)
    logger.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    return logger


def test_logger(log_path):
    logger, file_handler, console_handler = set_logger(log_path)

    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    logger.debug("debug")
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

if __name__ == '__main__':
    test_logger("test.log")
