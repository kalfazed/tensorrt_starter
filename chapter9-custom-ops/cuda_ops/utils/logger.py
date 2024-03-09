import colorlog
import logging
import numpy as np
import torch


class Logger():
    def __init__(self) -> None:
        self._log_colors_config = {
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red'}
        self._logger = self.set_logger()

    def set_logger(self):
        """Initialize a logger"""

        logger = logging.getLogger('logger_name')
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s(%(asctime)s) [%(levelname)s]:%(reset)s %(message)s',
            datefmt='%H:%M:%S',
            log_colors=self._log_colors_config
        )

        console_handler.setFormatter(console_formatter)

        if not logger.handlers:
            logger.addHandler(console_handler)

        console_handler.close()
        return logger

    def printTensorInformation(
        self,
        x: torch.Tensor,
        info: str = "",
        n: int = 5
    ):
        """print out first and last n value of a numpy array
        Args:
            x:    input array
            info: prefix of the output info
            n:    numbers of values to be printed
        """

        if x.device != 'cpu':
            x = x.cpu().numpy()

        if 0 in x.shape:
            self._logger.debug('%s:%s' % (info, str(x.shape)))
            self._logger.debug()
            return

        self._logger.debug('%s:%s' % (info, str(x.shape)))
        self._logger.debug(
            '\tSumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f' % (
                np.sum(abs(x)),
                np.var(x),
                np.max(x),
                np.min(x),
                np.sum(np.abs(np.diff(x.reshape(-1))))))
        self._logger.debug('\t%s ...  %s' % (
            x.reshape(-1)[:n],
            x.reshape(-1)[-n:]))

    def check(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weak=False,
        checkEpsilon=1e-5
    ):

        """check the differences between a and b
        Args:
            a:            input numpy array
            b:            input numpy array
            weak:         boolean value to decide if do the weak check
            checkEpsilon: float value, indicates that
                          the maxmium differences that can be tolerated
        """
        if a.device != 'cpu':
            a = a.cpu().numpy()

        if b.device != 'cpu':
            b = b.cpu().numpy()

        if weak:
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            res = np.all(np.abs(a - b) < checkEpsilon)
        else:
            res = np.all(a == b)
        diff0 = np.max(np.abs(a - b))
        diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
        self._logger.info("check:%s, absDiff=%f, relDiff=%f"
                          % (res, diff0, diff1))
        return res

    def debug(self, str):
        self._logger.debug(str)

    def info(self, str):
        self._logger.info(str)

    def warning(self, str):
        self._logger.warning(str)

    def error(self, str):
        self._logger.error(str)

    def critical(self, str):
        self._logger.critical(str)

    def setLevel(self, level):
        self._logger.setLevel(level)

    def test_logger(self):
        self._logger.setLevel(logging.DEBUG)
        self._logger.debug("debug")
        self._logger.info('info')
        self._logger.warning('warning')
        self._logger.error('error')
        self._logger.critical('critical')


if __name__ == '__main__':
    logger = Logger()
    # logger.test_logger()
    logger.info("this is a test")
