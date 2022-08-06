import logging
import os
import sys
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class MetricsManager:
    """
    Metrics manager logs for stdout, json and tensorboard

    :param log_dir: (str) Log directory.
    """

    def __init__(self, log_dir: str = "."):
        self.log_dir = log_dir

        self.tensorboard_writer = SummaryWriter(
            os.path.join(self.log_dir, "tensorboard")
        )

    def record_scalar(
        self,
        tag: str,
        scalar: float,
        total_steps: Optional[int] = None,
        tensorboard: bool = False,
    ) -> None:
        print("{}: {:<8.3g}".format(tag, scalar))

        if tensorboard:
            if total_steps is None:
                logger.warning("total_steps argument is required for tensorboard")
            self.tensorboard_writer.add_scalar(tag, scalar, total_steps)

    def dump(self) -> None:
        sys.stdout.flush()
        self.tensorboard_writer.flush()

    def close(self) -> None:
        self.tensorboard_writer.close()
