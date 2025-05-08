# Copyright 2019 Geoscience Australia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
from datetime import datetime
import logging
import logging.handlers
import errno
import os
import sys
import json

import numpy as np

class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.nan):
            return None
        return json.JSONEncoder.default(self, obj)

DETCAP_SCHEMA_VERSION = "1.1.0"
if os.path.exists(os.path.join("opt", "detcap", "DETCAP_COMMIT_HASH")):
    with open(os.path.join("opt", "detcap", "DETCAP_COMMIT_HASH"), "r") as f:
        DETCAP_COMMIT_HASH = f.read().strip()
else:
    DETCAP_COMMIT_HASH = "local"


class MetricConstants:
    """
    Container for metric name constants.
    """

    LATENCY = "latency"
    LATENCY_LABEL = "Time to detection [seconds] (stream latency + travel time)"
    LATENCY_UNITS = "s"
    THRESHOLD_MLA = "threshold_mla"
    THRESHOLD_MLA_LABEL = "Detection threshold [MLa] (90% confidence)"
    THRESHOLD_MLA_UNITS = "MLa"
    ALL_METRICS = [LATENCY, THRESHOLD_MLA]

    PLAYBACK_COMPATIBLE = [THRESHOLD_MLA]


def configure_exception_handler():
    """
    Configures an exception handler function which overrides the default
    excepthook. Allows for uncaught exceptions to be logged.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.getLogger().critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def configure_logger(
    level="INFO",
    log_file=None,
    quiet=False,
):
    """
    Set the log file and logging level. Override sys.excepthook with
    handle_exception to allow all uncaught exceptions to be logged.

    Args:
        directory (str): Directory where logs are stored.
        level (int): The logging level to use.
    """
    LOGGING_LEVELS = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
    }

    formatter = logging.Formatter(
        fmt="P%(process)d: %(levelname)s:%(name)s: %(message)s "
        "(%(asctime)s; %(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(LOGGING_LEVELS[level])

    # Turn down noisy loggers
    logging.getLogger("obspy").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if log_file is not None:
        try:
            os.makedirs(os.path.dirname(log_file))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        fh = logging.handlers.RotatingFileHandler(
            log_file, encoding="utf-8", maxBytes=100000000, backupCount=1
        )
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    if not quiet:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)
