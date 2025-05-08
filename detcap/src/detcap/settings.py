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
"""
The Settings module handles the setting of various things like directories,
update cycle length and log levels.

Some entries are only set here, while others can be set as environment
variables. The behaviour of the env vars is that this module will check
for a set env var. If not found, it sets a default value.

.. note::
    Variables marked as being environment variables are prepended with 
    **DETCAP_** in their environment variable form. E.g. to set log level 
    to debug via the environment, you would set **DETCAP_LOG_LEVEL=DEBUG**.
"""
import os
import uuid
import tempfile
import pkgutil


# When running in realtime mode, this is the size of the RingBuffer stores
METRIC_BUFFER_SIZE = 12000

# LTA in seconds to process on
LTA = 20

# How often maps are recalculated in seconds
UPDATE_CYCLE = 60

OLD_MAPS_TO_KEEP = 3

# -- PATHS -- #
LOG_FILE = os.environ.get("DETCAP_LOG_FILE", None)
"""str: Log directory. **Environment variable**."""

SAVE_DIRECTORY = os.environ.get("DETCAP_SAVE_DIRECTORY", tempfile.gettempdir())

# -- ACQUISITION -- #
INVENTORY = os.environ.get("DETCAP_INV_FILE", None)
"""str: Path to an inventory file. If None, then the inventory is loaded from FDSN.
    **Environment variable**."""

TEMP_INVENTORY_PATH = os.path.join(
    tempfile.gettempdir(), f"detcap_temp_inv_{uuid.uuid4().hex}.xml"
)
"""str: If no file is provided for inventory, then it gets written to this temporary file."""

FDSN_SERVER = os.environ.get("DETCAP_FDSN_SERVER", "IRIS")
"""str: URL of the FDSN server (used to get inventory when inventory file is provided).
    **Environment variable**."""

SEEDLINK_SERVER = os.environ.get(
    "DETCAP_SEEDLINK_SERVER", "rtserve.iris.washington.edu"
)
"""str: URL of the Seedlink server. **Environment variable**."""

CONFIG_FILE = os.environ.get("DETCAP_CONFIG_FILE", None)
"""str: Path of the config to use."""

# -- LOGGING -- #
LOG_LEVEL = os.environ.get("DETCAP_LOG_LEVEL", "INFO")
"""str: Sets the log level (DEBUG, INFO, WARNING, CRITICAL).
    **Environment variable**."""

LOG_QUIET = os.environ.get("DETCAP_LOG_QUIET", False)
"""str: Whether to print log messages to stdout. **Environment variable**."""

# -- CHANNEL FILTERING -- #
# Users can filter at the network/station level. We filter here
# on channels - in general there's only a subset of channels we're
# interested in for all tasks. This may change in future, and further
# filtering can be done per map.
# B = broadband, H = high broadband, S = short period
ALLOWED_BAND_CODES = ["B", "H", "S"]
# H = high gain seismometer, L = low gain seismometer
ALLOWED_INSTRUMENT_CODES = ["H", "L"]
# Z = vertical, N = north-south, E = east-west
# 1 and 2 are used for horizontal orientation when alignment is outside of 5 degrees of east/north
ALLOWED_ORIENTATION_CODES = ["Z", "N", "E", "1", "2"]

# -- IMPORTING -- #
MAP_IMPORTS = {"mla": "detcap.map.mla.MLAMap", "mb": "detcap.map.mb.MBMap"}
ALERT_IMPORTS = {
    "latency": "detcap.alerting.RegionalLatencyAlert",
    "magnitude": "detcap.alerting.RegionalMagnitudeAlert",
    "station_status": "detcap.alerting.StationStatusAlert",
}
