# Copyright 2019 Geoscience Australia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains methods for loading the Inventory.
"""
import os
import logging
import urllib3
from contextlib import nullcontext
from itertools import product
from collections import defaultdict
from typing import Sequence, Optional, NamedTuple, List, Iterable
from xml.etree import ElementTree as ET

import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.inventory.inventory import Inventory

from detcap.settings import (
    TEMP_INVENTORY_PATH,
    ALLOWED_BAND_CODES,
    ALLOWED_INSTRUMENT_CODES,
    ALLOWED_ORIENTATION_CODES,
)
from detcap.utils import PathType


_logger = logging.getLogger(__name__)


class FDSN:
    def __init__(self, fdsn_url):
        if not fdsn_url.startswith("http://"):
            self.base_url = f"http://{fdsn_url}"
        else:
            self.base_url = fdsn_url

    @staticmethod
    def station_request_builder(station_ids, level="response"):
        # Part of IA hotfix. See `station.stations_from_list`.
        # We always want AU.KMBL for its response regardless of if
        # it's requested or not.
        station_ids.add("AU.KMBL")

        # KIM03, KIM05 hotfix.
        station_ids.add("AU.CARL")

        allowed_channels = ",".join(
            "".join(t)
            for t in product(
                ALLOWED_BAND_CODES, ALLOWED_INSTRUMENT_CODES, ALLOWED_ORIENTATION_CODES
            )
        )
        now = UTCDateTime.now().strftime("%Y-%m-%dT%H:%M:%S")
        request = [f"level={level}"]
        request += ["format=sc3ml"]
        request += [
            f"{' '.join(s.split('.'))} * {allowed_channels} {now} {now}"
            for s in station_ids
        ]
        request_body = "\n".join(request)
        return request_body

    def request_inventory(self, ids, savepath):
        url = f"{self.base_url}/fdsnws/station/1/query"
        http = urllib3.PoolManager()
        r = http.request(
            "POST", url, body=self.station_request_builder(ids), preload_content=True
        )
        with open(savepath, "wb") as f:
            f.write(r.data)
            # for chunk in r.stream(1024):
            #   f.write(chunk)
        # r.release_conn()
        return savepath


def get_inventory(
    fdsn_url: Optional[str], stations: Iterable[str], filepath: Optional[str]
):
    """
    Gets response level inventory from FDSN.
    If the provided file already exists, this function does nothing.

    Note we don't check if the provided file is valid inventory if
    it exists, so ensure the path either points to a STATIONXML file
    or a non-existent file.

    Parameters
    ----------
    fdsn_url
        Base url of the fdsn service, e.g. "http://localhost:8081".
        FDSN is used if the provided inventory file doesn't exist.
    stream_ids
        A list of stations to get inventory for. Formatted as SEED ID
        e.g. 'AU.CARL'.
    filepath
        Path to the inventory to load. If the file doesn't exist,
        the inventory will be downloaded from FDSN and saved to this
        file. If no file is provied, the inventory is written to
        temporary storage.
    """
    _logger.info("Loading inventory")
    if filepath is None:
        _logger.info(
            "No inventory file provided, retrieving from FDSN and storing to temp file"
        )
        filepath = TEMP_INVENTORY_PATH
    elif os.path.exists(filepath):
        _logger.info(f"Found existing inventory at '{filepath}', using this inventory")
        return filepath
    else:
        _logger.info(f"Retrieving inventory from FDSN and storing to '{filepath}'")
    if fdsn_url is None:
        raise ValueError(
            f"Inventory file '{filepath}' doesn't exit and no FDSN URL was provided"
        )
    else:
        client = FDSN(fdsn_url)
        client.request_inventory(stations, filepath)
        _logger.info("Inventory written to %s", filepath)
        return filepath


def parse_inventory(filepath: PathType) -> Inventory:
    return obspy.read_inventory(filepath)
