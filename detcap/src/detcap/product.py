"""
The creatively named "product" unifies Maps, Alerts and whatever
else may be added, into a single container.

This allows passing data between maps and alerts without nonsensical
ownership and awkward interfaces. It also removes the responsibility
of Map being the parent of everything, which is good as it's already
a busy and specialised class.

It provides a place for data writing and archiving methods, removing
them from the App event loops.

It also allows us to construct a neat little unified package to upload
to SKIP.
"""
import os
import json
import logging
import fcntl
from typing import List, Optional, Tuple
from functools import lru_cache
from datetime import datetime

import numpy as np

from detcap.map import Map
from detcap.alerting import Alert
from detcap.skip import upload_detcap_product
from detcap.settings import OLD_MAPS_TO_KEEP
from detcap import DETCAP_SCHEMA_VERSION, DETCAP_COMMIT_HASH, Float32Encoder

_logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_logfile():
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "baseFilename"):
            return handler.baseFilename


class DetcapProduct:
    def __init__(
        self,
        map: Map,
        alerts: List[Alert],
        skip_upload: bool,
        save_directory: str,
        upload_frequency: int = 10,
        name_tag: Optional[str] = None,
    ):
        """ """
        self.map = map
        self.name = map.name
        self.alerts = alerts
        self.time_window_start = None
        self.time_window_end = None
        self.last_updated = None
        self.skip_upload = skip_upload
        self.upload_frequency = upload_frequency
        self.name_tag = name_tag
        self.skip_product_url: Optional[str] = None
        self.cycles_since_upload = 0

        self.save_directory = save_directory

        self.map_directory = os.path.join(self.save_directory, self.name) 
        os.makedirs(self.map_directory, exist_ok=True)

        self._files = []
        

    def __str__(self):
        return self.name

    @property
    def _human_last_updated(self):
        if self.last_updated is None:
            return "Never"
        else:
            return self.last_updated.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def _human_time_window_start(self):
        if self.time_window_start is None or self.time_window_end is None:
            return "Not Applicable"
        else:
            return self.time_window_start.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def _human_time_window_end(self):
        if self.time_window_end is None:
            return "Not Applicable"
        else:
            return self.time_window_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def metadata(self):
        return {
            "map": self.map.metadata,
            "alerts": {a.name: a.metadata for a in self.alerts},
            "last_updated": self._human_last_updated,
            "time_window_start": self._human_time_window_start,
            "time_window_end": self._human_time_window_end,
            "detcap_schema_version": DETCAP_SCHEMA_VERSION,
            "detcap_commit_hash": DETCAP_COMMIT_HASH,
            "skip_product_url": self.skip_product_url,
            "source": {
                "name": self.name_tag,
            },
        }

    @property
    def in_alert(self):
        return any(a.in_alert for a in self.alerts)

    @property
    def should_upload(self):
        return self.skip_upload and (
            self.cycles_since_upload >= self.upload_frequency
            or any(alert.should_upload for alert in self.alerts)
        )

    def write_data(self, map_data) -> str:
        """
        Write the map data and metadata together as a JSON file.
        """
        map_path = os.path.join(self.map_directory, self.map_file)
        map_data['metadata'] = self.metadata
        with open(map_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(map_data, f, cls=Float32Encoder)
        return map_path

    def update(self, start_time, end_time):
        self.cycles_since_upload += 1

        self.time_window_start = start_time
        self.time_window_end = end_time
        self.map.update()
        for a in self.alerts:
            a.update()
            if a.in_alert:
                _logger.info("Alert %s for map %s is in alert", a, self.map)
        self.last_updated = datetime.utcnow()

        map_data = self.map.as_geojson()

        if self.should_upload:
            try:
                skip_product = upload_detcap_product(
                    self,
                    map_data,
                    self.name_tag,
                    get_logfile(),
                )
                self.cycles_since_upload = 0
                self.skip_product_url = f"{os.environ['SKIP_URL']}/{skip_product.key}/info"
            except Exception:
                _logger.exception("%s: SKIP upload failed", self)
        else:
            self.skip_product_url = None
            _logger.debug(
                "%s: Skipping upload, uploading in %s cycles",
                self,
                self.upload_frequency - self.cycles_since_upload,
            )

        path = self.write_data(map_data)
        if OLD_MAPS_TO_KEEP > 0:
            self._files.append(path)
            if len(self._files) > OLD_MAPS_TO_KEEP:
                old_file = self._files.pop(0)
                _logger.info("%s: removing old file %s", self, old_file)
                os.remove(old_file)

    @property
    def map_file(self):
        return f"{self.name}_{self._human_last_updated.replace(':','')}.json"
