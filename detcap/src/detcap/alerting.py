import logging
import sys
from typing import NamedTuple, Union, Optional, Any

import numpy as np
from shapely.geometry import Polygon, Point

from detcap.inventory.station import Channel
from detcap.map import Map

_logger = logging.getLogger(__name__)


def parse_aggregator(aggr: str):
    if aggr == "max":
        return np.nanmax, "maximum"
    elif aggr == "min":
        return np.nanmin, "minimum"
    elif aggr == "avg":
        return np.nanmean, "average"
    elif aggr == "":
        return None, "None"
    else:
        raise ValueError(f"Unknown aggregator: {aggr}")


class Alert:
    def __init__(
        self,
        name: str,
        map: Map,
        metric: str,
        threshold: Union[float, int],
        aggregator: str,
        additional_msg: str = "",
        description: str = "",
        polygon: Optional[Any] = None,
        notify_after_cycles: Optional[int] = None,
        upload_after_cycles: Optional[int] = None,
    ):
        self.name = name
        self.map = map
        self.metric = metric
        self.threshold = threshold
        self.agg, self.agg_name = parse_aggregator(aggregator)
        self.additional_msg = additional_msg
        self.message: str = ""
        self.set_msg("Waiting on data")
        self.description = description
        self.latest_value = None
        self.in_alert = False
        self.alert_subsided = False
        self.cycles_in_alert = 0
        self.notify_after_cycles = (
            notify_after_cycles if notify_after_cycles is not None else sys.maxsize
        )
        self.upload_after_cycles = (
            upload_after_cycles if upload_after_cycles is not None else sys.maxsize
        )

        if polygon is not None:
            mask = np.zeros(self.map.lat_grid.shape, dtype=bool)
            for i, coord in self.map:
                if Point(coord.lon, coord.lat).intersects(polygon):
                    mask[i] = True
            self.mask = mask
        else:
            self.mask = np.ones(self.map.lat_grid.shape, dtype=bool)

    def _calculate(self):
        self.latest_value = self.agg(getattr(self.map, f"{self.metric}_grid")[self.mask])  # type: ignore

    def set_msg(self, msg: str):
        self.message = f"{msg}. {'Additional info: ' if self.additional_msg else ''}{self.additional_msg}"

    def update(self):
        old_value = self.latest_value
        self._calculate()
        if self.latest_value is None or np.isnan(self.latest_value):
            self.set_msg("Waiting on data")
            self.in_alert = False
        else:
            if self.latest_value >= self.threshold:
                self.alert_subsided = False
                if old_value is not None:
                    self.set_msg(
                        f"In alert: {self.agg_name} {self.metric} has exceeded threshold of {self.threshold} "
                        f"({old_value:.2f} => {self.latest_value:.2f})"
                    )
                else:
                    self.set_msg(
                        f"In alert: {self.agg_name} {self.metric} has exceeded threshold of {self.threshold} "
                        f"(initial reading {self.latest_value:.2f})"
                    )
                self.in_alert = True
            else:
                self.alert_subsided = self.in_alert
                if self.alert_subsided:
                    self.set_msg(
                        f"Okay: {self.agg_name} {self.metric} has returned below threshold of {self.threshold}")
                else:
                    self.set_msg(
                        f"Okay: {self.agg_name} {self.metric} is below threshold of {self.threshold}"
                    )
                
                self.in_alert = False

        self.cycles_in_alert = self.cycles_in_alert + 1 if self.in_alert else 0

    @property
    def should_notify(self):
        return self.cycles_in_alert >= self.notify_after_cycles

    @property
    def should_upload(self):
        return (self.cycles_in_alert >= self.upload_after_cycles) or self.alert_subsided

    @property
    def metadata(self):
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "last_value": None
            if self.latest_value is None or np.isnan(self.latest_value)
            else self.latest_value,
            "threshold": self.threshold,
            "in_alert": bool(self.in_alert),
            "should_notify": bool(self.should_notify),
            "should_upload": bool(self.should_upload),
            "alert_subsided": bool(self.alert_subsided),
            "cycles_in_alert": self.cycles_in_alert,
            "message": self.message,
        }

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class RelativeAlert(Alert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_value = None
        self.last_good_state_on_trigger = None

    @property
    def metadata(self):
        md = super().metadata
        md.update(
            {
                "last_good_state_on_trigger": self.last_good_state_on_trigger,
                "previous_value": self.previous_value,
            }
        )
        return md

    def _calculate(self):
        self.previous_value = self.latest_value
        super()._calculate()

    def update(self):
        """
        Relative alerting works by checking the difference between the two
        most recent measurements. If this difference is greater than
        the threshold, we return in alert and store the latest value
        that triggered the alert.

        On subsequent alerts, compare the latest value and triggering
        value. The alert remains in place until the difference between
        the latest value and triggering value drops below the threshold.
        """
        self._calculate()

        # Nothing to do
        if self.latest_value is None or self.previous_value is None:
            self.set_msg("Waiting on data")
            self.in_alert = False
        # Are we already triggered?
        elif self.last_good_state_on_trigger is not None:
            # If yes, check if we've returned to last known good state
            if self.latest_value <= self.last_good_state_on_trigger:
                self.set_msg(
                    f"Okay: {self.agg_name} {self.metric} has returned to good state "
                    f"({self.last_good_state_on_trigger:.2f} => {self.latest_value:.2f})"
                )
                self.last_good_state_on_trigger = None
                self.in_alert = False
                self.alert_subsided = False
            else:
                self.set_msg(
                    f"In alert: {self.agg_name} {self.metric} remains above good state "
                    f"({self.last_good_state_on_trigger:.2f})"
                )
                self.in_alert = True
                self.alert_subsided = False
        # Check if we've crossed the delta threshold
        elif self.latest_value - self.previous_value >= self.threshold:
            self.set_msg(
                f"In alert: {self.agg_name} {self.metric} has increased by at least threshold "
                f" {self.threshold} ({self.previous_value:.2f} => {self.latest_value:.2f})"
            )
            self.last_good_state_on_trigger = self.previous_value
            self.alert_subsided = False
            self.in_alert = True
        else:
            self.set_msg(
                f"Okay: {self.agg_name} {self.metric} has not increased by threshold {self.threshold} "
                f"({self.previous_value:.2f} => {self.latest_value:.2f})"
            )
            self.alert_subsided = False
            self.in_alert = False

        self.cycles_in_alert = self.cycles_in_alert + 1 if self.in_alert else 0
