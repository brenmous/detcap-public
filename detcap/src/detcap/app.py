"""
This module houses the main Detcap app. This is where the config is
parsed, the GUI is constructed and displayed, where acquisition starts
and where updates are fired.
"""
import os
import logging
import json
import importlib
import time
from itertools import chain
from typing import List, Optional, Set, Tuple

import numpy as np
import redis
from obspy.core.utcdatetime import UTCDateTime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from detcap.map import Map
from detcap.product import DetcapProduct
from detcap import alerting
from detcap.inventory import get_inventory
from detcap.inventory.station import Station, stations_from_list
from detcap.acquisition import ConcurrentSeedlinkClient
from detcap.settings import (
    TEMP_INVENTORY_PATH,
    MAP_IMPORTS,
    LTA,
    UPDATE_CYCLE,
    ALERT_IMPORTS,
)
from detcap.utils import filter_station_names, geojson_to_polygon

_logger = logging.getLogger(__name__)


class DetcapApp:
    """
    Main app class for realtime monitoring.
    """

    def __init__(
        self,
        config_file: str,
        seedlink_server: str,
        inventory_file: Optional[str],
        fdsn_url: str,
        save_directory: str,
        name_tag: Optional[str],
    ):
        """
        Parameters
        ----------
        config_file
            Path to the YAML config file.
        seedlink_server
            URL for the seedlink server, e.g. 'localhost:18000'
        inventory_file
            Path to an inventory file. Will be used if it exists and is
            valid StationXML, otherwise FDSN acquired inventory will
            be stored in this file. If not provided, a temp file is
            used.
        fdsn_url
            URL for the FDSNWS server, e.g. 'http://localhost:8081'
        tag
            Tag to be appended to map names (e.g. stack name).
        save_directory
            Directory where map files will be saved.
        """
        self.slc = ConcurrentSeedlinkClient(seedlink_server)
        available_stations = self.slc.available_stations()
        self.products, self.stations = products_from_config(
            config_file,
            available_stations,
            inventory_file,
            fdsn_url,
            name_tag,
            save_directory,
        )
        _logger.info("App init complete")

    async def __call__(self):
        scheduler = AsyncIOScheduler()
        # If we expand for different LTAs, we need to group maps
        # by LTAs and schedule updates at the same time.
        trigger = IntervalTrigger(seconds=UPDATE_CYCLE)
        self.update_job = scheduler.add_job(self.update, trigger=trigger)
        self.slc.subscribe(self.stations)
        scheduler.start()
        _logger.info("Monitoring started")
        await self.slc.run()
       
    async def update(self):
        try:
            # Get data from the start of the update window, so data
            # will be delayed by 40 seconds. Gives high latency stations
            # a chance to catch up
            start = UTCDateTime.now() - UPDATE_CYCLE
            end = start + LTA
            for s in self.stations:
                await s.handle_traces(start, end)

            for p in self.products:
                p.update(start, end)

        except Exception:
            _logger.exception("Update failed")


def products_from_config(
    config_file: str,
    available_stations: List[str],
    inventory_file: Optional[str],
    fdsn_url: Optional[str],
    name_tag: Optional[str],
    save_directory: str,
) -> Tuple[List[DetcapProduct], List[Station]]:
    """
    Reads a YAML file and constructs the required objects to
    generate the configured maps.
    """
    # Config field constants
    _ALERTS = "alerts"
    _NORTH = "north"
    _SOUTH = "south"
    _EAST = "east"
    _WEST = "west"
    _RESOLUTION = "resolution"
    _METRIC = "metric"
    _NSTATIONS = "n_stations"
    _WHITELIST = "whitelist"
    _BLACKLIST = "blacklist"
    _MAGNITUDE_SCALE_LOWER = "magnitude_scale_lower"
    _MAGNITUDE_SCALE_UPPER = "magnitude_scale_upper"
    _MAGNITUDE_SCALE_INTERVAL = "magnitude_scale_interval"
    _LATENCY_SCALE_LOWER = "latency_scale_lower"
    _LATENCY_SCALE_UPPER = "latency_scale_upper"
    _LATENCY_SCALE_INTERVAL = "latency_scale_interval"
    _SKIP_UPLOAD = "skip_upload"
    _SKIP_UPLOAD_NOMINAL_FREQ = "skip_upload_frequency"
    _STATION_SELECTION_KWARGS = "station_selection"

    with open(config_file, "r") as f:
        cfg = json.load(f)

    # Get a list of stations for each map
    map_station_names = {}
    # Search for blacklist/whitelist files
    cfg_dir = os.path.dirname(config_file)

    for name, params in cfg["maps"].items():
        whitelist_file = params.get(_WHITELIST)
        if whitelist_file is not None:
            try:
                filtered = filter_station_names(available_stations, whitelist_file, include=True)
            except FileNotFoundError:
                whitelist_file = os.path.join(cfg_dir, os.path.basename(whitelist_file))
                filtered = filter_station_names(available_stations, whitelist_file, include=True)
        else:
            filtered = available_stations

        blacklist_file = params.get(_BLACKLIST)
        if blacklist_file is not None:
            try:
                filtered = filter_station_names(filtered, blacklist_file)
            except FileNotFoundError:
                blacklist_file = os.path.join(cfg_dir, os.path.basename(blacklist_file))
                filtered = filter_station_names(filtered, blacklist_file)

        map_station_names[name] = filtered

    # Combined collection of stations required by all maps
    all_station_names: Set[str] = set(chain.from_iterable(map_station_names.values()))

    # Request inventory data for the included stations
    inventory_file = get_inventory(fdsn_url, all_station_names, inventory_file)

    # Construct Station objects from inventory data
    all_stations = stations_from_list(
        all_station_names, inventory_file, raise_on_unavailable=False
    )
    # We no longer need the inventory after this. If it was temporarily
    # stored then delete it.
    if inventory_file == TEMP_INVENTORY_PATH:
        os.remove(inventory_file)

    products = []
    for name, params in cfg["maps"].items():
        metric = params.get(_METRIC, cfg["defaults"][_METRIC])
        stations = [
            all_stations[sname]
            for sname in map_station_names[name]
            if sname in all_stations
        ]
        east = params[_EAST]
        south = params[_SOUTH]
        west = params[_WEST]
        north = params[_NORTH]
        res = params[_RESOLUTION]
        nsta = params[_NSTATIONS]
        magnitude_scale_lower = params.get(
            _MAGNITUDE_SCALE_LOWER, cfg["defaults"][_MAGNITUDE_SCALE_LOWER]
        )
        magnitude_scale_upper = params.get(
            _MAGNITUDE_SCALE_UPPER, cfg["defaults"][_MAGNITUDE_SCALE_UPPER]
        )
        magnitude_scale_interval = params.get(
            _MAGNITUDE_SCALE_INTERVAL, cfg["defaults"][_MAGNITUDE_SCALE_INTERVAL]
        )
        latency_scale_lower = params.get(
            _LATENCY_SCALE_LOWER, cfg["defaults"][_LATENCY_SCALE_LOWER]
        )
        latency_scale_upper = params.get(
            _LATENCY_SCALE_UPPER, cfg["defaults"][_LATENCY_SCALE_UPPER]
        )
        latency_scale_interval = params.get(
            _LATENCY_SCALE_INTERVAL, cfg["defaults"][_LATENCY_SCALE_INTERVAL]
        )
        skip_upload = params.get(_SKIP_UPLOAD, cfg["defaults"][_SKIP_UPLOAD])
        skip_upload_freq_nominal = params.get(
            _SKIP_UPLOAD_NOMINAL_FREQ, cfg["defaults"][_SKIP_UPLOAD_NOMINAL_FREQ]
        )
        station_selection_kwargs = params.get(
            _STATION_SELECTION_KWARGS, {}
        )
        m = _dynamic_import(MAP_IMPORTS[metric.lower()])(
            name=name,
            west=west,
            south=south,
            east=east,
            north=north,
            resolution=res,
            n_stations=nsta,
            stations=stations,
            magnitude_scale=(magnitude_scale_lower, magnitude_scale_upper),
            magnitude_interval=magnitude_scale_interval,
            latency_scale=(latency_scale_lower, latency_scale_upper),
            latency_interval=latency_scale_interval,
            station_selection_kwargs=station_selection_kwargs
        )
        alerts = params.get(_ALERTS, [])
        alerts = alerts_from_config(alerts, m, cfg_dir)
        products.append(
            DetcapProduct(
                m,
                alerts,
                skip_upload,
                save_directory,
                skip_upload_freq_nominal,
                name_tag,
            )
        )
    return products, list(all_stations.values())


def alerts_from_config(alert_cfgs, m: Map, cfg_dir: str):
    _NAME = "name"
    _TYPE = "type"
    _METRIC = "metric"
    _AGGREGATOR = "aggregator"
    _POLYGON = "polygon"
    _POLYGON_PROPS = "poly_has_properties"
    _THRESHOLD = "threshold"
    _ADDITIONAL_MSG = "additional_message"
    _DESCRIPTION = "description"
    _NOTIFY = "notify_after_cycles"
    _UPLOAD = "upload_after_cycles"

    alerts = []
    for cfg in alert_cfgs:
        if _POLYGON in cfg:
            try:
                polygon = geojson_to_polygon(cfg[_POLYGON], cfg.get(_POLYGON_PROPS, {}))
            except FileNotFoundError:
                polygon = geojson_to_polygon(
                    os.path.join(cfg_dir, cfg[_POLYGON]), cfg.get(_POLYGON_PROPS, {})
                )
        else:
            polygon = None
        args = (cfg[_NAME], m, cfg[_METRIC], cfg[_THRESHOLD], cfg[_AGGREGATOR])
        kwargs = dict(
            additional_msg=cfg.get(_ADDITIONAL_MSG, ""),
            description=cfg.get(_DESCRIPTION, ""),
            polygon=polygon,
            notify_after_cycles=cfg.get(_NOTIFY),
            upload_after_cycles=cfg.get(_UPLOAD),
        )
        if cfg[_TYPE] == "relative":
            alerts.append(alerting.RelativeAlert(*args, **kwargs))
        else:
            alerts.append(alerting.Alert(*args, **kwargs))

    return alerts


def _dynamic_import(path):
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)
