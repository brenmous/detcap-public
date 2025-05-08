from typing import Tuple, Sequence, Union, List, Dict, Any
from math import log10
import logging
import statistics

import numpy as np
import obspy as op
from shapely.geometry import Point

from detcap.utils import mla_polygons, get_peak_amplitudes, load_table, round_for_table, Coordinate
from detcap.inventory.station import Station, Channel
from detcap.map import Map

_logger = logging.getLogger(__name__)


class MLAMap(Map):
    def __init__(
        self,
        name: str,
        west: np.float32,
        south: np.float32,
        east: np.float32,
        north: np.float32,
        channel_amplitude_metric = "simulate_wa",
        channel_latency_metric = "latency",
        resolution: int = 1,
        n_stations: int = 4,
        stations: Sequence[Station] = [],
        magnitude_scale: Tuple[float, float] = (0.0, 6.0),
        magnitude_interval: float = 0.2,
        latency_scale: Tuple[float, float] = (
            0.0,
            600,
        ),
        latency_interval: float = 30,
        station_selection_kwargs: Dict[str, Any] = {}
    ):
        channel_properties = {"orientation": "vertical"}
        super().__init__(
            name,
            west=west,
            south=south,
            east=east,
            north=north,
            resolution=resolution,
            n_stations=n_stations,
            stations=stations,
            magnitude_scale=magnitude_scale,
            magnitude_interval=magnitude_interval,
            latency_scale=latency_scale,
            latency_interval=latency_interval,
            channel_properties=channel_properties,
            channel_amplitude_metric=channel_amplitude_metric,
            channel_latency_metric=channel_latency_metric,
            station_selection_kwargs=station_selection_kwargs
        )
        self.amplitude_units = "mm"
        self.magnitude_units = "mla"
        self.latency_units = "seconds"
        self.mla_lookup = self._assign_mla_function()
        self.tt_table = load_table("ak135_travel_times_P")
        self.attenuations = self._precalculate_attenuations()
        self.travel_times = self._precalculate_travel_times()

    def _build_coord_grid(self):
        lat_grid, lon_grid = super()._build_coord_grid()
        west_poly, east_poly, south_poly = mla_polygons()
        mask = np.zeros(lat_grid.shape, dtype=bool)
        for i, lat in np.ndenumerate(lat_grid):
            lon = lon_grid[i]
            point = Point(lon, lat)
            if not any(
                point.intersects(poly) for poly in (west_poly, east_poly, south_poly)
            ):
                mask[i] = True
        return np.where(mask, np.nan, lat_grid), np.where(mask, np.nan, lon_grid)

    def _assign_mla_function(self):
        def west_mla(dist):
            return (1.137 * log10(dist)) + (0.000657 * dist) + 0.66

        def east_mla(dist):
            return (1.34 * log10(dist / 100.0)) + (0.00055 * (dist - 100.0)) + 3.13

        def south_mla(dist):
            return (1.1 * log10(dist)) + (0.0013 * dist) + 0.7

        def no_zone(mag, dist):
            return np.nan

        lookup = {}
        west_poly, east_poly, south_poly = mla_polygons()
        for _, c in self:
            p = Point(c.lon, c.lat)
            if p.intersects(west_poly):
                fn = west_mla
            elif p.intersects(east_poly):
                fn = east_mla
            elif p.intersects(south_poly):
                fn = south_mla
            else:
                fn = no_zone

            lookup[c] = fn
        return lookup

    def amplitude_processor(self, trace_data: np.ndarray, sample_rate: float) -> float:
        values_as_mm = trace_data * 1e3
        peak_amplitudes, _ = get_peak_amplitudes(values_as_mm, sample_rate)
        return np.mean(peak_amplitudes)  # type: ignore

    @staticmethod
    def _deg2km(deg: np.float32):
        return np.float32(op.geodetics.degrees2kilometers(deg))

    def magnitude_attenuation(
        self,
        depth: float,
        station: Station,
        point: Coordinate,
    ) -> np.float32:
        distance = self._deg2km(self.distances[station][point])
        hypocentral = np.sqrt((depth**2) + (distance**2))
        return np.float32(self.mla_lookup[point](hypocentral))

    def latency_processor(self, trace_latencies: np.ndarray) -> float:
        return np.mean(trace_latencies) # type: ignore

    def travel_time(self, distance: np.float32) -> float:
        return self.tt_table[str(self.depth)][round_for_table(distance)]

    def station_selection(
        self,
        station_metrics: Dict[Station, Dict[str, float]],
        coordinate: Coordinate,
        additional_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Station], float, float]:
        # Take the closest n stations
        selected_stations = [
            k
            for k in sorted(
                station_metrics.keys(),
                key=lambda k: self.distances[k][coordinate],
            )
        ][: self.n_stations]
        return (
            selected_stations,
            statistics.median(
                station_metrics[s]["magnitude"] for s in selected_stations
            ),
            max(station_metrics[s]["ttd"] for s in selected_stations),
        )
