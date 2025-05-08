"""
Metric map class.
"""
from typing import Tuple, Sequence, NamedTuple, Optional, Dict, List, Any, Iterator
from dataclasses import dataclass
from collections import defaultdict
from functools import cached_property
import logging
import sys
import json
from math import log10

import geojsoncontour
import numpy as np
import obspy as op
from matplotlib import pyplot

from detcap.utils import (
    haversine,
    Coordinate
)
from detcap.inventory.station import Station, Channel
from detcap.settings import LTA

_logger = logging.getLogger(__name__)


class Map:
    """
    A map for calculating STA/LTA magnitude thresholds and their latencies.
    Intended to be subclassed, providing the following:
    """

    def __init__(
        self,
        name: str,
        west: np.float32,
        south: np.float32,
        east: np.float32,
        north: np.float32,
        channel_amplitude_metric: str,
        channel_latency_metric: str,
        resolution: int = 1,
        n_stations: int = 4,
        snr: float = 3.0,
        depth: float = 15.0,
        stations: Sequence[Station] = [],
        magnitude_scale: Tuple[float, float] = (0.0, 6.0),
        magnitude_interval: float = 0.2,
        latency_scale: Tuple[float, float] = (0, 600),
        latency_interval: float = 30,
        channel_properties={},
        station_selection_kwargs={}
    ):
        """
        Parameters
        ----------
        name
            Name of the map. Must be unique.
        east, south, west, north
            Bounds of the grid.
        resolution
            Size of the grid cells in degrees (resolution x resolution).
        n_stations
            List of :obj:`Station` objects that are relevant to this
            region (typically the full list of everything you're
            subscribing to).
        snr
            The signal-to-noise ratio that must be achieved by a signal
            against a station's current activity to be detected.
        all_streams
            A list of Stream objects. This is should contain every
            possible Stream we can subscribe to. Streams within
                        lta=lta,
            the selected radius of each grid cell are listed as
            monitoring that cell.
        *_scale
            Min/max of the colormap scale. Values above max will be
            displayed as grey. TODO: min.
        *_interval
            A value in the same units as the selected metric. This
            is the interval at which contour lines will be drawn
            and at which the colormap will be segmented. The upper limit
            is max value set by contour_cmap_scale.

            .. example::

                If we are monitoring latency with a cmap scale of
                `[0, 120]` and contour interval of 30, contour lines
                will be drawn at 0s, 30s, 60s, 90s, and 120s.

            A value of `sys.maxsize` is added to the end of the contour
            intevals. This ensures values over the cmap scale are still
            displayed, and will appear as a grey-filled contour.
        station_good_color, station_no_data_color, station_bad_data_color
            Name of a matplotlib color. Used for displaying station
            status.

        Attributes
        ----------
        magntidue_grid
            A 2D array. This contains the magntidue data
            for the map and is used to generate images. Index using
            [y][x].
        latency_grid
            A 2D array. This contains the latency data
            for the map and is used to generate images. Index using
            [y][x].
        lat_grid, lon_grid
            2D arrays containing the coordinates of the grid.
            Index using [y][x].
        _crd2sta
            A mapping of cell coordinates to a list of stations.
            These are the streams within the radius of the grid point.
        _station_set
            A Set of streams that contains all the stations
            involved with this map.
        station_status
            A mapping of stations to statuses, relfecting the state
            of the station over the last update/LTA period according
            to this map.
        time_window_start
            The start of the time window of the data used in this map.
            Optional and not used in calculation, only for metadata
            purposes.
        time_window_end
            The end of the time window of the data used in this map.
            Optional and not used in calculation, only for metadata
            purposes.
        """
        self.name = name
        self.west = np.float32(west)
        self.east = np.float32(east)
        self.north = np.float32(north)
        self.south = np.float32(south)
        self.resolution = resolution
        self.n_stations = n_stations
        self.n_stations_hard_limit = True
        self.snr = snr
        self.depth = depth

        self.lat_grid, self.lon_grid = self._build_coord_grid()

        self.magnitude_grid = np.empty_like(self.lat_grid)
        self.magnitude_grid.fill(np.nan)
        self.magnitude_explanation = defaultdict(list)

        self.latency_grid = np.empty_like(self.lat_grid)
        self.latency_grid.fill(np.nan)
        self.latency_explanation = defaultdict(list)

        self.stations = stations
        self.channel_properties = channel_properties
        self.channel_amplitude_metric = channel_amplitude_metric
        self.channel_latency_metric = channel_latency_metric
        self._register_metric_with_stations()
        self.distances = self._precalculate_distances()

        # Subclasses should call _precalculate_attenuations and _precalculate_travel_times
        # in their init after loading travel time table and doing any other setup required
        # (e.g. creating polygon lookup table for mla).
        self.attenuations = {}
        self.travel_times = {}

        self.channel_status = {}
        for station in self.stations:
            self.channel_status[station] = {}
            for channel in station.all_registered_channels:
                self.channel_status[station][channel.code] = {}
                self.channel_status[station][channel.code]["status"] = Channel.NO_DATA
                self.channel_status[station][channel.code]["message"] = (
                    (
                        f"Last time data was seen for this channel: "
                        f"{channel.data_last_received.datetime.strftime('%Y-%m-%dT%H%M%S')}"
                    )
                    if channel.data_last_received is not None
                    else "Never"
                )

        self.amplitude_units = "undefined"

        self.magnitude_contours = self._prepare_contours(
            magnitude_scale, magnitude_interval
        )
        self.magnitude_units = "undefined"

        self.latency_contours = self._prepare_contours(latency_scale, latency_interval)
        self.latency_units = "undefined"

        self.station_selection_kwargs = station_selection_kwargs

    def amplitude_processor(self, trace_data: np.ndarray, sample_rate: float) -> float:
        """
        Processes a float array of trace data (in m/s), returning an LTA
        amplitude value.

        Parameters
        ----------
        trace_data
            A float array of trace data.
        sample_rate
            Sample rate in Hz of the data.
        """
        raise NotImplementedError("Implement on subclass")

    def magnitude_attenuation(
        self, depth: float, station: Station, point: Coordinate 
    ) -> Optional[np.float32]:
        """
        Computes a magnitude given the relevant parameters.

        Parameters
        ----------
        sta_magnitude
            log10(x * snr) were x is the value output by
            amplitude_processor.
        distance
            Epicentral distance in degrees.
        depth
            Event depth in km.
        """
        raise NotImplementedError("Implement on subclass")

    def latency_processor(self, trace_latencies: np.ndarray) -> float:
        """
        Computes a single latency value (seconds) given an array of
        time deltas representing (arrival time - window start time)
        for a channel's traces.
        """
        raise NotImplementedError("Implement on subclass")

    def travel_time(self, distance: np.float32) -> float:
        """
        Computes time to detection given a channel's latency value
        and epicentral distance.

        Parameters
        ----------
        latency
            Chanenl latency in seconds.
        distance
            Epicentral distance in degrees.
        """
        raise NotImplementedError("Implement on subclass")

    def station_selection(
        self,
        station_metrics: Dict[str, Dict[str, float]],
        coordinate: Coordinate,
        additional_kwargs: Dict[str, Any]
    ) -> Tuple[List[Station], float, float]:
        """
        Given a dictionary of station metrics (),
        calculate a magnitude and time-to-detection value representative
        of a coordinate.

        Parameters
        ----------
        station_metrics
            Dictionary of format
            {"station_name": {'magnitude': value, 'ttd': value}}
            containing the magnitude and time-to-detection values for
            each station.

        Returns
        -------
        A tuple containing:
            - The list of N stations used in calculating the values
            - the magnitude value for the coordinate
            - the ttd value for the coordinate
        """
        raise NotImplementedError("Implement on subclass")

    def update(self, *args, **kwargs):
        """
        Where the "magic" happens. Needs cleaning up.

        Important to remember that:
            - Map subclasses need to implement:
                - amplitude processor: how the trace amplitudes for
                  each station are turned into a single value.
                  (e.g. average peak-to-peak).
                - magnitude processor: how the amplitude is turned
                  into a magnitude for a coordinate.
                - latency processor: how the latency for traces is turned
                  into a single value (e.g. max, average).
                - time-to-detection processor: how the latency value
                  is turned into time-to-detection for a coordinate.

        View mla.py and mb.py for examples of these.

        Also know:
            - Map subclasses need to specify what channel properties
              they are intersted in. When traces are fetched from
              stations, only the matching properties are considered
              (e.g. some magnitudes we only want vertical amplitude).
            - Map subclasses need to specify the trace metrics they
              require. This specifies what processing has been performed
              on the trace values requested. If your map subclass requires
              processing that doesn't already exist, it needs to
              be implemented in station.py.
        """
        # First portion - compute amplitude STA for each channel on each station,
        # and store the best channel STA for each station.
        best_station_amplitude = {}  # Holds the best channel STA for each station
        all_station_amplitudes = {}  # for IQR check
        best_station_latency = {}  # We store the latency of the best channel.

        # For storing explanatory metadata about the values calculated for each grid point.
        self.latency_explanation = defaultdict(list)
        self.magnitude_explanation = defaultdict(list)

        for station in self.stations:
            all_station_amplitudes[station] = {}
            best_station_amplitude[station] = None
            best_station_latency[station] = None
            channel_metrics = defaultdict(dict)

            # For each station's channels, we get the amplitudes of the traces
            # and calculate a long term average.
            best = sys.maxsize
            best_channel = None
            for code, values in station.get_values(
                self.channel_amplitude_metric, self.channel_properties
            ):
                # Handle channels that reported no data.
                if np.all(np.isnan(values)):
                    # We record some metadata in channel_status for display on the map.
                    self.channel_status[station][code]["status"] = Channel.NO_DATA
                    self.channel_status[station][code]["latency"] = None
                    self.channel_status[station][code]["amplitude"] = None
                    self.channel_status[station][code][
                        "message"
                    ] = f"Last time data was seen for this station: {station.get_last_received(code)}"
                    continue

                sample_rate = station.get_sample_rate(code)
                if sample_rate is None:
                    raise ValueError("Couldn't get sample rate for {code}")
                try:
                    amplitude = self.amplitude_processor(values, sample_rate)
                    if amplitude is not None and amplitude < best:
                        best = amplitude
                        best_channel = code
                    channel_metrics[code]["amplitude"] = amplitude
                except Exception as e:
                    _logger.error(f"Amplitude processor failed: {repr(e)}")

            # If no channels had data, then skip the station
            if best_channel is None:
                continue

            # Get latency for channels
            for code, values in station.get_values(
                self.channel_latency_metric, self.channel_properties
            ):
                if np.all(np.isnan(values)):
                    # We already record the no data status above in LTA calculation.
                    continue
                latency = self.latency_processor(values)
                channel_metrics[code]["latency"] = latency


            # Store the values for the best channel on this station
            best_station_amplitude[station] = channel_metrics[best_channel]["amplitude"]
            best_station_latency[station] = channel_metrics[best_channel]["latency"]

            # Store some more metadata about the channels
            self.channel_status[station][best_channel]["status"] = Channel.BEST
            self.channel_status[station][best_channel][
                "latency"
            ] = best_station_latency[station]
            self.channel_status[station][best_channel][
                "amplitude"
            ] = best_station_amplitude[station]
            self.channel_status[station][best_channel]["message"] = None

            for k, v in channel_metrics.items():
                latency = v.get("latency")
                amplitude = v.get("amplitude")
                if latency is None or amplitude is None:
                    continue
                all_station_amplitudes[station][k] = amplitude
                if k != best_channel:
                    self.channel_status[station][k]["status"] = Channel.GOOD
                    self.channel_status[station][k]["latency"] = latency
                    self.channel_status[station][k]["amplitude"] = amplitude
                    self.channel_status[station][k]["message"] = None

        # We check which stations are outside the interquartile range considering
        # the values of all stations in the map. This gets added to the station's channel_status
        # for display on the map.
        outlier_stations = self._check_outliers(all_station_amplitudes)

        # Now that we have the short term average and latency for each station, we can
        # compute the magnitude threshold and detection times for each coordinate.
        for i, c in self:
            station_metrics = {}
            for station in self.stations:
                attenuation = self.attenuations[station][c]
                # In some cases (e.g. mb distance needs to be between 5 and 105)
                # the station isn't applicable to the point. This is represented
                # by the attenuation being None.
                if attenuation is None:
                    continue
                amplitude = best_station_amplitude[station]
                # If the amplitude couldn't be calculated or the amplitude is outside
                # the IQR * 1.5 range, then we don't use the station.
                if amplitude is None or outlier_stations.get(station, False):
                    continue
                try:
                    noise_mag = log10(amplitude * self.snr) + attenuation
                except Exception:
                    _logger.exception(
                        f"{self.name}: Mag calculation failed for {station.name} "
                        f"(lta: {log10(amplitude * self.snr)}, correction: {attenuation}"
                    )
                    continue

                latency = best_station_latency[station]
                if latency is None:
                    continue
                travel_time = self.travel_times[station][c]
                ttd = latency + travel_time

                station_metrics[station] = {"magnitude": noise_mag, "ttd": ttd}

            # If we allow less than n_stations (n_stations_hard_limit = False) then we can continue
            # with less than n stations. Otherwise, if the hard limit is in place, or there is
            # no data at all, then we show nothing
            if (
                len(station_metrics) < self.n_stations and self.n_stations_hard_limit
            ) or len(station_metrics) == 0:
                # _logger.warning(f"Not enough data for point ({lat}, {lon})")
                self.magnitude_explanation[c.key].append([])
                self.latency_explanation[c.key].append([])
                self.magnitude_grid[i] = np.nan
                self.latency_grid[i] = np.nan
                continue

            (
                selected_stations,
                selected_magnitude,
                selected_ttd,
            ) = self.station_selection(station_metrics, c, self.station_selection_kwargs)
            for station in selected_stations:
                self.magnitude_explanation[c.key].append(
                    [station.name, self.attenuations[station][c]]
                )
                self.latency_explanation[c.key].append(
                    [station.name, self.travel_times[station][c]]
                )
            self.magnitude_grid[i] = selected_magnitude
            self.latency_grid[i] = selected_ttd

    def _check_outliers(self, all_station_ltas: dict) -> Dict[Station, bool]:
        """
        Check for station sensitivity outliers using IQR.

        We don't totally exclude a station if it is an outlier, as
        although we can be pretty sure certain readings are bullshit,
        any automated exclusion system has the risk of hiding genuine
        noise events or channel malfunctions.

        Instead we provide feedback to the user as a warning, and leave
        it them to identify if a particular station is putting out
        strange values consistently. Another option is setting a flat
        threshold of "impossible" values - this would require some
        discussion.

        Parameters
        ----------
        all_station_stas
            A dictionary of [station][code] = sta.
        """
        outlier_stations = {}
        all_ltas = []
        for station, channels in all_station_ltas.items():
            for code, lta in channels.items():
                if lta is not None:
                    all_ltas.append(lta)
        if all_ltas:
            q1 = np.quantile(all_ltas, 0.25)
            q3 = np.quantile(all_ltas, 0.75)
            lower_thresh = q1 - (q3 - q1) * 1.5
            upper_thresh = q3 + (q3 - q1) * 1.5
            for station, channel_ltas in all_station_ltas.items():
                for code, lta in channel_ltas.items():
                    if lta is None:
                        continue
                    if lta < lower_thresh:
                        self.channel_status[station][code]["status"] = Channel.WARNING
                        self.channel_status[station][code][
                            "message"
                        ] = "Amplitude is below LTA threshold"
                        outlier_stations[station] = True
                    elif lta > upper_thresh:
                        self.channel_status[station][code]["status"] = Channel.WARNING
                        self.channel_status[station][code][
                            "message"
                        ] = "Amplitude is above LTA threshold"
                        outlier_stations[station] = True

        return outlier_stations

    def _register_metric_with_stations(self):
        for station in self.stations:
            station.register_metric(
                self.channel_amplitude_metric, self.channel_properties
            )
            station.register_metric(
                self.channel_latency_metric, self.channel_properties
            )

    def _prepare_contours(self, scale, interval):
        contours = np.insert(
            np.append(
                np.arange(
                    scale[0],
                    scale[1] + interval,
                    interval,
                ),
                sys.maxsize,
            ),
            0,
            -sys.maxsize,
        )
        return contours

    def _create_contour_geojson(self, which="magnitude"):
        def _split_level(s):
            # horrible hack for dealing with '-' delimiter
            parts = s.split("-")
            if s.count("-") == 1:
                return float(parts[0]), float(parts[1])
            elif s.count("-") == 2 and s.startswith("-"):
                return float("-" + parts[1]), float(parts[2])
            elif s.count("-") == 2 and not s.startswith("-"):
                return float(parts[0]), float("-" + parts[2])
            elif s.count("-") == 3:
                return float("-" + parts[1]), float("-" + parts[3])
            else:
                raise ValueError(f"Couldn't derive min/max from {s}")

        if which == "magnitude":
            data = self.magnitude_grid
            contours = self.magnitude_contours
        else:
            data = self.latency_grid
            contours = self.latency_contours

        lons = self.lon_grid
        lats = self.lat_grid
        _, ax = pyplot.subplots()
        cont = ax.contourf(
            lons,
            lats,
            data,
            levels=contours,
        )
        cont_json = json.loads(geojsoncontour.contourf_to_geojson(cont))
        for feat in cont_json["features"]:
            level_min, level_max = _split_level(feat["properties"]["title"])
            if level_min == -sys.maxsize:
                level_min = "-infinity"
            elif level_max == sys.maxsize:
                level_max = "infinity"

            feat["properties"] = dict(level_min=level_min, level_max=level_max)
        pyplot.close()
        return cont_json

    def _create_station_geojson(self):
        records = {"type": "FeatureCollection", "features": []}
        for s in self.stations:
            records["features"].append(
                dict(
                    type="Feature",
                    geometry=dict(type="Point", coordinates=(s.coordinate.lon, s.coordinate.lat)),
                    properties=dict(
                        name=s.name, channel_statuses=self.channel_status[s]
                    ),
                )
            )
        return records

    def as_geojson(self):
        magnitude = self._create_contour_geojson(which="magnitude")
        latency = self._create_contour_geojson(which="latency")
        stations = self._create_station_geojson()
        latency_explanation = self.latency_explanation
        magnitude_explanation = self.magnitude_explanation
        data = dict(
            magnitude=magnitude,
            latency=latency,
            stations=stations,
            latency_explanation=latency_explanation,
            magnitude_explanation=magnitude_explanation,
        )
        return data

    def _precalculate_distances(self):
        distances = {station: {} for station in self.stations}
        for _, c in self:
            for station in self.stations:
                distances[station][c] = haversine(station.coordinate, c)
        return distances

    def _precalculate_attenuations(self):
        attenuation = {station: {} for station in self.stations}
        for _, c in self:
            for station in self.stations:
                attenuation[station][c] = self.magnitude_attenuation(
                    self.depth, station, c 
                )
        return attenuation

    def _precalculate_travel_times(self):
        travel_times = {station: {} for station in self.stations}
        for _, c in self:
            for station in self.stations:
                travel_times[station][c] = self.travel_time(
                    self.distances[station][c]
                )
        return travel_times

    @property
    def bounding_box(self):
        """A tuple of coordinates representing the west, south, east and north bounds."""
        return self.west, self.south, self.east, self.north

    @property
    def stream_names(self):
        if not hasattr(self, "stations"):
            return []
        else:
            names = []
            for station in self.stations:
                for channel in station:
                    names.append(channel.code)
            return names

    @cached_property
    def view_bounds(self):
        lon_buf = (self.east - self.west) * 0.07
        lat_buf = (self.north - self.south) * 0.07
        view_west = self.west - lon_buf
        view_east = self.east + lon_buf
        view_south = self.south - lat_buf
        view_north = self.north + lat_buf

        # If view bounds goes over limit, reduce visual buffer
        # This relies on checking if the grid bounds sans buffer
        # are already within limits
        if view_east - view_west > 360:
            over = (view_east - view_west) - 360
            view_east -= over / 2
            view_west += over / 2
        if view_north - view_south > 180:
            over = (view_north - view_south) - 180
            view_north -= over / 2
            view_south += over / 2
        return tuple(np.around((view_west, view_south, view_east, view_north), 2))

    @property
    def parameters(self):
        return {
            "name": self.name,
            "bounds": {
                "west": self.west,
                "south": self.south,
                "east": self.east,
                "north": self.north,
            },
            "resolution": self.resolution,
            "n_stations": self.n_stations,
            "lta_period": LTA,
            "magnitude_units": self.magnitude_units,
            "latency_units": self.latency_units,
            "amplitude_units": self.amplitude_units,
            "snr": self.snr,
        }

    @property
    def metadata(self):
        return {"parameters": self.parameters}

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def _build_coord_grid(self):
        """
        Origin of our grid cells is the center
        """
        if self.east <= self.west:
            raise ValueError(
                "East bound must be greater than west bound "
                f"(current: {self.west}W >= {self.east}E)"
            )
        if self.east - self.west > 360.0:
            raise ValueError(
                "Longitude cannot span more than 360 degrees: "
                f"{self.west}W - {self.east}W = {self.west - self.east} degrees"
            )
        if self.east < -360.0:
            raise ValueError(
                "East bound cannot be less than -360 degrees: "
                f"(current: {self.east}E)"
            )
        if self.west > 360.0:
            raise ValueError(
                "West bound cannot be greater than 360 degrees: "
                f"(current: {self.west}W)"
            )
        lons = np.arange(
            self.west, self.east + self.resolution, self.resolution
        ).astype(np.float32)
        lons += self.resolution / 2
        if lons[-1] != self.east:
            _logger.warning(
                f"East bound truncated to {lons[-1]} as we can't fit an even number "
                "of cells at the desired resoltuion."
            )

        if self.north <= self.south:
            raise ValueError(
                "North bound must be greater than south bound "
                f"(current: {self.south}S >= {self.north}N"
            )
        if self.north - self.south > 180.0:
            raise ValueError(
                "Latitude cannot span more than 180 degrees: "
                f"{self.north}N - {self.south}S = {self.north - self.south} degrees"
            )
        lats = np.arange(
            self.south, self.north + self.resolution, self.resolution
        ).astype(np.float32)
        lats += self.resolution / 2
        if lats[-1] != self.north:
            _logger.warning(
                f"North bound truncated to {lats[-1]} as we can't fit an even number "
                "of cells at the desired resolution."
            )

        return np.meshgrid(lats, lons, indexing="ij")

    def __iter__(self) -> Iterator[Tuple[Tuple[int,...], Coordinate]]:
        for i, lat in np.ndenumerate(self.lat_grid):
            lon = self.lon_grid[i]
            if np.isnan(lat) or np.isnan(lon):
                continue
            else:
                yield i, Coordinate(lat, lon)
