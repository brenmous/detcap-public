"""
Functionality for managing streams/stations.
"""
from typing import List, Sequence, Dict, Any, Callable, Set, Iterable, Optional
from functools import lru_cache
import logging
import copy

import numpy as np
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.inventory.inventory import Inventory
from obspy.core.inventory.response import Response

from detcap.utils import SyncStreamBuffer, Coordinate
from detcap.utils import PathType
from detcap.inventory.paz import WOODANDERSON, WWSSN_SP
from detcap.inventory import parse_inventory

_logger = logging.getLogger(__name__)


class Channel:
    BAD_DATA = "Bad data"
    """Stream status for when the received data can't be used in calulations."""
    NO_DATA = "No data"
    """Stream status for when no data was received during the update cycle."""
    WARNING = "Warning"
    """Stream status for when a station reports questionable values."""
    GOOD = "Good"
    """Stream status for when computable data was recieved during the update cycle."""
    BEST = "Best"
    """Best performing channel for the used metric."""

    BAD_STATUSES = [BAD_DATA, NO_DATA]

    def __init__(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        response: Response,
        sample_rate: float,
    ):
        """
        Represents a channel of a station.

        Parameters
        ----------
        network
            The code of the network that this channel belongs to.
        station
            The code of the station that this channel belongs to.
        location
            The code of the sensor location that this channel belongs to.
        channel
            The channel code.
        response
            The channel's response as an obspy Response object.
        sample_rate
            Sample rate of the channel.

        Attributes
        ----------
        horizontal
            True if the channel is horizontal, False if vertical.
        trace_buffer
            A SyncStreamBuffer containing traces for this channel.
        registered_metrics
            Trace processing functions that should be applied to this
            channel's traces on update.
        metric_results
            A mapping of processing function to result data.
        active
            True if this channel has been requested at least once,
            meaning traces for it will be received from seedlink.
            False otherwise.
        data_last_received
            The last time a trace for this channel was received.
        """
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.horizontal = channel[-1] in ("E", "N", "1", "2")
        self.response = response
        self.sample_rate = sample_rate
        self.trace_buffer = SyncStreamBuffer()
        self.registered_metrics: List[Callable] = []
        self.metric_results: Dict[Callable, np.ndarray] = {}
        self.active = False
        self.data_last_received = None

    def check_orientation(self, orientation: str) -> bool:
        """
        Returns whether the channel has the provided orientation.

        Parameters
        ----------
        orientation
            One of 'horizontal', 'vertical', or 'both'

        Returns
        -------
        bool
            Whether the channel has the provided orientation.
        """
        if orientation not in ("horizontal", "vertical", "both"):
            raise ValueError(
                f"'{orientation}' is invalid. Valid orientations are 'vertical', 'horizonal' "
                "and 'both'."
            )
        if orientation == "vertical":
            return not self.horizontal
        elif orientation == "horizontal":
            return self.horizontal
        elif orientation == "both":
            return True

    @property
    def code(self) -> str:
        """SEED ID of the channel."""
        return f"{self.network}.{self.station}.{self.location}.{self.channel}"

    @property
    def name(self) -> str:
        """Channel code."""
        return self.channel

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        return f"{self.name}: {self.response}"

    def register_metric(
        self,
        metric_name: Optional[str],
    ):
        """
        Registers a metric with this channel. When an update occurs,
        registered metrics are calculated on the channel and the results
        used by the maps that registered the metric.

        Parameters
        ----------
        metric_name
            Name of the metric function to register.
        """
        if metric_name is None:
            raise ValueError("No metric name was provided")

        fn = getattr(self, metric_name)

        if fn not in self.registered_metrics:
            self.registered_metrics.append(fn)

    async def handle_traces(self, starttime: UTCDateTime, endtime: UTCDateTime):
        """
        Gets the time-windowed subset of traces and calculates all
        registered trace metrics on them. The results are stored in
        `metric_results`.
        """
        traces = await self.trace_buffer.get(starttime, endtime)
        if traces:
            self._data_last_received = traces[-1].stats.arrivaltime  # type: ignore
        for tm in self.registered_metrics:
            try:
                # Obspy will modify data in-place so it's important to
                # use copies of the traces.
                self.metric_results[tm] = tm(traces.copy())
            except Exception:
                _logger.exception(f"Trace metric {tm.__name__} failed for {self.code}")
                self.metric_results[tm] = np.full(0, np.nan, dtype=np.float64)

    async def store_trace(self, trace: Trace):
        """
        Store a received trace in the Channel's buffer.
        """
        await self.trace_buffer.append(trace)

    def get_values(self, metric_name: Optional[str]) -> np.ndarray:
        """
        Get the computed values for a particular metric.
        """
        try:
            if metric_name is None:
                raise ValueError("Metric name cannot be None")
            return self.metric_results[getattr(self, metric_name)]
        except Exception:
            _logger.exception(f"No metric {metric_name} exists with {self.code}")
            raise

    def latency(self, traces: Stream, **kwargs):
        """
        Calculate the latency of the provided trace. Note that metric
        functions need to be variadic so they can be called generically.

        Parameters
        ----------
        traces
            The traces to process as an obspy Stream object.

        Raises
        ------
        AttributeError
            If the latency metric has not been registered with this stream.
        """
        if not traces:
            return np.full(0, np.nan, dtype=np.float64)

        res = []
        for t in traces:
            res.append(abs(t.stats.arrivaltime - t.stats.starttime))

        return np.array(res, dtype=np.float64)

    def simulate_wwssn_sp(self, traces: Stream):
        """
        Simulate a sequence of traces as though they were recorded
        on a WWSSN-SP seismometer. Used in calculating mb magnitudes.

        The simulated traces are stored in the metric's corresponding
        value store.

        Parameters
        ----------
        traces
            The traces to simulte as an obspy Stream object.
        """
        if not traces:
            return np.full(0, np.nan, dtype=np.float64)

        # Kind of silly. You either provide an inventory to "remove_response",
        # in which case obspy will search for it, or if it has a response
        # attached it will use that. You can't just provide a response.
        # To stop it from searching we attach the response before
        # doing calculations. The response has to be attached to each trace.
        for t in traces:
            t.stats.response = self.response
        traces.detrend()
        traces.filter("bandpass", freqmin=0.2, freqmax=7, corners=3, zerophase=True)
        # Water level is dropped to 60 from 600 based on
        # https://github.com/obspy/obspy/issues/1594#issuecomment-262232681 (basically stating
        # the 600 water level of simulate_seismometer is too conservative and 60 is more robust)
        traces.remove_response(
            inventory=None,
            output="VEL",
            water_level=60.0,
        )
        traces.simulate(
            paz_remove=None,
            paz_simulate=WWSSN_SP,
            water_level=60.0,
            simulate_sensitivity=False,
        )
        return np.concatenate([t.data for t in traces])

    def simulate_wa(self, traces: Stream):
        """
        Simulates a sequence of traces as though they were
        recorded on a Wood-Anderson seismometer. Used in calculating
        local magnitudes (e.g. mla).

        The simulated traces are stored in the metric's corresponding
        value store.

        Parameters
        ----------
        traces
            The traces to simulate as an Obspy Stream object.
        """
        if not traces:
            return np.full(0, np.nan, dtype=np.float64)

        for t in traces:
            t.stats.response = self.response
        traces.detrend()
        traces.filter("bandpass", freqmin=0.2, freqmax=7, corners=3, zerophase=True)
        traces.remove_response(inventory=None, output="VEL", water_level=60.0)
        traces.simulate(paz_remove=None, paz_simulate=WOODANDERSON, water_level=60.0)
        return np.concatenate([t.data for t in traces])


class Station:
    def __init__(
        self,
        network: str,
        station: str,
        latitude: np.float32,
        longitude: np.float32,
        channels: Iterable[Channel] = set(),
    ):
        self.network = network
        self.station = station
        self.coordinate = Coordinate(latitude, longitude)
        self.channels: Set[Channel] = set(channels)

    @property
    def name(self):
        return f"{self.network}.{self.station}"

    @property
    def code(self):
        return self.name

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        channels = [str(c) for c in self.channels]
        return f"{self.name}: {' | '.join(channels)}"

    def __iter__(self):
        for channel in self.channels:
            yield channel

    def get_channel(self, code: str) -> Channel:
        channel = next(
            (channel for channel in self.channels if channel.code == code), None
        )
        if channel is None:
            raise ValueError(f"No channel for {code} on station {self.name}")
        else:
            return channel

    @lru_cache
    def find_channels_with_properties(self, orientation="both"):
        return [channel for channel in self if channel.check_orientation(orientation)]

    @lru_cache
    def get_sample_rate(self, code: str) -> float:
        return self.get_channel(code).sample_rate

    def get_last_received(self, code):
        return self.get_channel(code).data_last_received

    @property
    def all_registered_channels(self) -> List[Channel]:
        return [channel for channel in self.channels if channel.active]

    def register_metric(
        self,
        metric_name: str,
        channel_properties: Dict[str, Any] = {},
    ):
        for channel in self.find_channels_with_properties(**channel_properties):
            channel.register_metric(metric_name)
            channel.active = True

    async def handle_traces(self, starttime, endtime):
        for channel in self.all_registered_channels:
            await channel.handle_traces(starttime, endtime)

    def get_values(self, metric_name: Optional[str], channel_properties: Dict[str, Any] = {}):
        result = []
        for channel in self.find_channels_with_properties(**channel_properties):
            result.append((channel.code, channel.get_values(metric_name)))
        return result

    def get_values_for_code(
        self, metric_name: str, code: str, channel_properties: Dict[str, Any] = {}
    ):
        for channel in self.find_channels_with_properties(**channel_properties):
            if channel.code == code:
                return channel.get_values(metric_name)
        return None


def _iter_inventory(inventory: Inventory):
    for network in inventory:
        for station in network:
            for channel in station:
                yield network, station, channel


class InventoryDataNotFound(Exception):
    """Raised when trying to extract data from inventory for a station that isn't present."""


def stations_from_list(
    station_names: Iterable[str],
    inventory_file: PathType,
    raise_on_unavailable: bool = True,
) -> Dict[str, Station]:
    """
    Creates Station objects from a list of station names and an inventory.

    Parameters
    ----------
    stream_ids
        A list of station names, of form "{NET}.{STA}".
    inventory
        An Obspy inventory to load metadata from.
    raise_on_unavailable
        If True, raise an InventoryDataNotFound exception if inventory
        metadata cannot be found for a station. Otherwise, log it as a
        warning.

    Returns
    -------
    List[Station]
        A list of Stations initialized with metadata.

    Raises
    ------
    InventoryDataNotFound
        If a station present in `station_names` doesn't have
        a corresponding inventory entry.
    """
    _logger.info("Loading stream metadata")
    inventory = parse_inventory(inventory_file)
    network_codes, station_codes = list(
        zip(*[name.split(".") for name in station_names])
    )
    stations = set()

    # IA stations have weird metadata (BH and SH channels running
    # off the same instrument) that messes up response removal.
    # We take a Trillium 120 response off our network
    # and use it as the response for IA stations.
    # Big thanks to Hugh for figuring out the problem.
    IA_REPLACEMENTS = {}
    try:
        ia_replacement_response = inventory.get_response(
            "AU.KMBL.00.BHZ", UTCDateTime.now()
        )
    except Exception:
        _logger.exception("No IA replacement response found")
    KIM_REPLACEMENTS = {}
    # It appears KIM03 and KIM05 did not have their metadata updated
    # the same as other related stations. 
    try:
        kim_replacement_response = inventory.get_response(
            "AU.CARL.00.HHZ", UTCDateTime.now()
        )
    except Exception:
        _logger.exception("No KIM replacement response found")

    for network in inventory:
        if network.code not in network_codes:
            _logger.warning(f"Network {network.code} not requested, skipping it")
            continue

        for station in network:
            if station.code not in station_codes:
                _logger.warning(f"Station {station.code} not requested, skipping it")
                continue
            our_station = Station(
                network.code, station.code, np.float32(station.latitude), np.float32(station.longitude)
            )
            stations.add(our_station)

            for channel in station:
                name = f"{network.code}.{station.code}.{channel.location_code}.{channel.code}"
                if network.code == "IA":
                    response = IA_REPLACEMENTS.get(channel.code)
                    if response is None:
                        try:
                            response = inventory.get_response(f"AU.KMBL.00.{channel.code}", UTCDateTime.now())
                            IA_REPLACEMENTS[channel.code] = response
                        except Exception:
                            _logger.warning(
                                f"Could not find substitute response for {name}, it will be excluded"
                            )
                            continue
                    _logger.warning(
                        f"Replacing {name} response with response from AU.KMBL.00.{channel.code}"
                    )
                    # We need to set the station's particular sensitivity (gain) on the
                    # substitute response.
                    copied_response = copy.deepcopy(response)
                    copied_response.instrument_sensitivity = (
                        channel.response.instrument_sensitivity
                    )
                elif network.code == "AU" and station.code in ("KIM03", "KIM05"):
                    response = KIM_REPLACEMENTS.get(channel.code)
                    if response is None:
                        try:
                            response = inventory.get_response(f"AU.CARL.00.{channel.code}", UTCDateTime.now())
                            KIM_REPLACEMENTS[channel.code] = response
                        except Exception:
                            _logger.warning(
                                f"Could not find substitute response for {name}, it will be excluded"
                            )
                            continue
                    _logger.warning(
                        f"Replacing {name} response with response from AU.CARL.00.{channel.code}"
                    )
                    # We need to set the station's particular sensitivity (gain) on the
                    # substitute response.
                    copied_response = copy.deepcopy(response)
                    copied_response.instrument_sensitivity = (
                        channel.response.instrument_sensitivity
                    )
                    
                else:
                    response = channel.response

                our_station.channels.add(
                    Channel(
                        network=network.code,
                        station=station.code,
                        location=channel.location_code,
                        channel=channel.code,
                        response=response,
                        sample_rate=channel.sample_rate,
                    )
                )
    missing = [
        name
        for name in station_names
        if name not in [station.name for station in stations]
    ]
    if missing:
        msg = (
            f"Could not find inventory metadata for requested stations: "
            f"{', '.join(missing)}"
        )
        if raise_on_unavailable:
            raise InventoryDataNotFound(msg)
        else:
            _logger.warning(msg)

    return {station.name: station for station in stations}
