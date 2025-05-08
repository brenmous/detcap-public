from typing import Tuple, Sequence, Union, Optional, List, Dict, Any
import logging
import statistics

import numpy as np

from detcap.utils import (
    get_peak_amplitudes_period_proportional as pa_proportional,
    load_table,
    round_for_table,
    seiscomp_q,
    Coordinate
)
from detcap.inventory.station import Station, Channel
from detcap.map import Map

_logger = logging.getLogger(__name__)


class MBMap(Map):
    def __init__(
        self,
        name: str,
        west: np.float32,
        south: np.float32,
        east: np.float32,
        north: np.float32,
        channel_amplitude_metric = "simulate_wwssn_sp",
        channel_latency_metric = "latency",
        resolution: int = 1,
        n_stations: int = 4,
        snr: float = 3.0,
        depth: float = 15.0,
        stations: Sequence[Station] = [],
        magnitude_scale: Tuple[float, float] = (0.0, 6.0),
        magnitude_interval: float = 0.2,
        latency_scale: Tuple[float, float] = (
            0.0,
            600,
        ),
        latency_interval: int = 30,
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
            snr=snr,
            depth=depth,
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
        self.n_stations_hard_limit = True
        self.amplitude_units = "\u03BCm"  # micrometers
        self.magnitude_units = "mb"
        self.latency_units = "seconds"
        self.tt_table = load_table("ak135_travel_times_P")
        self.attenuations = self._precalculate_attenuations()
        self.travel_times = self._precalculate_travel_times()

    def amplitude_processor(self, trace_data: np.ndarray, sample_rate: float) -> Optional[float]:
        # Convert to micrometres
        values_as_um = trace_data * 1e6
        peak_amplitudes, periods = pa_proportional(values_as_um, sample_rate)
        # T < 3s for mb according to IASPEI. Filter out anything longer than that.
        a_on_t = [a / t for a, t in zip(peak_amplitudes, periods) if t < 3.0]
        if not a_on_t:
            return
        return np.mean(a_on_t)

    def magnitude_attenuation(
        self,
        depth: float,
        station: Station,
        point: Coordinate,
    ) -> Optional[np.float32]:
        distance = self.distances[station][point]
        # Epicentral distance limited between 20 and 100 degrees according to IASPEI.
        # Seiscomp goes from 5 to 105, but this can be changed in Seiscomp config,
        # so we should parametrize this and set it to the same values.
        if not 5 <= distance <= 105:
            return None
        # There's a few choices of q_factor:
        # - the original Richter Gutenberg function, intended for use with amplitudes
        #   in micrometres. Mostly superceded.
        # - a version of Richter Gutenberg intended for use with nanometres, as part
        #   of "The new IASPEI standards", accompanied by a -3 correction term.
        #   Used by NEIC.
        # - Veith Clawson. I'm assuming intended for use with nanometres given it's
        #   lower values compared to factors intended for micrometres,
        #   but I was never able to confirm. This is the q_factors_P table in the
        #   package data.
        # - Seiscomp's Q factor, credited to Saul & Bormann 2007. This is the one we
        #   use as we're trying to replicate Seiscomp whenever we can. It's intended
        #   for use with micrometres, as Seiscomp converts amplitudes to um before
        #   applying it. On their mb info page, their formula has the -3 correction term,
        #   but this doesn't seem applied, see comment below.
        q_factor = seiscomp_q(distance, depth)
        # I'm sceptical of the -3 correction and when it is applied.
        # IASPEI and Seiscomp share the same mb formula. Although it's not explained,
        # I believe it's applied when the amplitude is taken as nanometres and used
        # with appropriately scaled Q factor. Seiscomp doesn't seem to apply it in
        # in their bmagnz function. If we do apply it when using micrometres, the
        # values are far too low (1 - 2mb). More reasonable values are achieved
        # when processing in nanometres and applying this correction (4 - 6mb).
        # However this is a good 0.2 - 4mu above processing in micrometres and without
        # correction, and because the results are fairly consistent between using
        # Veith-Clawson and this method, and it because it replicates what the Seiscomp
        # code is actually doing, it's what I prefer.
        return q_factor

    def latency_processor(self, trace_latencies: np.ndarray) -> float:
        return np.mean(trace_latencies)  # type: ignore

    def travel_time(self, distance: np.float32) -> float:
        return self.tt_table[str(self.depth)][round_for_table(distance)]

    def station_selection(
        self,
        station_metrics: Dict[Station, Dict[str, float]],
        coordinate: Coordinate,
        additional_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Station], float, float]:

        selected = [i for i in sorted(station_metrics.items(),
                                      key=lambda i: i[1]['ttd'])][:self.n_stations]
        return (
            [s[0] for s in selected],
            statistics.median(s[1]['magnitude'] for s in selected),
            max(s[1]['ttd'] for s in selected)
        )
