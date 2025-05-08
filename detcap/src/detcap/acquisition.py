"""
Non-blocking seedlink clients.
"""
from typing import List, Dict, Optional, Union
from multiprocessing import Process, Queue
from datetime import datetime
from contextlib import nullcontext
import logging
import asyncio

from obspy.clients.seedlink.client.seedlinkconnection import SeedLinkConnection
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.clients.seedlink.slpacket import SLPacket
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace
from obspy.core.stream import Stream
import numpy as np
import xmltodict
import aioprocessing

from detcap.inventory.station import Station


_logger = logging.getLogger(__name__)


class ConcurrentSeedlinkClient:
    """
    A Seedlink client that can run in one or more background
    processes.

    Internally it starts multiple EasySeedLinkClients and splits
    subscriptions between them.

    Clients pass recieved traces to a queue, which is emptied
    out by threaded workers that transfer them to the corresponding
    Channel's SyncStreamBuffer.
    """

    def __init__(self, url: str, clients: int = 1):
        self.url = url
        self.channels = {}
        self.clients = list(range(clients))
        self.queues = {
            client: aioprocessing.AioQueue(maxsize=32000) for client in self.clients
        }
        self.processes: Dict[int, aioprocessing.AioProcess] = {}

    def _validate_trace(self, trace: Trace):
        # TODO once cloudwatch is setup, add these to a stream that
        # collects "weird" errors that need to be investigated.

        # TODO
        # Some stations have ridiculous clock errors that report packets
        # from the future. If the error is greater than a second then
        # we dump it.
        if trace.stats.endtime - trace.stats.arrivaltime > 1:
            _logger.warning(
                "%s reported trace from the future, end time > arrival time by "
                "%s seconds. This packet is being ignored.",
                trace.get_id(),
                abs(trace.stats.endtime - trace.stats.arrivaltime),
            )
            return False

        # TODO
        # This is some sort of weird mismatch between what a station
        # is called in the inventory and what seedlink calls it.
        # Need to work it out.
        if trace.get_id() not in self.channels:
            _logger.warning(
                "Received trace %s but this channel was never subscribed to (intentionally)",
                trace.get_id(),
            )
            return False

        return True

    async def transfer(self, client: int):
        while True:
            trace = await self.queues[client].coro_get()
            if trace is None:
                return
            if self._validate_trace(trace):
                await self.channels[trace.get_id()].store_trace(trace)

    def subscribe(self, stations: List[Station]):
        for s in stations:
            for channel in s.all_registered_channels:
                net, sta, loc, cha = channel.code.split(".")
                loc = "" if loc == "--" else loc
                self.channels[f"{net}.{sta}.{loc}.{cha}"] = channel
        _logger.info("Subscribed to stations")

    def available_stations(self):
        """
        Start a client and do an INFO request to get the stations
        available from Seedlink.
        """
        slc = EasySeedLinkClient(self.url, autoconnect=False)
        slc.connect()
        station_dict = xmltodict.parse(slc.get_info(level="STATIONS"))
        slc.close()
        stations = []
        for station in station_dict["seedlink"]["station"]:
            net = station["@network"]
            sta = station["@name"]
            stations.append(f"{net}.{sta}")
        return stations

    async def run(self):
        try:
            sub_lists = [
                list(subarray)
                for subarray in np.array_split(
                    list(self.channels.keys()), len(self.clients)
                )
            ]
            for client in self.clients:
                self.processes[client] = aioprocessing.AioProcess(
                    target=make_and_start_client,
                    args=(self.url, sub_lists[client], self.queues[client]),
                )
                self.processes[client].start()
                await self.transfer(client)
        finally:
            for process in self.processes.values():
                await process.coro_join()


def make_and_start_client(url: str, subscription_list: List[str], queue: Queue):
    try:
        return BackgroundSeedlinkClient(url, subscription_list, queue).run()
    finally:
        queue.put(None)


class BackgroundSeedlinkClient(EasySeedLinkClient):
    def __init__(self, url: str, subscription_list: List[str], queue: Queue):
        super().__init__(server_url=url, autoconnect=False)
        self.queue = queue
        self.connect()
        for code in subscription_list:
            net, sta, loc, cha = code.split(".")
            loc = "" if loc == "--" else loc
            self.select_stream(net, sta, f"{loc}{cha}")

    def on_data(self, trace: Trace):
        """
        Record arrival time and place in queue.
        """
        trace.stats.arrivaltime = UTCDateTime()
        self.queue.put(trace)

    def on_terminate(self):
        _logger.info("Seedlink server terminated connection")

    def on_seedlink_error(self):
        _logger.error("Seedlink server sent an error packet")
