import json
import math
import logging
import os, json, shutil
from glob import glob
import concurrent.futures

from src.seg2map import exceptions
from src.seg2map import common

from typing import List, Tuple
import platform
import tqdm
import tqdm.auto
import zipfile
from area import area
import numpy as np
import geopandas as gpd
import asyncio
import aiohttp
import tqdm.asyncio
import nest_asyncio
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
import ee
from osgeo import gdal
import asyncio
from typing import List
import aiohttp

logger = logging.getLogger(__name__)


import asyncio

class DownloadManager:
    def __init__(self, session_manager, max_concurrent_downloads=10):
        self.session_manager = session_manager
        self.max_concurrent_downloads = max_concurrent_downloads
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.batches = []

    def add_batch(self, batch):
        self.batches.append(batch)

    async def start_downloads(self):
        for batch in self.batches:
            await self.download_batch(batch)

    async def download_batch(self, batch):
        tasks = []
        async with self.semaphore:
            async with self.session_manager.get_session() as session:
                for item in batch:
                    task = asyncio.create_task(self.download_item(session, item))
                    tasks.append(task)
                completed, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                while len(pending) > 0:
                    new_tasks = []
                    async with self.session_manager.get_session() as session:
                        for task in pending:
                            if task.cancelled():
                                continue
                            try:
                                await task
                            except Exception as e:
                                print(f"Error: {e}")
                                session.close()
                                async with self.session_manager.get_session() as session:
                                    new_task = asyncio.create_task(self.download_item(session, task.item))
                                    new_tasks.append(new_task)
                                continue
                            new_tasks.append(task)
                        completed, pending = await asyncio.wait(new_tasks, return_when=asyncio.FIRST_EXCEPTION)
            await self.cleanup(batch, completed)

    async def download_item(self, session, item):
        # download item using session
        pass

    async def cleanup(self, batch, completed):
        # cleanup logic
        pass



class DownloadManager:
    def __init__(self, max_concurrent_sessions=10, session_timeout=60):
        self.sessions = []
        self.session_timeout = session_timeout
        self.max_concurrent_sessions = max_concurrent_sessions
        self.tasks = []
        self.completed_tasks = set()
        # GEE allows for 20 concurrent requests at once
        self.limit = asyncio.Semaphore(20)

    def get_limit(self):
        return self.limit


    async def download(self, url, session):
        # Implement your download logic here
        pass

    async def check_sessions(self):
        # Check if any sessions are about to timeout
        now = asyncio.get_running_loop().time()
        self.sessions = [s for s in self.sessions if s.expiration_time > now]
        
        # Create new sessions if necessary
        while len(self.sessions) < self.max_concurrent_sessions:
            session = aiohttp.ClientSession()
            self.sessions.append(session)

    async def run(self, urls: List[str]):
        # Start download tasks
        for url in urls:
            session = self.sessions[0]
            task = asyncio.create_task(self.download(url, session))
            self.tasks.append(task)
            self.sessions = self.sessions[1:] + [session]

        # Monitor sessions and tasks
        while len(self.tasks) > 0:
            await self.check_sessions()
            done, pending = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                self.tasks.remove(task)
                self.completed_tasks.add(task)
            for task in pending:
                self.tasks.remove(task)
                session = next((s for s in self.sessions if not s.closed), None)
                if session is None:
                    raise Exception("All sessions closed")
                self.sessions.remove(session)
                self.sessions.append(session)
                self.tasks.append(task)

        # Close all sessions
        for session in self.sessions:
            await session.close()
