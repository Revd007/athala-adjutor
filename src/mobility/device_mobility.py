import asyncio
import websockets
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DEVICE_CONFIGS

class DeviceMobility:
    def __init__(self):
        self.devices = DEVICE_CONFIGS
        self.db = DatabaseManager()
        logger.info("DeviceMobility initialized with PostgreSQL support")

    async def connect_device(self, device_id):
        try:
            device = self.devices.get(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")
            async with websockets.connect(f"ws://{device['host']}:{device['port']}") as websocket:
                await websocket.send(f"Connected to {device_id}")
                response = await websocket.recv()
                self.db.store_update_log("mobility", f"connected_{device_id}")
                logger.info(f"Connected to {device_id}: {response}")
                return response
        except Exception as e:
            logger.error(f"Error connecting to device {device_id}: {str(e)}")
            return None

    def start_mobility(self):
        try:
            for device_id in self.devices:
                asyncio.run(self.connect_device(device_id))
            logger.info("Device mobility started")
        except Exception as e:
            logger.error(f"Error starting device mobility: {str(e)}")