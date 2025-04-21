import asyncio
import websockets
import pyaudio
import cv2
import mss
from logger import logger
from src.utils.database_manager import DatabaseManager

class StreamHandler:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.pa = pyaudio.PyAudio()
        self.db = DatabaseManager()
        logger.info("StreamHandler initialized with PostgreSQL support")

    async def stream_audio(self, websocket):
        try:
            stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
            while True:
                data = stream.read(1024)
                await websocket.send(data)
                self.db.store_update_log("streaming", "audio_streamed")
        except Exception as e:
            logger.error(f"Error streaming audio: {str(e)}")

    async def stream_video(self, websocket):
        try:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                await websocket.send(buffer.tobytes())
                self.db.store_update_log("streaming", "video_streamed")
            cap.release()
        except Exception as e:
            logger.error(f"Error streaming video: {str(e)}")

    async def stream_screen(self, websocket):
        try:
            with mss.mss() as sct:
                while True:
                    screenshot = sct.shot()
                    img = cv2.imread(screenshot.rgb)
                    img = cv2.resize(img, (1280, 720))
                    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    await websocket.send(buffer.tobytes())
                    self.db.store_update_log("streaming", "screen_streamed")
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error streaming screen: {str(e)}")

    async def stream_text(self, text):
        try:
            async with websockets.connect(f"ws://{self.host}:{self.port}") as websocket:
                await websocket.send(text)
                self.db.store_update_log("streaming", "text_streamed")
                logger.info(f"Streamed text: {text}")
        except Exception as e:
            logger.error(f"Error streaming text: {str(e)}")

    async def handler(self, websocket, path):
        try:
            async for message in websocket:
                if message == "audio":
                    await self.stream_audio(websocket)
                elif message == "video":
                    await self.stream_video(websocket)
                elif message == "screen":
                    await self.stream_screen(websocket)
        except Exception as e:
            logger.error(f"Error in stream handler: {str(e)}")

    def start_server(self):
        try:
            server = websockets.serve(self.handler, self.host, self.port)
            asyncio.get_event_loop().run_until_complete(server)
            asyncio.get_event_loop().run_forever()
            logger.info("Streaming server started")
        except Exception as e:
            logger.error(f"Error starting streaming server: {str(e)}")