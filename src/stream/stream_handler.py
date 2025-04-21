import cv2
import asyncio
from fastapi import WebSocket
from logger import logger

class StreamHandler:
    def __init__(self):
        self.cap = None

    async def stream_video(self, websocket: WebSocket, source=0):
        """Stream video via WebSocket."""
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError("Cannot open video source")

            await websocket.accept()
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_bytes(buffer.tobytes())
                await asyncio.sleep(0.03)  # ~30 fps

            await websocket.close()
        except Exception as e:
            logger.error(f"Video streaming error: {e}")
            raise
        finally:
            if self.cap:
                self.cap.release()

if __name__ == "__main__":
    # Test requires running with FastAPI
    pass