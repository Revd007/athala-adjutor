import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import os
import asyncio
from logger import logger

class VoiceHandler:
    def __init__(self, output_dir="./data/audio"):
        self.recognizer = sr.Recognizer()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def speech_to_text(self, audio_file=None):
        """Convert speech to text using microphone or audio file."""
        try:
            if audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
            else:
                with sr.Microphone() as source:
                    logger.info("Listening for voice input...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source, timeout=5)

            text = self.recognizer.recognize_google(audio)
            logger.info(f"Speech to text: {text}")
            return text
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            raise
        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            raise

    def text_to_speech(self, text, output_file=None):
        """Convert text to speech and save to file."""
        try:
            if not output_file:
                output_file = f"{self.output_dir}/output_{int(time.time())}.mp3"
            
            tts = gTTS(text=text, lang='en')
            tts.save(output_file)
            
            # Convert to wav for compatibility
            audio = AudioSegment.from_mp3(output_file)
            wav_file = output_file.replace(".mp3", ".wav")
            audio.export(wav_file, format="wav")
            
            logger.info(f"Text to speech saved: {wav_file}")
            return wav_file
        except Exception as e:
            logger.error(f"Text to speech error: {e}")
            raise

    async def stream_voice(self, text, websocket):
        """Stream voice response via WebSocket."""
        try:
            output_file = self.text_to_speech(text)
            audio = AudioSegment.from_file(output_file)
            chunk_size = 1024
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                await websocket.send_bytes(chunk.raw_data)
                await asyncio.sleep(0.01)
            logger.info("Voice streamed via WebSocket")
        except Exception as e:
            logger.error(f"Voice streaming error: {e}")
            raise

if __name__ == "__main__":
    import time
    handler = VoiceHandler()
    # Test speech-to-text
    text = handler.speech_to_text()
    print(f"Recognized: {text}")
    # Test text-to-speech
    handler.text_to_speech("Yo bro, this is Athala Adjutor!", "test_output.mp3")