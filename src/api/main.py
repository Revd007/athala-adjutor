from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from src.dialog.dialog import DialogModel
from src.codegen.codegen import CodeGenerator
from src.math.math_solver import MathSolver
from src.finance.market_analyzer import MarketAnalyzer
from src.sysadmin.sysadmin_manager import SysadminManager
from src.stream.stream_handler import StreamHandler
from src.stream.voice_handler import VoiceHandler
from src.network.network_manager import NetworkManager
from src.utils.captcha_solver import CaptchaSolver
from src.utils.web_crawler import WebCrawler
from src.ai.rag import RAG
from src.ai.enhanced_rag import EnhancedRAG
from src.mobility.device_mobility import DeviceMobility
from logger import logger

app = FastAPI(title="Athala Adjutor API")

class PromptRequest(BaseModel):
    prompt: str

class CodeRequest(BaseModel):
    prompt: str
    language: str

class MathRequest(BaseModel):
    problem: str

class TradingRequest(BaseModel):
    data: list[dict]

class CaptchaRequest(BaseModel):
    url: str

class CrawlRequest(BaseModel):
    component: str
    url: str

class VoiceRequest(BaseModel):
    text: str

@app.post("/api/dialog")
async def dialog(request: PromptRequest):
    try:
        dialog = DialogModel()
        response = dialog.generate_response(request.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Dialog error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/codegen")
async def codegen(request: CodeRequest):
    try:
        codegen = CodeGenerator()
        code = codegen.generate_code(request.prompt, request.language)
        return {"code": code}
    except Exception as e:
        logger.error(f"Codegen error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/math")
async def math(request: MathRequest):
    try:
        math = MathSolver()
        solution = math.solve(request.problem)
        return {"solution": solution}
    except Exception as e:
        logger.error(f"Math error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trade")
async def trade(request: TradingRequest):
    try:
        trading = MarketAnalyzer()
        result = trading.analyze(request.data)
        return result
    except Exception as e:
        logger.error(f"Trading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sysadmin/threats")
async def sysadmin_threats():
    try:
        sysadmin = SysadminManager()
        threats = sysadmin.analyze_traffic()
        return {"threats": threats}
    except Exception as e:
        logger.error(f"Sysadmin error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/captcha/solve")
async def solve_captcha(request: CaptchaRequest):
    try:
        captcha_solver = CaptchaSolver()
        from src.utils.browser_manager import BrowserManager
        browser = BrowserManager()
        driver = browser.launch_undetected_browser()
        driver.get(request.url)
        solution = captcha_solver.solve_captcha(driver)
        browser.close_browser()
        return {"solution": solution}
    except Exception as e:
        logger.error(f"CAPTCHA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/crawl")
async def crawl(request: CrawlRequest):
    try:
        crawler = WebCrawler()
        result = crawler.crawl(request.component, request.url)
        return {"result": result}
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag")
async def rag(request: PromptRequest):
    try:
        rag = RAG()
        response = rag.generate(request.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced_rag")
async def enhanced_rag(request: PromptRequest):
    try:
        enhanced_rag = EnhancedRAG()
        response = enhanced_rag.generate(request.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Enhanced RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/network/monitor")
async def network_monitor():
    try:
        network = NetworkManager()
        result = network.monitor_network()
        return {"result": result}
    except Exception as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mobility/connect")
async def mobility_connect(device_id: str):
    try:
        mobility = DeviceMobility()
        response = await mobility.connect_device(device_id)
        return {"response": response}
    except Exception as e:
        logger.error(f"Mobility error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/stt")
async def speech_to_text():
    try:
        voice_handler = VoiceHandler()
        text = voice_handler.speech_to_text()
        return {"text": text}
    except Exception as e:
        logger.error(f"Speech to text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/tts")
async def text_to_speech(request: VoiceRequest):
    try:
        voice_handler = VoiceHandler()
        output_file = voice_handler.text_to_speech(request.text)
        return {"audio_file": output_file}
    except Exception as e:
        logger.error(f"Text to speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/voice/stream")
async def voice_stream(websocket: WebSocket):
    try:
        await websocket.accept()
        voice_handler = VoiceHandler()
        async for text in websocket.iter_text():
            await voice_handler.stream_voice(text, websocket)
        await websocket.close()
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)