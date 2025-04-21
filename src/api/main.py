from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from src.ai.model import TransformerModel  # Dialog
from src.ai.codegen import CodeGenerator
from src.math.solver import MathSolver
from src.finance.market_analyzer import MarketAnalyzer
from src.finance.trading_dashboard import TradingDashboard
from src.sysadmin.parser import SysadminParser
from src.sysadmin.log_analyzer import LogAnalyzer
from src.stream.stream_handler import StreamHandler
from src.stream.voice_handler import VoiceHandler
from src.network.network_manager import NetworkManager
from src.utils.captcha_solver import CaptchaSolver
from src.deepsearch.scraper import WebScraper
from src.rag.rag import RAG
from src.rag.reranker import Reranker
from src.mobility.device_mobility import DeviceMobility
from src.project.project_manager import ProjectManager
from src.project.learning_path import LearningPath
from src.project.collaboration import Collaboration
from src.project.security import SecurityAnalyzer
from src.health.opencv_analyzer import OpenCVAnalyzer
from src.health.health_recommender import HealthRecommender
from logger import logger

app = FastAPI(title="AI Agent Assistant API")

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

class ProjectRequest(BaseModel):
    project_id: str

@app.post("/api/dialog")
async def dialog(request: PromptRequest):
    try:
        model = TransformerModel()
        response = model.generate(request.prompt)
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

@app.post("/api/trade/dashboard")
async def trade_dashboard(request: TradingRequest):
    try:
        dashboard = TradingDashboard()
        result = dashboard.generate_dashboard_data(request.data)
        return result
    except Exception as e:
        logger.error(f"Trading dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sysadmin/analyze")
async def sysadmin_analyze():
    try:
        parser = SysadminParser()
        threats = parser.analyze()
        return {"threats": threats}
    except Exception as e:
        logger.error(f"Sysadmin error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sysadmin/logs")
async def log_analyze():
    try:
        analyzer = LogAnalyzer()
        result = analyzer.analyze_logs()
        return {"result": result}
    except Exception as e:
        logger.error(f"Log analyzer error: {e}")
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
        crawler = WebScraper()
        result = crawler.scrape(request.url)
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

@app.post("/api/rerank")
async def rerank(request: PromptRequest):
    try:
        reranker = Reranker()
        response = reranker.rerank(request.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Rerank error: {e}")
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

@app.websocket("/api/video/stream")
async def video_stream(websocket: WebSocket):
    try:
        await websocket.accept()
        stream_handler = StreamHandler()
        await stream_handler.stream_video(websocket)
    except Exception as e:
        logger.error(f"Video stream error: {e}")
        raise

@app.post("/api/project/manage")
async def manage_project(request: ProjectRequest):
    try:
        manager = ProjectManager()
        result = manager.manage(request.project_id)
        return {"result": result}
    except Exception as e:
        logger.error(f"Project manager error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/learning_path")
async def learning_path(request: PromptRequest):
    try:
        learning = LearningPath()
        path = learning.generate_path(request.prompt)
        return {"path": path}
    except Exception as e:
        logger.error(f"Learning path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/collaborate")
async def collaborate(request: ProjectRequest):
    try:
        collab = Collaboration()
        result = collab.collaborate(request.project_id)
        return {"result": result}
    except Exception as e:
        logger.error(f"Collaboration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/security")
async def security_analyze():
    try:
        security = SecurityAnalyzer()
        result = security.analyze()
        return {"result": result}
    except Exception as e:
        logger.error(f"Security error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/health/pose")
async def pose_analyze():
    try:
        analyzer = OpenCVAnalyzer()
        result = analyzer.analyze_pose()
        return {"result": result}
    except Exception as e:
        logger.error(f"Pose analyzer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/health/recommend")
async def health_recommend(request: PromptRequest):
    try:
        recommender = HealthRecommender()
        result = recommender.recommend(request.prompt)
        return {"result": result}
    except Exception as e:
        logger.error(f"Health recommender error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)