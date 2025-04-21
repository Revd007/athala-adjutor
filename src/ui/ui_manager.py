from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.utils.database_manager import DatabaseManager
from logger import logger

app = FastAPI()
db = DatabaseManager()

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Athala Adjutor</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto p-4">
                <h1 class="text-3xl font-bold text-center">Athala Adjutor</h1>
                <div class="mt-4">
                    <input id="prompt" class="w-full p-2 border rounded" placeholder="Enter your prompt...">
                    <button onclick="sendPrompt()" class="mt-2 bg-blue-500 text-white p-2 rounded">Send</button>
                </div>
                <div id="response" class="mt-4 p-4 bg-white rounded shadow"></div>
                <img src="naruto_shimeji.png" class="absolute bottom-0 right-0 w-24 animate-bounce">
            </div>
            <script>
                async function sendPrompt() {
                    const prompt = document.getElementById('prompt').value;
                    const response = await fetch('/api/dialog', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt})
                    });
                    const data = await response.json();
                    document.getElementById('response').innerText = data.response;
                }
            </script>
        </body>
        </html>
        """
        db.store_update_log("ui", "accessed")
        logger.info("UI accessed")
        return html_content
    except Exception as e:
        logger.error(f"Error rendering UI: {str(e)}")
        return "<h1>Error rendering UI</h1>"