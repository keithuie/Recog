import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Recog Interface", description="Premium Modern GUI for Recog")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request, "title": "Recog Dashboard"})

def start():
    """Launch the application server."""
    print("Starting Recog Interface...")
    uvicorn.run("interface.server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    start()
