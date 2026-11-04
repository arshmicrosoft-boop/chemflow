import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routes import router
from .database import init_db

app = FastAPI(title="ChemFlow API", description="Full-stack scientific research platform backend")

# Added before CORSMiddleware so it runs inside it: an unhandled error would
# otherwise leave without CORS headers and reach the browser as "Failed to fetch".
@app.middleware("http")
async def report_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.include_router(router)
