from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.ask_question import router as ask_question_router
from routes.upload_files import router as upload_files_router


app = FastAPI(
title="Medical Assistant API",
description="Medical Assistant Chatbot",
)


# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


# -----------------------------
# Exception Middleware (SAFE)
# -----------------------------
@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
try:
return await call_next(request)
except Exception:
return JSONResponse(
status_code=500,
content={"error": "Internal Server Error"},
)


# -----------------------------
# Health / Root (MANDATORY FOR RENDER)
# -----------------------------
@app.get("/")
def root():
return {"status": "Medical Assistant API running"}


@app.get("/health")
def health():
return {"status": "ok"}


# -----------------------------
# Routers
# -----------------------------
app.include_router(upload_files_router)
app.include_router(ask_question_router)
