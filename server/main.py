from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middelware.exception_handler import catch_exception_middleware
from routes.upload_files import router as upload_files_router
from routes.ask_question import router as ask_question_router




app = FastAPI(title="Medical Assistant API",description="Medical Assistant Chatbot")

# cross-origin resource sharing setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# middleware exception handler
app.middleware("https")(catch_exception_middleware)


# routers
#1.upload pdf doc
app.include_router(upload_files_router)

#2.asking queries

app.include_router(ask_question_router)

