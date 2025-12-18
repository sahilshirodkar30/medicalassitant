from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
from fastapi.responses import JSONResponse
from logger import logger
from modules.local_vectorstore import load_vectorstore

router = APIRouter()


def process_files_background(files: List[UploadFile]):
    """
    Heavy processing runs here
    (PDF parsing, embeddings, vector DB updates)
    """
    try:
        logger.info("Background task started: processing PDFs")
        load_vectorstore(files)
        logger.info("Background task completed: vectorstore updated")
    except Exception:
        logger.exception("Error in background PDF processing")


@router.post("/upload_pdfs/")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    try:
        logger.info("Received uploaded files")

        # Run heavy task in background (Render-safe)
        background_tasks.add_task(process_files_background, files)

        return {
            "message": "Files received. Processing started in background."
        }

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
