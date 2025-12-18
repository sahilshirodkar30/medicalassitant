import logging


def setup_loggger(name="medical_assistant_chatbot"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger


logger = setup_loggger()
logger.info("RAG process started")
logger.debug("Debug message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
logger.info("RAG process completed")