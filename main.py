#!/usr/bin/env python3
"""
Main entry point for running the Speite API server

This script starts the FastAPI server for the offline speech-to-text system.
"""

import sys
import logging
import uvicorn

from speite.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """
    Start the FastAPI server.
    """
    logger.info("=" * 60)
    logger.info("Speite - Offline Speech-to-Text System")
    logger.info("=" * 60)
    logger.info(f"Model: {settings.whisper_model_name}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Language: {settings.language}")
    logger.info(f"Host: {settings.api_host}")
    logger.info(f"Port: {settings.api_port}")
    logger.info("=" * 60)
    
    # Run the server
    uvicorn.run(
        "speite.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Disable reload for production
        log_level="info"
    )


if __name__ == "__main__":
    main()
