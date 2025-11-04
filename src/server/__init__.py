from server.server import APP
import uvicorn


def serve():
    uvicorn.run(
        APP,
        host="0.0.0.0",
        port=8000,
        log_level="trace",
        timeout_graceful_shutdown=5,
    )
