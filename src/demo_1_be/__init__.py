import uvicorn

def main() -> None:
    """Entry point for the application script."""
    uvicorn.run("demo_1_be.app:app", host="0.0.0.0", port=8000, reload=True)
