"""
Start the Python server for the web UI.

    python run.py                # http://localhost:8000, WebSocket at ws://localhost:8000/ws
    python run.py --port 9000
    python run.py --reload       # restart automatically when you edit a .py file
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
