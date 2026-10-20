"""SIGTERM ends a process hosting the loopback generate endpoint (agentic_miner.serve_loopback), even while a turn
is in flight (uvicorn's own handling would wait for that request before letting the signal through)."""
import os
import signal
import subprocess
import sys

CHILD = r"""
import asyncio, socket
import httpx
from fastapi import FastAPI
from reliquary.miner.agentic_miner import serve_loopback

app = FastAPI()

@app.get("/slow")
async def slow():
    await asyncio.sleep(60)

async def main():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    await serve_loopback(app, port)
    async with httpx.AsyncClient(timeout=120) as client:
        pending = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/slow"))
        await asyncio.sleep(0.5)
        print("serving", flush=True)
        await pending

asyncio.run(main())
"""


def test_sigterm_ends_the_hosting_process():
    proc = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True, cwd=os.getcwd())
    try:
        assert proc.stdout.readline().strip() == "serving"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        proc.kill()
