"""Cancellation reaches encoder subprocesses, not just the MLX worker."""

import asyncio
import contextlib
import multiprocessing
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

from media_gen.worker import WorkerDied, WorkerProcess


class EncodingBackend:
    def run(self, params, progress):
        with subprocess.Popen(
            [
                shutil.which("ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=256x256:rate=16",
                "-t",
                "60",
                "-c:v",
                "libx264",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as encoder:
            Path(params["pid_path"]).write_text(str(encoder.pid))
            progress(5)
            if params.get("exit_leader"):
                os._exit(3)
            encoder.wait()
        return {}


def encoding_worker(conn, spec, session_ready):
    # Inject only in this test child; exercise the real protocol and session setup.
    from media_gen import worker

    worker._load_backend = lambda _: EncodingBackend()
    worker.child_main(conn, spec, session_ready)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
@pytest.mark.parametrize("leader_exits_first", [False, True])
def test_cancel_kills_encoder_process_group(tmp_path, leader_exits_first):
    async def check():
        worker = WorkerProcess("encoding-test", {})
        ctx = multiprocessing.get_context("spawn")
        worker._process = ctx.Process(
            target=encoding_worker, args=(worker._child, {}, worker._session_ready), daemon=True
        )
        encoder_pid = None
        task = None
        try:
            await worker.start()
            assert os.getpgid(worker.pid) == worker.pid
            updates = []
            path = tmp_path / "encoder.pid"
            task = asyncio.create_task(
                worker.run(
                    "encode", {"pid_path": str(path), "exit_leader": leader_exits_first}, updates.append
                )
            )
            deadline = time.monotonic() + 10
            while not path.exists() or not updates:
                assert time.monotonic() < deadline
                await asyncio.sleep(0.05)
            encoder_pid = int(path.read_text())
            assert psutil.pid_exists(encoder_pid)
            assert updates == [5]
            if leader_exits_first:
                while worker.alive:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.05)
                assert psutil.Process(encoder_pid).status() != psutil.STATUS_ZOMBIE
            worker.kill()
            with pytest.raises(WorkerDied):
                await asyncio.wait_for(task, 10)
            deadline = time.monotonic() + 10
            while psutil.pid_exists(encoder_pid):
                if psutil.Process(encoder_pid).status() == psutil.STATUS_ZOMBIE:
                    break
                assert time.monotonic() < deadline, "ffmpeg survived cancellation"
                await asyncio.sleep(0.05)
        finally:
            worker.kill()
            await worker.stop()
            if task and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if encoder_pid and psutil.pid_exists(encoder_pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(encoder_pid, signal.SIGKILL)

    asyncio.run(check())


def test_kill_before_session_never_signals_parent_group(monkeypatch):
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._session_ready = threading.Event()
    worker._process = Mock()
    worker._process.is_alive.return_value = True
    group_kill = Mock()
    monkeypatch.setattr(os, "killpg", group_kill)
    worker.kill()
    worker._process.kill.assert_called_once_with()
    group_kill.assert_not_called()


def test_forced_stop_escalates_to_owned_group(monkeypatch):
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.model_id = "timeout-test"
    worker._session_ready = threading.Event()
    worker._session_ready.set()
    worker._process = Mock(pid=123456)
    worker._process.is_alive.side_effect = [True, True, False]
    worker._conn = Mock()
    group_kill = Mock()
    monkeypatch.setattr(os, "killpg", group_kill)
    asyncio.run(worker.stop())
    group_kill.assert_called_once_with(123456, signal.SIGKILL)
    assert [call.args for call in worker._process.join.call_args_list] == [(15,), (5,)]
    worker._conn.close.assert_called_once_with()
    assert not worker._session_ready.is_set()
