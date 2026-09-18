"""Opt-in acceptance test against the real qs CLI and a running local server.

Set DS4_QS_ROOT to the qs checkout and DS4_QS_BASE_URL to its /v1 endpoint.
The test runs the reported prompt twice, interrupting/restarting only the client.
Set DS4_QS_REQUIRE_COMPLETION=1 to also require a spontaneous final answer.
"""
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import subprocess
import unittest


PROMPT = "testa una skill e un tool, anche il tool nella skill, fai un check completo delle capacità"


@unittest.skipUnless(os.getenv("DS4_QS_ROOT") and os.getenv("DS4_QS_BASE_URL"),
                     "set DS4_QS_ROOT and DS4_QS_BASE_URL for the real qs CLI")
class QsClientLiveTests(unittest.TestCase):
    def test_tools_survive_interrupted_client_restart(self):
        """Check real tool/skill round trips, then restart only the client.

        Repeating successful capabilities is not a protocol failure. Cancel
        deliberately once each client has exercised the required capabilities
        and a return to root, without requiring the model to stop voluntarily.
        """
        uv = shutil.which("uv")
        self.assertIsNotNone(uv, "uv must be on PATH")
        env = dict(os.environ, BASE_URL=os.environ["DS4_QS_BASE_URL"],
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with tempfile.TemporaryDirectory(prefix="qs-live-") as temp:
            log_dir = Path(os.getenv("DS4_QS_LOG_DIR", temp))
            log_dir.mkdir(parents=True, exist_ok=True)
            for run in range(2):
                path = log_dir / f"qs-restart-{run + 1}.log"
                with self.subTest(client_run=run + 1), path.open("wb") as output:
                    proc = subprocess.Popen(
                        [uv, "run", "qs"], cwd=os.environ["DS4_QS_ROOT"],
                        stdin=subprocess.PIPE, stdout=output,
                        stderr=subprocess.STDOUT, env=env,
                        start_new_session=os.name != "nt",
                    )
                    try:
                        proc.stdin.write((PROMPT + "\nexit\n").encode("utf-8"))
                        proc.stdin.flush()
                        deadline = time.monotonic() + 600
                        while True:
                            text = path.read_text(encoding="utf-8", errors="replace")
                            self.assertNotIn("Traceback (most recent call last)", text)
                            self.assertNotIn("Provider response failed", text)
                            ready = (
                                "tool_end: mock-tool" in text
                                and "tool_start: mock-skill" in text
                                and any("tool_end: " + name in text for name in
                                        ("mock-sub-tool", "shared-tool", "nested-sub-tool"))
                                and "tool_end: exit_agent (depth=0, path=/)" in text
                            )
                            if ready:
                                # A completed call chain has reached root; the
                                # next client must recover even if decode is active.
                                if proc.poll() is not None:
                                    self.assertEqual(proc.returncode, 0, text[-16000:])
                                break
                            self.assertIsNone(proc.poll(), text[-16000:])
                            self.assertLess(time.monotonic(), deadline, text[-16000:])
                            time.sleep(0.25)
                    finally:
                        if proc.poll() is None:
                            if os.name == "nt":
                                subprocess.run(["taskkill", "/PID", str(proc.pid),
                                                "/T", "/F"], capture_output=True,
                                               check=True)
                            else:
                                os.killpg(proc.pid, signal.SIGTERM)
                            proc.wait(timeout=20)
                        proc.stdin.close()

    @unittest.skipUnless(os.getenv("DS4_QS_REQUIRE_COMPLETION"),
                         "set DS4_QS_REQUIRE_COMPLETION to require a final answer")
    def test_capabilities_survive_client_restart(self):
        uv = shutil.which("uv")
        self.assertIsNotNone(uv, "uv must be on PATH")
        env = dict(os.environ, BASE_URL=os.environ["DS4_QS_BASE_URL"],
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        log_dir = os.getenv("DS4_QS_LOG_DIR")
        for run in range(2):
            with self.subTest(client_run=run + 1):
                result = subprocess.run(
                    [uv, "run", "qs"], cwd=os.environ["DS4_QS_ROOT"],
                    input=PROMPT + "\nexit\n", text=True, encoding="utf-8",
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, timeout=600,
                )
                if log_dir:
                    target = Path(log_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"qs-client-{run + 1}.log").write_text(
                        result.stdout, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stdout[-16000:])
                self.assertNotIn("Traceback (most recent call last)", result.stdout)
                self.assertIn("tool_start: mock-skill", result.stdout)
                self.assertIn("tool_start: mock-tool", result.stdout)
                self.assertRegex(result.stdout,
                                 r"tool_start: (mock-sub-tool|shared-tool|nested-sub-tool)")
                self.assertIn("Completed node run: node=main depth=0", result.stdout)
                self.assertIn("Exiting...", result.stdout)


if __name__ == "__main__":
    unittest.main()
