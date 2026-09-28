"""Loader-side handle on moku/dropper_server.py (a child process in the moku env)."""
import json
import os
import queue
import subprocess
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, 'moku', 'dropper_server.py')
MOKU_PYTHON = os.path.expanduser('~/.conda/envs/moku/bin/python')


class DropperError(RuntimeError):
    pass


class DropperClient:
    def __init__(self, conf_path, fake=True, python=None, heartbeat_s=5.0, log=print):
        py = python or MOKU_PYTHON
        if not os.path.exists(py):
            raise DropperError(f"moku env python not found: {py}")
        cmd = [py, SERVER, '--conf', conf_path, '--heartbeat-s', str(heartbeat_s)]
        if fake:
            cmd.append('--fake')
        self.fake = fake
        self.log = log
        self.events = queue.Queue()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.running = False
        self.stopped = None
        self.t_last_ping = 0.0
        self.ready = self.wait_for('ready', timeout=30)
        if self.ready is None:
            raise DropperError(f"dropper server did not become ready (fake={fake}); "
                               f"events: {self.drain()}")

    def _read(self):
        for line in self.p.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                ev = {'ev': 'garbage', 'line': line}
            self.events.put(ev)
        self.events.put({'ev': 'eof'})

    def _send(self, cmd):
        try:
            self.p.stdin.write(json.dumps({'cmd': cmd}) + '\n')
            self.p.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def wait_for(self, name, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            try:
                ev = self.events.get(timeout=max(0.01, t_end - time.time()))
            except queue.Empty:
                break
            self._track(ev)
            if ev['ev'] == name:
                return ev
            if ev['ev'] == 'stopped' and name != 'stopped':
                return None
        return None

    def _track(self, ev):
        if ev['ev'] == 'stopped':
            self.running = False
            self.stopped = ev
        elif ev['ev'] == 'error':
            self.log(f"dropper error: {ev.get('message')}")

    def drain(self):
        out = []
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                return out
            self._track(ev)
            out.append(ev)

    def start(self):
        self._send('start')
        self.running = True
        self.t_start = time.time()

    def ping(self, every_s=1.0):
        if self.running and time.time() - self.t_last_ping >= every_s:
            self._send('ping')
            self.t_last_ping = time.time()

    def stop(self, timeout=10.0):
        t = time.time()
        self._send('stop')
        ev = self.stopped or self.wait_for('stopped', timeout)
        if ev is None:
            raise DropperError("dropper did not confirm stop -- CHECK THE MOKU OUTPUT")
        ev = dict(ev)
        ev['client_latency_s'] = (ev.get('t_output_off') or time.time()) - t
        return ev

    def close(self):
        try:
            if self.stopped is None:
                self.stop()
        finally:
            try:
                self.p.stdin.close()
            except Exception:
                pass
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
