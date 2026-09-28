"""Piezo-dropper child process for the particle loader (runs in the `moku` env).

The camera SDKs live in the `thorcam` conda env and the Moku client in the
`moku` env, so the loader (thorcam) spawns this script (moku) and talks to it
over stdin/stdout, one JSON object per line.

Commands (stdin):
    {"cmd": "start"}    run the ramp planned from --conf (once per process)
    {"cmd": "stop"}     switch the output Off now, then set_defaults + relinquish
    {"cmd": "ping"}     heartbeat; answered with {"ev": "pong"}

Events (stdout):
    ready     connected (or fake) and planned: n_emitted, max_time, peak_v, conf_copy
    pulse     k, amp, t_elapsed
    stopped   reason ('stop'|'eof'|'heartbeat'|'complete'|'error'), t_stop_request,
              t_output_off (time.time() stamps), latency_s
    error     message

Deadman: the output is switched Off if stdin closes (the loader died) or no
command arrives for --heartbeat-s while the ramp is running. Nothing is emitted
until "start", so spawning the process alone never moves the piezo.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_core  # noqa: E402


def emit(ev, **kw):
    kw['ev'] = ev
    kw.setdefault('t', time.time())
    sys.stdout.write(json.dumps(kw) + '\n')
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--conf', required=True, help='conf_sweep.yaml to plan the ramp from')
    ap.add_argument('--fake', action='store_true', help='use FakeWaveformGenerator (no instrument)')
    ap.add_argument('--fake-latency-s', type=float, default=0.05)
    ap.add_argument('--heartbeat-s', type=float, default=5.0)
    ap.add_argument('--allow-headroom-warnings', action='store_true',
                    help='run even if sweep_limits reports the ramp exceeds the output '
                         'range (the loader runs unattended, so this is refused by default)')
    ap.add_argument('--no-conf-copy', action='store_true',
                    help='do not write the dated conf copy to the data directory')
    args = ap.parse_args(argv)

    plan = sweep_core.plan_ramp(sweep_core.load_conf(args.conf))
    if plan['headroom']['messages'] and not args.allow_headroom_warnings:
        emit('error', message='refusing: ' + ' | '.join(plan['headroom']['messages']))
        emit('stopped', reason='refused', t_stop_request=None, t_output_off=None,
             latency_s=None, calls=None)
        return 2
    if args.fake:
        args.no_conf_copy = True   # never litter the data directory from a fake run
    if args.fake:
        from fake_moku import FakeWaveformGenerator
        wg = FakeWaveformGenerator(plan['ip_address'], force_connect=True,
                                   call_latency_s=args.fake_latency_s)
    else:
        from moku.instruments import WaveformGenerator
        wg = WaveformGenerator(sweep_core.format_ip(plan['ip_address']), force_connect=True)

    lock = threading.Lock()
    stop_event = threading.Event()
    state = {'reason': None, 't_stop_request': None, 'last_cmd': time.monotonic(),
             'started': False}
    start_event = threading.Event()

    def request_stop(reason):
        if state['reason'] is None:
            state['reason'] = reason
            state['t_stop_request'] = time.time()
        stop_event.set()
        start_event.set()  # release main thread if never started

    def reader():
        for line in sys.stdin:
            state['last_cmd'] = time.monotonic()
            try:
                cmd = json.loads(line).get('cmd')
            except ValueError:
                emit('error', message='bad command line: ' + line.strip())
                continue
            if cmd == 'start':
                start_event.set()
            elif cmd == 'stop':
                request_stop('stop')
            elif cmd == 'ping':
                emit('pong')
        request_stop('eof')

    def watchdog():
        while not stop_event.wait(0.2):
            if state['started'] and time.monotonic() - state['last_cmd'] > args.heartbeat_s:
                request_stop('heartbeat')

    conf_copy = None
    if not args.no_conf_copy:
        conf_copy = sweep_core.save_conf_copy(plan)
    emit('ready', n_emitted=plan['n_emitted'], max_time=plan['max_time'],
         peak_v=plan['headroom']['peak_v'], headroom_ok=not plan['headroom']['messages'],
         warnings=plan['headroom']['messages'], conf_copy=conf_copy, fake=args.fake)

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    start_event.wait()
    try:
        if not stop_event.is_set():
            state['started'] = True
            state['last_cmd'] = time.monotonic()
            sweep_core.run_sweep(
                wg, plan, stop_event=stop_event, lock=lock,
                on_pulse=lambda k, a, te: emit('pulse', k=k, amp=a, t_elapsed=te))
            if state['reason'] is None:
                state['reason'] = 'complete'
                state['t_stop_request'] = time.time()
    except Exception as e:
        emit('error', message=repr(e))
        state['reason'] = state['reason'] or 'error'
        state['t_stop_request'] = state['t_stop_request'] or time.time()
    finally:
        try:
            sweep_core.output_off(wg, plan['channel'], lock=lock)
        except Exception as e:
            emit('error', message='output_off failed: ' + repr(e))
        t_off = time.time()
        try:
            with lock:
                wg.set_defaults()
                wg.relinquish_ownership()
        except Exception as e:
            emit('error', message='shutdown failed: ' + repr(e))
        emit('stopped', reason=state['reason'], t_stop_request=state['t_stop_request'],
             t_output_off=t_off,
             latency_s=(t_off - state['t_stop_request']) if state['t_stop_request'] else None,
             calls=(wg.names() if args.fake else None))


if __name__ == '__main__':
    sys.exit(main())
