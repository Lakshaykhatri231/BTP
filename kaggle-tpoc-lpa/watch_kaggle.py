"""Live progress for a Kaggle kernel run.

Usage:
    python3 watch_kaggle.py [owner/kernel-slug] [max_seconds]

Streams the kernel's execution log as it is produced (Kaggle's SSE log
endpoint -- `kaggle kernels logs` only returns text AFTER the run ends), and
prints each line with a local wall-clock stamp so you can tell "working" from
"stuck": a healthy run keeps emitting lines; a hung one goes silent, and the
'silent for Ns' heartbeat below makes that obvious.

Ctrl-C stops watching only -- it never affects the run itself.
"""
import sys
import time
import threading
from kaggle.api.kaggle_api_extended import KaggleApi

DEFAULT = "khatrilakshay2004/tpoc-lpa-phase-2-gpu-llm"
NOISE = ("Debugger warning", "make the debugger miss", "to python to disable frozen",
         "Note: Debugging will proceed")


def main():
    kernel = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    max_s = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
    api = KaggleApi()
    api.authenticate()

    last = [time.time()]
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            quiet = time.time() - last[0]
            if quiet >= 30:
                print(f"  ... silent for {quiet:.0f}s (run still open; no new output)", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    def run_status():
        try:
            return str(api.kernels_status(kernel).status).split(".")[-1]
        except Exception as e:
            return f"UNKNOWN({type(e).__name__})"

    t0 = time.time()
    n = 0
    print(f"watching {kernel} (max {max_s:.0f}s) -- Ctrl-C to stop watching", flush=True)
    try:
        # The log endpoint errors (HTTP 5xx) until the session has actually been
        # allocated to a worker, and the stream can also drop mid-run. So keep
        # reconnecting for as long as the run is still open, and only stop once
        # it has genuinely finished.
        # A reconnect replays the whole log, and the replay is chunked differently
        # from the live feed, so counting events to skip is unreliable (it
        # produced duplicated blocks). De-duplicate by CONTENT instead: a line
        # is printed only if we haven't already printed that exact text.
        printed = set()
        while time.time() - t0 <= max_s:
            try:
                for ev in api.kernels_logs_stream(kernel):
                    data = ev.get("data") if isinstance(ev, dict) else str(ev)
                    data = str(data).rstrip()
                    last[0] = time.time()
                    if not data or any(k in data for k in NOISE):
                        continue
                    if data in printed:
                        continue
                    printed.add(data)
                    n += 1
                    print(f"{time.strftime('%H:%M:%S')} | {data[:200]}", flush=True)
                    if time.time() - t0 > max_s:
                        break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                st = run_status()
                if st in ("COMPLETE", "ERROR", "CANCELLED"):
                    print(f"-- run finished with status {st} --")
                    return
                print(f"  ... log stream not ready yet ({type(e).__name__}); run status={st}; "
                      f"retrying in 15s", flush=True)
                last[0] = time.time()
                time.sleep(15)
                continue
            st = run_status()
            if st in ("COMPLETE", "ERROR", "CANCELLED"):
                print(f"-- stream ended ({n} lines): run finished with status {st} --")
                return
            time.sleep(5)  # stream dropped but run still open -> reconnect
        print(f"(stopped watching after {max_s:.0f}s)")
    except KeyboardInterrupt:
        print("\n(stopped watching; the Kaggle run is unaffected)")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
