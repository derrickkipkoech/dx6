"""vst_out.py - plays DX Arranger's MIDI through a VST3 instrument plugin (e.g. MT Power Drum Kit)
inside the app itself: no FluidSynth, no loopMIDI, no separate host.

Needs:  pip install pedalboard sounddevice     (both are bundled into the Windows .exe by the build)
Used when config.json has  "synth": "vst"  and  "vst_path" pointing at the .vst3 file.

How the timing works: every MIDI message is stamped with the moment it arrived. The audio callback renders the
plugin one small block at a time and places each message inside the NEXT block at the same offset it arrived
at. So there is a fixed delay of one block (512 frames = ~11 ms) but no jitter.
"""
import os
import sys
import threading
import time


def probe(path, rate=44100):
    """Runs in a CHILD process (DX-Arranger.exe --vst-probe <plugin>): load the plugin and play one kick offline.
    If the plugin crashes Windows, only this helper dies and the app itself carries on."""
    import faulthandler
    faulthandler.enable()
    print("STEP importing libraries", flush=True)
    import numpy as np
    from pedalboard import load_plugin
    print("STEP loading plugin", flush=True)
    plugin = load_plugin(path)
    print("STEP plugin loaded: " + str(getattr(plugin, "name", "plugin")), flush=True)
    print("STEP playing a test kick", flush=True)
    audio = plugin([(bytes([0x99, 36, 110]), 0.0)], duration=0.6, sample_rate=rate, num_channels=2,
                   buffer_size=512, reset=True)
    print("PEAK %.6f" % float(np.abs(audio).max()), flush=True)


def check_in_child(path):
    """Returns the helper's peak level, or exits with a message saying how far it got before it crashed."""
    import subprocess
    cmd = ([sys.executable] if getattr(sys, "frozen", False)
           else [sys.executable, os.path.abspath(sys.argv[0])]) + ["--vst-probe", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           creationflags=0x08000000 if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        sys.exit("The plugin did not finish loading within 2 minutes.")
    lines = [x for x in (r.stdout + "\n" + r.stderr).splitlines() if x.strip()]
    steps = [x for x in lines if x.startswith("STEP")]
    if r.returncode != 0:
        where = steps[-1][5:] if steps else "starting"
        err = lines[-1] if lines else ""
        sys.exit(f"The plugin crashed while: {where} (exit code {r.returncode}). {err}")
    for x in lines:
        if x.startswith("PEAK"):
            return float(x.split()[1])
    return 0.0


class VstOut:
    dry = False

    def __init__(self, path, rate=44100, block=512, gain=1.0, device=None):
        try:
            import numpy as np
            import sounddevice as sd
            from pedalboard import load_plugin
        except ImportError as err:
            sys.exit(f"VST hosting needs the 'pedalboard' and 'sounddevice' packages ({err}).")
        self.np, self.sd = np, sd
        path = self._locate(path)
        peak = check_in_child(path)             # a crash here is survivable: only the helper process dies
        if peak < 1e-4:
            print("! The plugin produced no sound for a test kick. Its sound content was probably not found: put "
                  "MT-PowerDrumKit-Content.pdk next to the plugin, or set the content folder in the plugin's window.")
        else:
            print("VST plugin test passed.")
        try:
            self.plugin = load_plugin(path)
        except Exception as err:
            sys.exit(f"Could not load the plugin: {err}")
        if not getattr(self.plugin, "is_instrument", True):
            sys.exit("That plugin is an effect, not an instrument (it makes no sound from MIDI).")
        self.rate, self.block = int(rate), int(block)
        self.base_gain, self.master = float(gain), 1.0
        self.lock = threading.Lock()
        self.pending = []                     # (bytes, arrival time)
        self.error = ""
        self._first = True
        self._prev = time.perf_counter()
        self.stream = sd.OutputStream(samplerate=self.rate, blocksize=self.block, channels=2, dtype="float32",
                                      device=self._find_device(device), callback=self._callback, latency="low")
        self.stream.start()

    # ---- helpers ------------------------------------------------------------------
    @staticmethod
    def _locate(path):
        """Use vst_path if it exists; otherwise take the first .vst3 next to it, then in the standard VST3 folder."""
        if os.path.exists(path):
            return path
        import glob
        here = os.path.dirname(path) or "."
        spots = [here, os.path.join(os.environ.get("CommonProgramFiles", r"C:\Program Files\Common Files"), "VST3")]
        for folder in spots:
            found = sorted(glob.glob(os.path.join(folder, "*.vst3")))
            if found:
                print("Using the VST3 plugin:", found[0])
                return found[0]
        sys.exit(f"No VST3 plugin found.\n  Looked for: {path}\n  and any .vst3 in: " + "; ".join(spots) +
                 "\nPut the .vst3 next to DX-Arranger.exe or set \"vst_path\" in config.json.")

    def _find_device(self, name):
        if not name:
            return None
        for i, d in enumerate(self.sd.query_devices()):
            if d["max_output_channels"] > 0 and str(name).lower() in d["name"].lower():
                return i
        print(f"! audio output '{name}' not found; using the default. Available: " +
              ", ".join(d["name"] for d in self.sd.query_devices() if d["max_output_channels"] > 0))
        return None

    def _render(self, msgs, seconds, reset):
        return self.plugin(msgs, duration=seconds, sample_rate=self.rate, num_channels=2,
                           buffer_size=self.block, reset=reset)

    def _self_test(self):
        """Play one kick offline. A silent result almost always means the plugin cannot find its sound content."""
        try:
            a = self._render([(bytes([0x99, 36, 110]), 0.0)], 0.6, True)
            if float(self.np.abs(a).max()) < 1e-4:
                print("! The plugin produced no sound for a test kick. Its sound content was probably not found: run "
                      "MT Power Drum Kit's standalone once and point it at your content folder, then restart.")
            else:
                print("VST plugin ready:", getattr(self.plugin, "name", "plugin"))
        except Exception as err:
            print("! Plugin self-test failed:", err)

    # ---- audio thread -------------------------------------------------------------
    def _callback(self, outdata, frames, tinfo, status):
        now = time.perf_counter()
        prev, self._prev = self._prev, now
        with self.lock:
            batch, self.pending = self.pending, []
        dur = frames / self.rate
        msgs = [(m, min(max(t - prev, 0.0), dur * 0.999)) for m, t in batch]
        try:
            audio = self._render(msgs, dur, self._first)
            self._first = False
            block = audio.T * (self.base_gain * self.master)
            n = min(frames, block.shape[0])
            outdata[:n] = block[:n]
            if n < frames:
                outdata[n:] = 0
        except Exception as err:               # never let the audio thread die
            outdata.fill(0)
            if str(err) != self.error:
                self.error = str(err)
                print("! plugin error:", err)

    # ---- the interface the engine uses (same as the other outputs) -----------------
    def send(self, m):
        now = time.perf_counter()
        with self.lock:
            self.pending.append((bytes(m), now))

    def send_many(self, msgs):
        now = time.perf_counter()
        with self.lock:
            self.pending.extend((bytes(m), now) for m in msgs)

    def set_gain(self, pct):
        self.master = max(0, min(100, pct)) / 100.0
        if pct <= 0:                           # MUTE: also stop any ringing notes
            self.send_many([bytes([0xB0 | ch, 120, 0]) for ch in range(16)])

    def close(self):
        try:
            self.send_many([bytes([0xB0 | ch, 123, 0]) for ch in range(16)])
            time.sleep(self.block / self.rate * 2)
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
