#!/usr/bin/env python3
# =============================================================================
#  ECG SYSTEM — LAPTOP  (TCP CLIENT)
#  File : ecg_laptop_client_final.py
#  Run  : python3 ecg_laptop_client_final.py 10.42.0.51
# =============================================================================
#
#  INSTALL  (run once on laptop)
#  pip install numpy matplotlib
#
#  WHAT SAVES WHERE
#  ─────────────────────────────────────────────────────────────────────────
#  ~/Downloads/ECG_Received/Session_TIMESTAMP/
#      ecg_live_TIMESTAMP.csv          ← continuous raw stream (100 Hz)
#      loop_0002_HHMMSS/
#          ECG_Filtered.png
#          Raw_vs_Filtered.png
#          ... (all plots)
#          clinical_summary.txt
#
#  The CSV records every stream sample the Pi sends to the laptop.
#  The Pi streams at 100 Hz (STREAM_DECIMATE = 5 on 500 Hz source).
#  Columns: timestamp_s, raw_mV
#  Note: the laptop receives only the RAW stream, not the filtered signal.
#  The filtered signal + R-peak markers are only available on the Pi CSV.
#  To get filtered data: use the Pi's ecg_live_*.csv file.
#
# =============================================================================

import sys, os, time, threading, socket, struct, json, base64, io, csv
import warnings, logging, signal
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("ECG-Laptop")

import numpy as np

import matplotlib
for _b in ["TkAgg", "Qt5Agg", "Qt6Agg", "WXAgg"]:
    try: matplotlib.use(_b); break
    except Exception: continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# =============================================================================
#  USER CONFIGURATION
# =============================================================================

PI_HOST         = "10.42.0.51"
PI_PORT         = 9000
SCROLL_SECONDS  = 8
OUTPUT_BASE_DIR = os.path.expanduser("~/Downloads/ECG_Received/Chest_Patch")

# =============================================================================
#  FOLDER HELPERS
# =============================================================================

def create_session_folder() -> Path:
    ts     = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(OUTPUT_BASE_DIR) / f"Session_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Save folder  →  {folder}")
    return folder


def create_loop_folder(session: Path, loop_idx: int) -> Path:
    ts     = time.strftime("%H%M%S")
    folder = session / f"loop_{loop_idx:04d}_{ts}"
    folder.mkdir(exist_ok=True)
    return folder

# =============================================================================
#  CSV LOGGER  (laptop side)
#  Writes every stream sample received from the Pi.
#  The Pi streams raw ADC values at 100 Hz (500 Hz / STREAM_DECIMATE=5).
#  Columns: timestamp_s, raw_mV
# =============================================================================

class LaptopCSVLogger:
    """
    Opens one CSV file at session start and appends every stream
    sample (t, v) the moment it arrives from the Pi.
    No buffering delay — data is on disk within milliseconds of receipt.
    Closed cleanly on Ctrl+C via close().
    """

    HEADER = ["timestamp_s", "raw_mV"]

    def __init__(self, session_folder: Path):
        ts           = time.strftime("%Y%m%d_%H%M%S")
        self._path   = session_folder / f"ecg_live_{ts}.csv"
        self._fh     = open(self._path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.HEADER)
        self._fh.flush()
        self._lock   = threading.Lock()
        self._count  = 0
        log.info(f"Laptop CSV  →  {self._path.name}")

    def write(self, t: float, v: float):
        """Write one sample. Called from drain thread — lock protected."""
        try:
            with self._lock:
                self._writer.writerow([f"{t:.6f}", f"{v:.4f}"])
                self._count += 1
        except Exception as e:
            log.warning(f"CSV write error: {e}")

    def close(self):
        """Flush and close on Ctrl+C."""
        try:
            with self._lock:
                self._fh.flush()
                self._fh.close()
            sz = self._path.stat().st_size // 1024
            log.info(f"Laptop CSV closed  →  {self._path.name}"
                     f"  ({self._count} samples, {sz} KB)")
            print(f"\n  Laptop ECG CSV:\n  {self._path}\n")
        except Exception as e:
            log.warning(f"CSV close error: {e}")

# =============================================================================
#  AUTO SAVE PLOTS  — no GUI, no input(), runs safely in background thread
# =============================================================================

def auto_save_plots(loop_folder: Path, result: dict):
    loop_idx   = result.get("loop", 0)
    ts_str     = result.get("ts", time.strftime("%H:%M:%S")).replace(":", "")
    plots      = result.get("plots", {})
    plot_names = result.get("plot_names", [])

    order = ["ecg", "raw_vs_filt", "fft", "welch",
             "p_fft", "qrs_fft", "t_fft", "st", "rr", "summary"]
    key_to_name = {}
    for i, key in enumerate(order):
        key_to_name[key] = plot_names[i] if i < len(plot_names) else key.upper()

    saved = 0
    for key in order:
        b64 = plots.get(key, "")
        if not isinstance(b64, str) or not b64 or key == "summary_txt":
            continue
        stem     = key_to_name.get(key, key.upper())
        out_path = loop_folder / f"loop{loop_idx:04d}_{ts_str}_{stem}.png"
        try:
            out_path.write_bytes(base64.b64decode(b64))
            saved += 1
        except Exception as e:
            log.error(f"  Could not save '{stem}': {e}")

    txt = result.get("summary_txt", "")
    if txt:
        txt_path = loop_folder / f"loop{loop_idx:04d}_{ts_str}_clinical_summary.txt"
        try:
            txt_path.write_text(txt, encoding="utf-8")
            saved += 1
        except Exception as e:
            log.error(f"  Could not save summary text: {e}")

    log.info(f"  Auto-saved loop {loop_idx}  →  {loop_folder.name}"
             f"  ({saved} files)")

# =============================================================================
#  TCP CLIENT
# =============================================================================

class TCPClient:
    def __init__(self, host: str, port: int):
        self._host = host; self._port = port
        self.stream_q: deque = deque(maxlen=5000)
        self.result_q: deque = deque(maxlen=20)
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True,
                         name="TCPClient").start()

    def _run(self):
        while not self._stop.is_set():
            try:
                log.info(f"Connecting to  {self._host}:{self._port} ...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self._host, self._port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(None)
                log.info("Connected to Raspberry Pi")
                self._recv(sock)
            except Exception as e:
                if not self._stop.is_set():
                    log.warning(f"Connection error: {e}  -- retrying in 3s")
                    time.sleep(3.0)

    def _recv(self, sock):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    log.warning("Server closed connection"); return
                buf += chunk
                while len(buf) >= 4:
                    length = struct.unpack(">I", buf[:4])[0]
                    if len(buf) < 4 + length: break
                    payload = buf[4:4+length]; buf = buf[4+length:]
                    try:
                        msg = json.loads(payload.decode("utf-8"))
                        if   msg.get("type") == "stream":
                            self.stream_q.append((msg["t"], msg["v"]))
                        elif msg.get("type") == "result":
                            self.result_q.append(msg)
                    except Exception as e:
                        log.warning(f"Packet parse error: {e}")
            except Exception as e:
                if not self._stop.is_set():
                    log.warning(f"Recv error: {e}")
                return

    def stop(self): self._stop.set()

# =============================================================================
#  LIVE DASHBOARD
# =============================================================================

class Dashboard:
    BG     = "#0d1117"
    AX_BG  = "#161b22"
    GREEN  = "#00d26a"
    RED    = "#f85149"
    BLUE   = "#58a6ff"
    ORANGE = "#f0883e"
    MUTED  = "#8b949e"
    TEXT   = "#c9d1d9"
    BORDER = "#30363d"

    def __init__(self, scroll_sec=SCROLL_SECONDS, fs_hint=100):
        self._scroll_sec = scroll_sec
        self._buf_size   = scroll_sec * fs_hint * 2
        self._t_buf: deque = deque(maxlen=self._buf_size)
        self._v_buf: deque = deque(maxlen=self._buf_size)
        self._lock = threading.Lock()

        self._feats    = {}
        self._abn      = []
        self._loop     = 0
        self._ts       = "--:--:--"
        self._snr_db   = None
        self._n_rp     = 0

        self._result_ready  = threading.Event()
        self._latest_result = {}

        plt.style.use("dark_background")
        self._build_live_window()
        self._build_result_window()

    def _build_live_window(self):
        self.fig_live = plt.figure(
            figsize=(16, 7), facecolor=self.BG,
            num="ECG Live -- Raspberry Pi")

        gs = gridspec.GridSpec(3, 7, figure=self.fig_live,
                               height_ratios=[4, 1, 1],
                               hspace=0.55, wspace=0.30)

        self.ax_ecg = self.fig_live.add_subplot(gs[0, :])
        self.ax_ecg.set_facecolor(self.AX_BG)
        self.ax_ecg.set_title("Live ECG  --  connecting...",
                               color=self.BLUE, fontsize=11)
        self.ax_ecg.set_xlabel("Time (s)", color=self.MUTED, fontsize=8)
        self.ax_ecg.set_ylabel("mV",       color=self.MUTED, fontsize=8)
        self.ax_ecg.tick_params(colors=self.MUTED, labelsize=7)
        for sp in self.ax_ecg.spines.values():
            sp.set_edgecolor(self.BORDER)
        self._ecg_line, = self.ax_ecg.plot([], [], color=self.GREEN,
                                            lw=0.7, rasterized=True)
        self._rp_scat   = self.ax_ecg.scatter([], [], color=self.RED,
                                               s=14, zorder=5)

        tile_defs = [
            ("HR",    "HR (bpm)"),
            ("QTc",   "QTc (ms)"),
            ("PR",    "PR (ms)"),
            ("QRS",   "QRS (ms)"),
            ("ST",    "ST (mV)"),
            ("RMSSD", "RMSSD (ms)"),
            ("SNR",   "SNR (dB)"),
        ]
        self._tile_texts = {}
        for col, (key, label) in enumerate(tile_defs):
            ax = self.fig_live.add_subplot(gs[1, col])
            ax.set_facecolor(self.AX_BG)
            for sp in ax.spines.values(): sp.set_edgecolor(self.BORDER)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.72, "--", transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#f0f6fc")
            ax.text(0.5, 0.18, label, transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=7, color=self.MUTED)
            self._tile_texts[key] = ax.texts[0]

        ax_f = self.fig_live.add_subplot(gs[2, :])
        ax_f.set_facecolor(self.AX_BG)
        for sp in ax_f.spines.values(): sp.set_edgecolor(self.BORDER)
        ax_f.set_xticks([]); ax_f.set_yticks([])
        self._findings_txt = ax_f.text(
            0.01, 0.5, "Waiting for first analysis...",
            transform=ax_f.transAxes, ha="left", va="center",
            fontsize=8, color=self.TEXT)

        self.fig_live.suptitle(
            "ECG Real-Time Analysis  --  ADS1115 + Raspberry Pi",
            color=self.BLUE, fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def _build_result_window(self):
        self.fig_res = plt.figure(
            figsize=(16, 10), facecolor=self.BG,
            num="ECG Analysis Results")
        self.fig_res.suptitle(
            "Waiting for first analysis result...",
            color=self.BLUE, fontsize=11)
        plt.tight_layout()

    def feed_stream(self, t: float, v: float):
        with self._lock:
            self._t_buf.append(t); self._v_buf.append(v)

    def feed_result(self, result: dict):
        self._latest_result = result
        self._feats  = result.get("feats", {})
        self._abn    = result.get("abn",   [])
        self._loop   = result.get("loop",  0)
        self._ts     = result.get("ts",    "--:--:--")
        self._snr_db = result.get("snr_db",  None)
        self._n_rp   = result.get("n_rpeaks", 0)
        if result.get("plots"):
            self._result_ready.set()

    def _update_live(self, _frame):
        with self._lock:
            t_data = np.array(self._t_buf)
            v_data = np.array(self._v_buf)

        if len(t_data) > 1:
            t_end   = t_data[-1]
            t_start = t_end - self._scroll_sec
            mask    = t_data >= t_start
            ts, vs  = t_data[mask], v_data[mask]
            self._ecg_line.set_data(ts, vs)
            self.ax_ecg.set_xlim(t_start, t_end)
            if len(vs) > 0:
                pad = max(0.1, 0.15 * (np.max(vs) - np.min(vs)))
                self.ax_ecg.set_ylim(np.min(vs) - pad, np.max(vs) + pad)

            hr  = self._feats.get("HR")
            hrs = f"   HR:{hr:.0f}bpm" if hr else ""
            sns = (f"   SNR:{self._snr_db:.1f}dB"
                   if self._snr_db is not None else "")
            self.ax_ecg.set_title(
                f"Live ECG  --  Loop {self._loop}  {self._ts}"
                f"{hrs}{sns}   [{self._n_rp} R-peaks]",
                color=self.BLUE, fontsize=10)

        def _fmt(key, to_ms=False, dec=1):
            v = self._feats.get(key)
            if v is None: return "--"
            try:
                return (f"{float(v)*1000:.{dec}f}" if to_ms
                        else f"{float(v):.{dec}f}")
            except: return "--"

        tile_vals = [
            ("HR",    _fmt("HR",      dec=0),            (60,   100)),
            ("QTc",   _fmt("QTc",     to_ms=True, dec=0),(350,  450)),
            ("PR",    _fmt("PR",      to_ms=True, dec=0),(120,  200)),
            ("QRS",   _fmt("QRS_dur", to_ms=True, dec=0),(60,   100)),
            ("ST",    _fmt("ST_dev",  dec=3),            (-0.1, 0.1)),
            ("RMSSD", _fmt("RMSSD",   to_ms=True, dec=0), None),
            ("SNR",
             f"{self._snr_db:.1f}" if self._snr_db is not None else "--",
             None),
        ]

        for key, disp, ref in tile_vals:
            obj = self._tile_texts[key]; obj.set_text(disp)
            if key == "SNR" and self._snr_db is not None:
                c = (self.GREEN  if self._snr_db >= 20 else
                     self.ORANGE if self._snr_db >= 10 else
                     self.RED)
            elif ref and disp != "--":
                try:
                    vn = float(disp)
                    c  = self.RED if (vn < ref[0] or vn > ref[1]) else "#f0f6fc"
                except: c = "#f0f6fc"
            else:
                c = "#f0f6fc"
            obj.set_color(c)

        if self._abn:
            self._findings_txt.set_text(
                "[ABN]   " + "   |   ".join(self._abn[:4]))
            self._findings_txt.set_color(self.RED)
        else:
            self._findings_txt.set_text("[OK]  All features within normal ranges")
            self._findings_txt.set_color(self.GREEN)

        return (self._ecg_line, self._rp_scat,
                self._findings_txt, *self._tile_texts.values())

    def _poll_results(self, _frame):
        if self._result_ready.is_set():
            self._result_ready.clear()
            try:
                self._redraw_result_window(self._latest_result)
            except Exception as e:
                log.warning(f"Result window redraw error: {e}")
        return []

    def _redraw_result_window(self, result: dict):
        plots = {k: v for k, v in result.get("plots", {}).items()
                 if k != "summary_txt" and isinstance(v, str) and v}
        if not plots: return

        self.fig_res.clear()
        order = ["ecg", "raw_vs_filt", "fft", "welch",
                 "p_fft", "qrs_fft", "t_fft", "st", "rr", "summary"]
        avail = [(k, plots[k]) for k in order if k in plots]
        n     = len(avail)
        if n == 0: return

        cols = min(n, 3); rows = (n + cols - 1) // cols
        gs   = gridspec.GridSpec(rows, cols, figure=self.fig_res,
                                 hspace=0.4, wspace=0.25)

        for idx, (key, b64) in enumerate(avail):
            ax = self.fig_res.add_subplot(gs[idx // cols, idx % cols])
            ax.set_facecolor(self.AX_BG); ax.axis("off")
            try:
                img = plt.imread(io.BytesIO(base64.b64decode(b64)))
                ax.imshow(img, aspect="auto")
            except Exception:
                ax.text(0.5, 0.5, f"[{key}]", ha="center", va="center",
                        color=self.MUTED, fontsize=9)
            ax.set_title(key.replace("_", " ").upper(),
                         color=self.BLUE, fontsize=8, pad=3)

        snr_i = (f"   SNR:{result.get('snr_db', 0):.1f}dB"
                 if result.get("snr_db") is not None else "")
        self.fig_res.suptitle(
            f"Analysis Results  --  Loop {result.get('loop', 0)}"
            f"  {result.get('ts', '')}{snr_i}",
            color=self.BLUE, fontsize=11, fontweight="bold")
        self.fig_res.patch.set_facecolor(self.BG)
        plt.figure(self.fig_res.number)
        self.fig_res.canvas.draw_idle()

    def start(self):
        self._anim_live = FuncAnimation(
            self.fig_live, self._update_live,
            interval=100, blit=False, cache_frame_data=False)
        self._anim_res  = FuncAnimation(
            self.fig_res, self._poll_results,
            interval=500, blit=False, cache_frame_data=False)
        plt.show(block=True)

# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    host = sys.argv[1] if len(sys.argv) > 1 else PI_HOST

    print("\n" + "=" * 62)
    print("  ECG SYSTEM  --  LAPTOP  (TCP CLIENT)")
    print("=" * 62)
    print(f"  Pi address  :  {host}:{PI_PORT}")
    print(f"  Save folder :  {OUTPUT_BASE_DIR}")
    print("\n  Plots saved automatically when received.")
    print("  CSV saved continuously from live stream.")
    print("  Press Ctrl+C on the Pi first, then Ctrl+C here.\n")

    out = Path(OUTPUT_BASE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    _t = out / "_test.tmp"
    try:
        _t.write_text("ok"); _t.unlink()
    except Exception as e:
        log.error(f"Cannot write to {out}: {e}")
        sys.exit(1)

    session    = create_session_folder()
    client     = TCPClient(host, PI_PORT)
    dash       = Dashboard(scroll_sec=SCROLL_SECONDS)
    csv_logger = LaptopCSVLogger(session)   # ← open CSV at session start

    def _shutdown(sig_num, frame):
        log.info("Shutting down...")
        client.stop()
        csv_logger.close()   # ← flush and close CSV on Ctrl+C
        plt.close("all")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Drain thread ──────────────────────────────────────────────────────
    # NO GUI calls here.
    # Stream samples  → write to CSV  +  feed dashboard buffer
    # Result packets  → feed dashboard state  +  save plot files
    def _drain():
        while True:
            try:
                # Stream samples → CSV + live ECG buffer
                while client.stream_q:
                    t, v = client.stream_q.popleft()
                    csv_logger.write(t, v)   # ← write every sample to CSV
                    dash.feed_stream(t, v)

                # Result packets → dashboard + plot files
                while client.result_q:
                    result  = client.result_q.popleft()
                    loop    = result.get("loop", 0)
                    abn     = result.get("abn", [])
                    feats   = result.get("feats", {})
                    snr     = result.get("snr_db", None)
                    n_rp    = result.get("n_rpeaks", 0)

                    hr_s  = (f"{feats.get('HR', 0.0):.1f}"
                             if feats.get("HR") is not None else "---")
                    snr_s = f"{snr:.1f}dB" if snr is not None else "---"
                    flag  = ("  [ABN] " + " | ".join(abn[:2])
                             if abn else "  [OK]")
                    log.info(f"Loop {loop:04d}  HR:{hr_s}  SNR:{snr_s}"
                             f"  Rpeaks:{n_rp}{flag}")

                    dash.feed_result(result)

                    if result.get("plots"):
                        lf = create_loop_folder(session, loop)
                        auto_save_plots(lf, result)

            except Exception as e:
                log.error(f"Drain thread error: {e}")
                import traceback; traceback.print_exc()

            time.sleep(0.02)

    threading.Thread(target=_drain, daemon=True, name="DrainThread").start()

    dash.start()   # blocks main thread

    client.stop()
    print(f"\n  All data saved in:\n  {session}\n")


if __name__ == "__main__":
    main()
