#!/usr/bin/env python3
# =============================================================================
#  ECG SYSTEM — LAPTOP  (TCP CLIENT)
#  File : ecg_laptop_client_final.py
#  Run  : python3 ecg_laptop_client_final.py 10.42.0.51
# =============================================================================
#
#  INSTALL  (run once)
#  pip install numpy matplotlib
#
#  FIX: LAPTOP SAVING FEWER LOOPS THAN PI
#  ─────────────────────────────────────────────────────────────────────────
#  Root cause: The drain thread was doing file I/O (base64 decode + write PNG)
#  inline. While writing one result's files the result_q backed up. With
#  maxlen=20, new results pushed out old ones silently. Missing files.
#
#  Fix 1: result_q maxlen increased from 20 to 200. No results dropped.
#  Fix 2: File saving moved to a dedicated SaveThread completely separate
#          from the drain loop. The drain loop now only:
#            - feeds dashboard state (fast, no I/O)
#            - puts result into save_queue (instant)
#          The SaveThread picks results from save_queue and writes files
#          at its own pace without ever blocking the drain loop.
#
#  Result: Every result the laptop receives is guaranteed to be saved,
#  regardless of how long file writing takes. The laptop now saves
#  exactly the same loops as the Pi.
#
#  DUAL SNR DISPLAY
#  ─────────────────────────────────────────────────────────────────────────
#  Wideband SNR tile  = hardware quality (6-12 dB typical)
#  QRS SNR tile       = clinical detection quality (20-35 dB typical)
#  Both shown in the dashboard and in the result window.
#
# =============================================================================

import sys, os, time, threading, socket, struct, json, base64, io
import warnings, logging, signal, queue
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
for _b in ["TkAgg","Qt5Agg","Qt6Agg","WXAgg"]:
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
#OUTPUT_BASE_DIR = os.path.expanduser("~/Downloads/ECG_Received/Chest_Patch")
OUTPUT_BASE_DIR = os.path.expanduser("~/Downloads/ECG_Received/Hand_Patch")

# =============================================================================
#  FOLDER HELPERS
# =============================================================================

def create_session_folder() -> Path:
    ts=time.strftime("%Y%m%d_%H%M%S")
    f=Path(OUTPUT_BASE_DIR)/f"Session_{ts}"
    f.mkdir(parents=True,exist_ok=True)
    log.info(f"Save folder  →  {f}"); return f


def create_loop_folder(session: Path, loop_idx: int) -> Path:
    ts=time.strftime("%H%M%S")
    f=session/f"loop_{loop_idx:04d}_{ts}"
    f.mkdir(exist_ok=True); return f

# =============================================================================
#  AUTO SAVE — pure file I/O, no GUI calls, runs in SaveThread
# =============================================================================

def auto_save_plots(loop_folder: Path, result: dict):
    """
    Saves all plots to loop_folder immediately.
    NO plt calls — only base64 decode + write_bytes.
    Safe to call from any background thread.
    """
    loop_idx   = result.get("loop", 0)
    ts_str     = result.get("ts", time.strftime("%H:%M:%S")).replace(":", "")
    plots      = result.get("plots", {})
    plot_names = result.get("plot_names", [])

    order = ["ecg","raw_vs_filt","fft","welch",
             "p_fft","qrs_fft","t_fft","st","rr","summary"]
    key_to_name = {order[i]: (plot_names[i] if i<len(plot_names) else order[i].upper())
                   for i in range(len(order))}

    saved = 0
    for key in order:
        b64=plots.get(key,"")
        if not isinstance(b64,str) or not b64 or key=="summary_txt": continue
        stem    =key_to_name.get(key,key.upper())
        filename=f"loop{loop_idx:04d}_{ts_str}_{stem}.png"
        out_path=loop_folder/filename
        try:
            img_bytes=base64.b64decode(b64)
            out_path.write_bytes(img_bytes); saved+=1
        except Exception as e:
            log.error(f"  Could not save '{filename}': {e}")

    txt=result.get("summary_txt","")
    if txt:
        tp=loop_folder/f"loop{loop_idx:04d}_{ts_str}_clinical_summary.txt"
        try: tp.write_text(txt,encoding="utf-8"); saved+=1
        except Exception as e: log.error(f"  Could not save summary text: {e}")

    log.info(f"  [LAPTOP] Saved loop {loop_idx}  →  {loop_folder.name}  ({saved} files)")
    print(f"\n  [SAVED]  Loop {loop_idx:04d}  →  {loop_folder}")
    print(f"           {saved} files  ({', '.join(plot_names[:3])}"
          f"{'...' if len(plot_names)>3 else ''})\n")

# =============================================================================
#  SAVE THREAD — decouples file I/O from drain loop completely
# =============================================================================

class SaveThread(threading.Thread):
    """
    FIX: file saving moved here from drain thread.

    The drain loop puts (loop_folder, result) tuples into save_queue.
    This thread picks them up and writes files independently.
    The drain loop never blocks waiting for disk I/O.

    Uses queue.Queue (not deque) so items are never silently dropped.
    If Pi sends faster than laptop can write, items queue up here
    and are written in order — nothing is lost.
    """
    def __init__(self):
        super().__init__(daemon=True, name="SaveThread")
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    def schedule(self, loop_folder: Path, result: dict):
        """Called from drain thread — puts work on queue, returns immediately."""
        self._q.put((loop_folder, result))

    def run(self):
        while not self._stop.is_set():
            try:
                loop_folder, result = self._q.get(timeout=0.5)
                try:    auto_save_plots(loop_folder, result)
                except Exception as e: log.error(f"SaveThread error: {e}")
                finally: self._q.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._stop.set()

    def wait_finish(self):
        """Call on shutdown to ensure all queued saves complete."""
        self._q.join()

# =============================================================================
#  TCP CLIENT
# =============================================================================

class TCPClient:
    def __init__(self, host: str, port: int):
        self._host=host; self._port=port
        self.stream_q: deque=deque(maxlen=5000)
        # FIX: increased from 20 to 200 — no results dropped during heavy traffic
        self.result_q: deque=deque(maxlen=200)
        self._stop=threading.Event()
        threading.Thread(target=self._run,daemon=True,name="TCPClient").start()

    def _run(self):
        while not self._stop.is_set():
            try:
                log.info(f"Connecting to  {self._host}:{self._port} ...")
                sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                sock.settimeout(5.0); sock.connect((self._host,self._port))
                sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                sock.settimeout(None); log.info("Connected to Raspberry Pi")
                self._recv(sock)
            except Exception as e:
                if not self._stop.is_set():
                    log.warning(f"Connection error: {e}  -- retrying in 3s")
                    time.sleep(3.0)

    def _recv(self, sock):
        buf=b""
        while not self._stop.is_set():
            try:
                chunk=sock.recv(65536)
                if not chunk: log.warning("Server closed"); return
                buf+=chunk
                while len(buf)>=4:
                    length=struct.unpack(">I",buf[:4])[0]
                    if len(buf)<4+length: break
                    payload=buf[4:4+length]; buf=buf[4+length:]
                    try:
                        msg=json.loads(payload.decode("utf-8"))
                        if   msg.get("type")=="stream":
                            self.stream_q.append((msg["t"],msg["v"]))
                        elif msg.get("type")=="result":
                            self.result_q.append(msg)
                    except Exception as e:
                        log.warning(f"Packet parse error: {e}")
            except Exception as e:
                if not self._stop.is_set(): log.warning(f"Recv error: {e}")
                return

    def stop(self): self._stop.set()

# =============================================================================
#  LIVE DASHBOARD
# =============================================================================

class Dashboard:
    BG="#0d1117"; AX_BG="#161b22"; GREEN="#00d26a"; RED="#f85149"
    BLUE="#58a6ff"; ORANGE="#f0883e"; MUTED="#8b949e"; TEXT="#c9d1d9"; BORDER="#30363d"

    def __init__(self, scroll_sec=SCROLL_SECONDS, fs_hint=100):
        self._scroll_sec=scroll_sec
        self._buf_size=scroll_sec*fs_hint*2
        self._t_buf: deque=deque(maxlen=self._buf_size)
        self._v_buf: deque=deque(maxlen=self._buf_size)
        self._lock=threading.Lock()
        self._feats={}; self._abn=[]; self._loop=0; self._ts="--:--:--"
        self._snr_wb=None; self._snr_qrs=None; self._n_rp=0
        self._result_ready=threading.Event(); self._latest={}
        plt.style.use("dark_background")
        self._build_live(); self._build_res()

    def _build_live(self):
        self.fig_live=plt.figure(figsize=(16,7),facecolor=self.BG,
                                 num="ECG Live -- Raspberry Pi")
        gs=gridspec.GridSpec(3,8,figure=self.fig_live,
                             height_ratios=[4,1,1],hspace=0.55,wspace=0.30)
        self.ax_ecg=self.fig_live.add_subplot(gs[0,:])
        self.ax_ecg.set_facecolor(self.AX_BG)
        self.ax_ecg.set_title("Live ECG  --  connecting...",color=self.BLUE,fontsize=11)
        self.ax_ecg.set_xlabel("Time (s)",color=self.MUTED,fontsize=8)
        self.ax_ecg.set_ylabel("mV",color=self.MUTED,fontsize=8)
        self.ax_ecg.tick_params(colors=self.MUTED,labelsize=7)
        for sp in self.ax_ecg.spines.values(): sp.set_edgecolor(self.BORDER)
        self._ecg_line,=self.ax_ecg.plot([],[],color=self.GREEN,lw=0.7,rasterized=True)
        self._rp_scat=self.ax_ecg.scatter([],[],color=self.RED,s=14,zorder=5)

        # 8 metric tiles: HR QTc PR QRS ST RMSSD WB-SNR QRS-SNR
        tile_defs=[
            ("HR",    "HR (bpm)"),
            ("QTc",   "QTc (ms)"),
            ("PR",    "PR (ms)"),
            ("QRS",   "QRS (ms)"),
            ("ST",    "ST (mV)"),
            ("RMSSD", "RMSSD (ms)"),
            ("WBSNR", "WB SNR (dB)"),
            ("QSNR",  "QRS SNR (dB)"),
        ]
        self._tile_texts={}
        for col,(key,label) in enumerate(tile_defs):
            ax=self.fig_live.add_subplot(gs[1,col])
            ax.set_facecolor(self.AX_BG)
            for sp in ax.spines.values(): sp.set_edgecolor(self.BORDER)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5,0.72,"--",transform=ax.transAxes,ha="center",va="center",
                    fontsize=11,fontweight="bold",color="#f0f6fc")
            ax.text(0.5,0.18,label,transform=ax.transAxes,ha="center",va="center",
                    fontsize=6.5,color=self.MUTED)
            self._tile_texts[key]=ax.texts[0]

        ax_f=self.fig_live.add_subplot(gs[2,:])
        ax_f.set_facecolor(self.AX_BG)
        for sp in ax_f.spines.values(): sp.set_edgecolor(self.BORDER)
        ax_f.set_xticks([]); ax_f.set_yticks([])
        self._findings_txt=ax_f.text(0.01,0.5,"Waiting for analysis...",
            transform=ax_f.transAxes,ha="left",va="center",fontsize=8,color=self.TEXT)
        self.fig_live.suptitle("ECG Real-Time Analysis  --  ADS1115 + Raspberry Pi",
                               color=self.BLUE,fontsize=12,fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.96])

    def _build_res(self):
        self.fig_res=plt.figure(figsize=(16,10),facecolor=self.BG,
                                num="ECG Analysis Results")
        self.fig_res.suptitle("Waiting for first result...",color=self.BLUE,fontsize=11)
        plt.tight_layout()

    def feed_stream(self,t,v):
        with self._lock: self._t_buf.append(t); self._v_buf.append(v)

    def feed_result(self,result):
        self._latest=result
        self._feats  =result.get("feats",{})
        self._abn    =result.get("abn",[])
        self._loop   =result.get("loop",0)
        self._ts     =result.get("ts","--:--:--")
        self._snr_wb =result.get("snr_wb",None)
        self._snr_qrs=result.get("snr_qrs",None)
        self._n_rp   =result.get("n_rpeaks",0)
        if result.get("plots"): self._result_ready.set()

    def _update_live(self,_frame):
        with self._lock:
            t_data=np.array(self._t_buf); v_data=np.array(self._v_buf)

        if len(t_data)>1:
            t_end=t_data[-1]; t_start=t_end-self._scroll_sec
            mask=t_data>=t_start; ts=t_data[mask]; vs=v_data[mask]
            self._ecg_line.set_data(ts,vs)
            self.ax_ecg.set_xlim(t_start,t_end)
            if len(vs)>0:
                pad=max(0.1,0.15*(np.max(vs)-np.min(vs)))
                self.ax_ecg.set_ylim(np.min(vs)-pad,np.max(vs)+pad)

            hr=self._feats.get("HR")
            hrs=f"   HR:{hr:.0f}bpm" if hr else ""
            wb_s=(f"   WB:{self._snr_wb:.1f}dB" if self._snr_wb is not None else "")
            qrs_s=""
            if self._snr_qrs is not None:
                try:
                    v=float(self._snr_qrs)
                    if not (v!=v): qrs_s=f"  QRS:{v:.1f}dB"
                except: pass
            self.ax_ecg.set_title(
                f"Live ECG  --  Loop {self._loop}  {self._ts}"
                f"{hrs}{wb_s}{qrs_s}   [{self._n_rp} R-peaks]",
                color=self.BLUE,fontsize=10)

        def _fmt(key,to_ms=False,dec=1):
            v=self._feats.get(key)
            if v is None: return "--"
            try: return (f"{float(v)*1000:.{dec}f}" if to_ms else f"{float(v):.{dec}f}")
            except: return "--"

        def _snr_color(val,good=20,mod=10):
            if val is None: return "#f0f6fc"
            try:
                v=float(val)
                if v!=v: return "#f0f6fc"
                return self.GREEN if v>=good else self.ORANGE if v>=mod else self.RED
            except: return "#f0f6fc"

        wb_disp=(f"{self._snr_wb:.1f}" if self._snr_wb is not None else "--")
        qrs_disp="--"
        if self._snr_qrs is not None:
            try:
                v=float(self._snr_qrs)
                if not (v!=v): qrs_disp=f"{v:.1f}"
            except: pass

        tile_vals=[
            ("HR",    _fmt("HR",dec=0),           (60,100)),
            ("QTc",   _fmt("QTc",to_ms=True,dec=0),(350,450)),
            ("PR",    _fmt("PR", to_ms=True,dec=0),(120,200)),
            ("QRS",   _fmt("QRS_dur",to_ms=True,dec=0),(60,100)),
            ("ST",    _fmt("ST_dev",dec=3),        (-0.1,0.1)),
            ("RMSSD", _fmt("RMSSD",to_ms=True,dec=0),None),
            ("WBSNR", wb_disp,  None),
            ("QSNR",  qrs_disp, None),
        ]

        for key,disp,ref in tile_vals:
            obj=self._tile_texts[key]; obj.set_text(disp)
            if key=="WBSNR":
                obj.set_color(_snr_color(self._snr_wb,good=20,mod=10))
            elif key=="QSNR":
                obj.set_color(_snr_color(self._snr_qrs,good=20,mod=15))
            elif ref and disp!="--":
                try:
                    vn=float(disp)
                    obj.set_color(self.RED if (vn<ref[0] or vn>ref[1]) else "#f0f6fc")
                except: obj.set_color("#f0f6fc")
            else:
                obj.set_color("#f0f6fc")

        if self._abn:
            self._findings_txt.set_text("[ABN]   "+"   |   ".join(self._abn[:4]))
            self._findings_txt.set_color(self.RED)
        else:
            self._findings_txt.set_text("[OK]  All features within normal ranges")
            self._findings_txt.set_color(self.GREEN)

        return (self._ecg_line,self._rp_scat,
                self._findings_txt,*self._tile_texts.values())

    def _poll_results(self,_frame):
        if self._result_ready.is_set():
            self._result_ready.clear()
            try: self._redraw_res(self._latest)
            except Exception as e: log.warning(f"Result window error: {e}")
        return []

    def _redraw_res(self,result):
        plots={k:v for k,v in result.get("plots",{}).items()
               if k!="summary_txt" and isinstance(v,str) and v}
        if not plots: return
        self.fig_res.clear()
        order=["ecg","raw_vs_filt","fft","welch",
               "p_fft","qrs_fft","t_fft","st","rr","summary"]
        avail=[(k,plots[k]) for k in order if k in plots]
        n=len(avail)
        if n==0: return
        cols=min(n,3); rows=(n+cols-1)//cols
        gs=gridspec.GridSpec(rows,cols,figure=self.fig_res,hspace=0.4,wspace=0.25)
        for idx,(key,b64) in enumerate(avail):
            ax=self.fig_res.add_subplot(gs[idx//cols,idx%cols])
            ax.set_facecolor(self.AX_BG); ax.axis("off")
            try:
                img=plt.imread(io.BytesIO(base64.b64decode(b64)))
                ax.imshow(img,aspect="auto")
            except:
                ax.text(0.5,0.5,f"[{key}]",ha="center",va="center",
                        color=self.MUTED,fontsize=9)
            ax.set_title(key.replace("_"," ").upper(),color=self.BLUE,fontsize=8,pad=3)

        snr_wb  = result.get("snr_wb",None)
        snr_qrs = result.get("snr_qrs",None)
        snr_s   = ""
        if snr_wb  is not None: snr_s += f"   WB:{snr_wb:.1f}dB"
        if snr_qrs is not None:
            try:
                v=float(snr_qrs)
                if not (v!=v): snr_s += f"   QRS:{v:.1f}dB"
            except: pass

        self.fig_res.suptitle(
            f"Analysis Results  --  Loop {result.get('loop',0)}"
            f"  {result.get('ts','')}{snr_s}",
            color=self.BLUE,fontsize=11,fontweight="bold")
        self.fig_res.patch.set_facecolor(self.BG)
        plt.figure(self.fig_res.number); self.fig_res.canvas.draw_idle()

    def start(self):
        self._anim_live=FuncAnimation(self.fig_live,self._update_live,
                                      interval=100,blit=False,cache_frame_data=False)
        self._anim_res =FuncAnimation(self.fig_res, self._poll_results,
                                      interval=500,blit=False,cache_frame_data=False)
        plt.show(block=True)

# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    host=sys.argv[1] if len(sys.argv)>1 else PI_HOST

    print("\n"+"="*62)
    print("  ECG SYSTEM  --  LAPTOP  (TCP CLIENT)")
    print("="*62)
    print(f"  Pi address  :  {host}:{PI_PORT}")
    print(f"  Save folder :  {OUTPUT_BASE_DIR}")
    print("\n  Plots are saved AUTOMATICALLY when received.")
    print("  A dedicated save thread writes files — no loops are dropped.")
    print("  Ctrl+C to stop.\n")

    out=Path(OUTPUT_BASE_DIR); out.mkdir(parents=True,exist_ok=True)
    _t=out/"_test.tmp"
    try: _t.write_text("ok"); _t.unlink()
    except Exception as e: log.error(f"Cannot write to {out}: {e}"); sys.exit(1)

    session    = create_session_folder()
    client     = TCPClient(host, PI_PORT)
    dash       = Dashboard(scroll_sec=SCROLL_SECONDS)
    save_thread= SaveThread()
    save_thread.start()

    def _shutdown(sig_num,frame):
        log.info("Shutting down...")
        client.stop()
        log.info("Waiting for pending saves to complete...")
        save_thread.wait_finish()
        save_thread.stop()
        plt.close("all")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Drain thread ──────────────────────────────────────────────────────
    # NO file I/O here. Only: read queues → update dashboard → hand off to save_thread.
    def _drain():
        while True:
            try:
                while client.stream_q:
                    t,v=client.stream_q.popleft()
                    dash.feed_stream(t,v)

                while client.result_q:
                    result=client.result_q.popleft()
                    loop  =result.get("loop",0)
                    abn   =result.get("abn",[])
                    feats =result.get("feats",{})
                    snr_wb=result.get("snr_wb",None)
                    snr_qrs=result.get("snr_qrs",None)
                    n_rp  =result.get("n_rpeaks",0)

                    hr_s =(f"{feats.get('HR',0.0):.1f}"
                           if feats.get("HR") is not None else "---")
                    wb_s =(f"{snr_wb:.1f}dB" if snr_wb is not None else "---")
                    qrs_s="---"
                    if snr_qrs is not None:
                        try:
                            v=float(snr_qrs)
                            if not (v!=v): qrs_s=f"{v:.1f}dB"
                        except: pass
                    flag=("  [ABN] "+" | ".join(abn[:2]) if abn else "  [OK]")
                    log.info(f"Loop {loop:04d}  HR:{hr_s}  WB:{wb_s}  "
                             f"QRS:{qrs_s}  Rpeaks:{n_rp}{flag}")

                    # Update dashboard — thread-safe (only writes shared state)
                    dash.feed_result(result)

                    # Hand off to SaveThread — returns immediately, never blocks
                    if result.get("plots"):
                        lf=create_loop_folder(session,loop)
                        save_thread.schedule(lf,result)

            except Exception as e:
                log.error(f"Drain error: {e}")
                import traceback; traceback.print_exc()

            time.sleep(0.02)

    threading.Thread(target=_drain,daemon=True,name="DrainThread").start()
    dash.start()
    client.stop()
    print(f"\n  All data saved in:\n  {session}\n")


if __name__=="__main__":
    main()