#!/usr/bin/env python3
# =============================================================================
#  ECG SYSTEM — RASPBERRY PI  (TCP SERVER)
#  File : ecg_pi_server_final.py
#  Run  : python3 ecg_pi_server_final.py
# =============================================================================
#
#  WIRING  (ADS1115 → Raspberry Pi 40-pin header)
#  ─────────────────────────────────────────────────────────────────────────
#  ADS1115  +    →  Pi Pin 1   (3.3 V)
#  ADS1115  -    →  Pi Pin 6   (GND)
#  ADS1115  D    →  Pi Pin 3   (GPIO 2 / I2C1 SDA)
#  ADS1115  C    →  Pi Pin 5   (GPIO 3 / I2C1 SCL)
#  ADS1115  ADDR →  LEFT  (address = 0x48)
#  ECG output   →  ADS1115  A0
#
#  FIXES IN THIS VERSION
#  ─────────────────────────────────────────────────────────────────────────
#  FIX-1  ADS1115 P0 attribute error
#    Error: "module 'adafruit_ads1x15.ads1115' has no attribute 'P0'"
#    Root cause: Newer Adafruit library moved P0/P1/P2/P3 channel constants.
#    Fix: Try ADS.P0 first; if AttributeError, fall back to integer channel (0).
#    AnalogIn(ads, 0) is equivalent to AnalogIn(ads, ADS.P0) in new library.
#
#  FIX-2  ADS_GAIN changed from 1 to 8
#    Old: gain=1 → ±4.096V range → LSB = 0.125 mV
#         ECG signal 1.4 mV peak uses only 11 of 32768 levels → poor resolution
#    New: gain=8 → ±0.512V range → LSB = 0.016 mV
#         ECG signal 1.4 mV peak uses 87 levels → 18 dB better resolution
#    WARNING: Ensure your ECG circuit output stays within ±0.512V.
#    If VREF_MID_VOLTS = 1.65V and ECG swings ±0.3V, output = 1.35–1.95V
#    → outside ±0.512V range → use gain=4 (±1.024V) as safer middle ground.
#    Default set to gain=4 for safety. Change to 8 if circuit confirmed.
#
#  FIX-3  Dual SNR — wideband + QRS-based (clinical standard)
#    Wideband SNR (existing): measures raw hardware noise, gives 6-8 dB.
#    QRS SNR (new, IEC 60601-2-47 standard): measures QRS power vs noise
#    floor during isoelectric baseline. Gives 25-35 dB. Clinically meaningful.
#    Both are reported. Wideband = hardware quality. QRS = detection quality.
#
#  CSV LOGGING
#  ─────────────────────────────────────────────────────────────────────────
#  One CSV file per session, records every sample continuously.
#  Columns: timestamp_s, raw_mV, filtered_mV, is_r_peak, loop, hr_bpm, snr_db
#
# =============================================================================

import os, sys, time, threading, logging, socket, json, struct, csv
import base64, io, signal, warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("ECG-Pi")

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, iirnotch
from scipy.fft    import rfft, rfftfreq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
#  USER CONFIGURATION
# =============================================================================

DATA_SOURCE       = "ads1115"

ADS_I2C_ADDRESS   = 0x48
# GAIN REVERTED TO 1 — gain=4 caused ADC saturation.
# With VREF_MID_VOLTS=1.65V the ECG circuit output is ~1.35–1.95V.
# gain=4 → ±1.024V range — this clips the 1.65V signal to the rail.
# gain=1 → ±4.096V range — safely covers 0–3.3V, works with any ECG circuit.
# Only increase gain if you confirm the ECG output after VREF subtraction
# stays within the gain's range (e.g. gain=2 → ±2.048V, gain=4 → ±1.024V).
ADS_GAIN          = 1
ADS_CHANNEL       = 0

FS                = 500
BUFFER_SECONDS    = 10
LOOP_INTERVAL_SEC = 2.0
SAVE_EVERY_N_LOOPS= 2

VREF_MID_VOLTS    = 1.65
VOLTS_TO_MV       = 1000.0
NOTCH_FREQ_HZ     = 50.0

TCP_HOST          = "0.0.0.0"
TCP_PORT          = 9000
STREAM_DECIMATE   = 5

OUTPUT_BASE_DIR   = os.path.expanduser("~/Downloads/ECG/Chest_Patch")
#OUTPUT_BASE_DIR   = os.path.expanduser("~/Downloads/ECG/Hand_Patch")

SNR_POOR_DB       = 10.0   # wideband SNR threshold
QRS_SNR_POOR_DB   = 20.0   # QRS-based SNR threshold (clinical)

# =============================================================================
#  CLINICAL REFERENCE RANGES
# =============================================================================

ECG_REF = {
    "HR"  : {"range"    : (60,    100  )},
    "P"   : {"amplitude": (0.05,  0.25 ), "duration": (0.08,  0.10)},
    "PR"  : {"duration" : (0.12,  0.20 )},
    "QRS" : {"amplitude": (0.5,   2.0  ), "duration": (0.06,  0.10)},
    "ST"  : {"deviation": (-0.1,  0.1  )},
    "T"   : {"amplitude": (0.05,  0.5  ), "duration": (0.10,  0.25)},
    "QT"  : {"duration" : (0.35,  0.45 )},
}

# =============================================================================
#  FOLDER HELPERS
# =============================================================================

def create_session_folder() -> Path:
    ts     = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(OUTPUT_BASE_DIR) / f"Session_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Session folder  →  {folder}")
    return folder


def create_loop_folder(session: Path, loop_idx: int) -> Path:
    ts     = time.strftime("%H%M%S")
    folder = session / f"loop_{loop_idx:04d}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

# =============================================================================
#  CSV LIVE LOGGER
# =============================================================================

class CSVLogger:
    HEADER = [
        "timestamp_s", "raw_mV", "filtered_mV",
        "is_r_peak", "loop", "hr_bpm", "snr_db",
    ]

    def __init__(self, session_folder: Path):
        ts            = time.strftime("%Y%m%d_%H%M%S")
        self._path    = session_folder / f"ecg_live_{ts}.csv"
        self._fh      = open(self._path, "w", newline="", buffering=1)
        self._writer  = csv.writer(self._fh)
        self._writer.writerow(self.HEADER)
        self._fh.flush()
        self._last_t  = -1.0
        log.info(f"CSV  ->  {self._path.name}")

    def write(self, t_arr, raw_arr, filtered_arr, r_peaks,
              loop_idx: int, hr, snr_db):
        rp_set = set(t_arr[r_peaks].tolist()) if len(r_peaks) else set()
        tol    = 1.5 / FS
        mask   = t_arr > self._last_t
        new_t  = t_arr[mask]; new_raw=raw_arr[mask]; new_flt=filtered_arr[mask]
        if len(new_t) == 0: return
        hr_str  = f"{hr:.2f}"     if hr     is not None else ""
        snr_str = f"{snr_db:.2f}" if snr_db is not None else ""
        rows = []
        for i in range(len(new_t)):
            t=new_t[i]; is_rp=0
            if rp_set:
                nearest=min(rp_set,key=lambda x:abs(x-t))
                if abs(nearest-t)<=tol: is_rp=1
            rows.append([f"{t:.6f}",f"{new_raw[i]:.4f}",
                         f"{new_flt[i]:.4f}",is_rp,loop_idx,hr_str,snr_str])
        try:
            self._writer.writerows(rows)
            self._last_t = float(new_t[-1])
        except Exception as e:
            log.warning(f"CSV write error: {e}")

    def close(self):
        try:
            self._fh.flush(); self._fh.close()
            sz = self._path.stat().st_size // 1024
            log.info(f"CSV closed  ->  {self._path.name}  ({sz} KB)")
            print(f"\n  Full ECG CSV:\n  {self._path}\n")
        except Exception as e:
            log.warning(f"CSV close error: {e}")

# =============================================================================
#  DEMO SOURCE
# =============================================================================

class DemoSource:
    def __init__(self, fs=FS, bpm=72.0):
        self._fs=fs; self._bpm=bpm; self._t=0.0
        self._buf: deque=deque()
        self._lock=threading.Lock(); self._stop=threading.Event()
        threading.Thread(target=self._run,daemon=True,name="Demo").start()

    @staticmethod
    def _make_beat(fs, bpm):
        rr=60.0/bpm; n=int(rr*fs); t=np.linspace(0.0,rr,n,endpoint=False)
        g=lambda mu,s,a: a*np.exp(-((t-mu)**2)/(2*s**2))
        return (g(0.10,0.025,0.18)+g(0.18,0.008,-0.10)+g(0.20,0.012,1.60)
               +g(0.22,0.008,-0.25)+g(0.32,0.040,0.35))

    def _run(self):
        step=0
        while not self._stop.is_set():
            bpm=self._bpm+4.0*np.sin(step*0.05); step+=1
            beat=self._make_beat(self._fs,bpm)
            chunk=beat+np.random.normal(0.0,0.022,len(beat))
            times=self._t+np.arange(len(chunk))/self._fs
            self._t+=len(chunk)/self._fs
            with self._lock: self._buf.extend(zip(times.tolist(),chunk.tolist()))
            time.sleep(len(beat)/self._fs*0.95)

    def read(self):
        with self._lock: out=list(self._buf); self._buf.clear()
        return out

    def stop(self): self._stop.set()

# =============================================================================
#  ADS1115 SOURCE  — FIX-1: P0 attribute error + FIX-2: gain=4
# =============================================================================

class ADS1115Source:
    """
    FIX-1 explanation:
    Newer adafruit_ads1x15 library removed the P0/P1/P2/P3 module-level
    constants from ads1115. AnalogIn now accepts a plain integer (0,1,2,3)
    for the channel. The fix tries the old attribute access first and falls
    back to the integer method which works on all library versions.

    FIX-2 explanation:
    gain=4 means ±1.024V input range.
    LSB = 1.024/32768 × 2 = 0.0000625 V = 0.0625 mV per step.
    A 1.4mV R-peak = 22 LSBs (vs 11 with gain=1). Resolution improved ~2×.
    If your ECG signal is centred on VREF_MID_VOLTS=1.65V and swings ±0.3V,
    actual range needed = 1.35V to 1.95V. This exceeds ±1.024V around 0V
    but the ADS1115 measures single-ended vs GND, so range is 0 to +1.024V.
    Ensure your ECG output after DC offset removal stays within 0 to +1.024V.
    If your circuit outputs 0–3.3V centred at 1.65V, set VREF_MID_VOLTS=1.65
    and use gain=1 or gain=2 to be safe.
    """
    _VALID_RATES = (8, 16, 32, 64, 128, 250, 475, 860)

    def __init__(self, address=ADS_I2C_ADDRESS, gain=ADS_GAIN,
                 channel=ADS_CHANNEL, fs=FS):
        import board, busio
        import adafruit_ads1x15.ads1115 as ADS_MODULE
        from adafruit_ads1x15.analog_in import AnalogIn

        i2c = busio.I2C(board.SCL, board.SDA)

        # FIX-1a: Mode enum import
        try:
            from adafruit_ads1x15.ads1x15 import Mode
        except ImportError:
            try:
                Mode = ADS_MODULE.Mode
            except AttributeError:
                class Mode:
                    CONTINUOUS = 0

        ads = ADS_MODULE.ADS1115(i2c, address=address, gain=gain)
        ads.mode = Mode.CONTINUOUS
        best = min((r for r in self._VALID_RATES if r >= fs), default=860)
        ads.data_rate = best

        # FIX-1b: P0/P1/P2/P3 channel constants
        # Try old-style module attribute first, fall back to integer index.
        # Both work identically with AnalogIn in all library versions.
        try:
            chan_map = {
                0: ADS_MODULE.P0,
                1: ADS_MODULE.P1,
                2: ADS_MODULE.P2,
                3: ADS_MODULE.P3,
            }
            ch = chan_map[channel]
            log.info(f"ADS1115 channel: using ADS_MODULE.P{channel} (old-style)")
        except AttributeError:
            # Newer library — AnalogIn accepts integer directly
            ch = channel
            log.info(f"ADS1115 channel: using integer {channel} (new-style)")

        self._chan = AnalogIn(ads, ch)
        log.info(f"ADS1115  0x{address:02X}  gain={gain}  ±{4.096/gain:.4f}V  "
                 f"{best}SPS  A{channel}  LSB={4.096/gain/32768*2*1000:.4f}mV")

        self._fs=fs; self._dt=1.0/fs; self._t=0.0
        self._buf: deque=deque()
        self._lock=threading.Lock(); self._stop=threading.Event(); self._errs=0
        threading.Thread(target=self._run,daemon=True,name="ADS1115").start()

    def _run(self):
        dt=self._dt
        while not self._stop.is_set():
            t_next=time.monotonic()+dt*0.98
            try:
                mv=(self._chan.voltage-VREF_MID_VOLTS)*VOLTS_TO_MV
                with self._lock:
                    self._buf.append((self._t,mv)); self._t+=dt
                self._errs=0
            except Exception as e:
                self._errs+=1
                if self._errs<=5 or self._errs%200==0:
                    log.warning(f"ADS1115 error #{self._errs}: {e}")
                time.sleep(dt)
            slack=t_next-time.monotonic()
            if slack>0: time.sleep(slack)

    def read(self):
        with self._lock: out=list(self._buf); self._buf.clear()
        return out

    def stop(self): self._stop.set()

# =============================================================================
#  DSP PIPELINE
# =============================================================================

def bandpass_filter(sig, fs):
    nyq=0.5*fs; b,a=butter(4,[0.5/nyq,40.0/nyq],btype="band")
    if len(sig)<3*(max(len(a),len(b))-1): return sig.copy()
    return filtfilt(b,a,sig)


def notch_filter(sig, fs, freq=NOTCH_FREQ_HZ):
    b,a=iirnotch(freq/(0.5*fs),Q=30.0)
    if len(sig)<18: return sig.copy()
    return filtfilt(b,a,sig)


def compute_snr_wideband(raw, filtered):
    """
    WIDEBAND SNR — hardware quality metric.
    Measures: how noisy is the raw ADC output overall?
    Formula: SNR = 10 × log10( mean(filtered²) / mean((raw-filtered)²) )
    Typical result: 6–12 dB for a basic ADS1115 circuit.
    Interpretation: characterises the hardware + electrode setup quality.
    NOT the clinical SNR — just a hardware benchmark.

    DC / FLAT SIGNAL GUARD:
    If the raw signal variance is near zero, the ADC is reading a pure DC
    voltage — ECG is not connected, electrodes are off, or ADC is saturated.
    In this case the bandpass filter outputs near-zero, giving a huge negative
    SNR like −184 dB which is confusing. We detect this and return nan instead
    so the summary cleanly shows 'N/A' with a clear diagnostic message.
    Detection threshold: std(raw) < 0.5 mV  (genuine ECG has std > 5 mV)
    """
    # DC/flat signal detection
    raw_std = float(np.std(raw))
    if raw_std < 0.5:
        # Signal is flat DC — ADC saturated, electrode off, or hardware fault
        # Return nan so summary shows N/A instead of a misleading −180 dB
        return np.nan

    noise=raw-filtered
    sp=float(np.mean(filtered**2)); np_=float(np.mean(noise**2))
    if np_<1e-12: return 99.0
    return float(10.0*np.log10(sp/np_))


def compute_snr_qrs(filtered, r_peaks, fs):
    """
    QRS-BASED SNR — clinical signal quality metric.
    Standard: IEC 60601-2-47 (ambulatory ECG) methodology.

    MATHEMATICS:
    For each detected R-peak at index r:
      QRS window  = filtered[r - 50ms : r + 50ms]   (100ms centred on R)
      P_qrs       = mean( QRS_window² )              (power of QRS complex)

    For the isoelectric baseline (electrically quiet period):
      Baseline window = filtered[r - 280ms : r - 150ms]
                        (130ms window ending 150ms before R, i.e. after T-wave
                         of previous beat and before P-wave of this beat)
      P_baseline  = var( baseline_window )  = mean( (x - mean(x))² )
                    (variance, not mean squared, because baseline may have
                     residual DC offset after filtering)

    QRS SNR (per beat) = 10 × log10( P_qrs / P_baseline )

    Final SNR = median of per-beat SNRs (median is robust to outlier beats)

    WHY THIS GIVES 20–35 dB:
      P_qrs     ≈ mean([0.5², 1.0², 1.4², 1.0², 0.5², ...]) ≈ 0.6–0.9 mV²
      P_baseline ≈ var(flat line with tiny noise) ≈ 0.0003–0.001 mV²
      Ratio     ≈ 600–3000
      SNR       = 10 × log10(1000) = 30 dB

    WHY THIS IS CLINICALLY CORRECT:
      The isoelectric baseline contains ONLY noise (no cardiac signal).
      Its variance is the true noise floor of your measurement system.
      The QRS window contains signal + noise, but signal dominates.
      The ratio directly answers: "can the detector reliably see QRS above noise?"
      This is exactly what IEC 60601-2-47 requires.
    """
    if len(r_peaks) < 2:
        return np.nan

    qrs_half  = int(0.050 * fs)   # 50ms either side of R = 100ms total QRS window
    bl_start  = int(0.280 * fs)   # baseline starts 280ms before R
    bl_end    = int(0.150 * fs)   # baseline ends   150ms before R
    # → 130ms window in the isoelectric region between T-wave and P-wave

    snr_per_beat = []
    N = len(filtered)

    for r in r_peaks:
        # QRS window
        q0 = max(0, r - qrs_half)
        q1 = min(N, r + qrs_half)
        if (q1 - q0) < 10:
            continue
        qrs_seg = filtered[q0:q1]
        p_qrs   = float(np.mean(qrs_seg**2))

        # Isoelectric baseline window
        b0 = max(0, r - bl_start)
        b1 = max(0, r - bl_end)
        if (b1 - b0) < 10:
            continue
        bl_seg    = filtered[b0:b1]
        p_baseline = float(np.var(bl_seg))   # variance = noise power, DC removed

        if p_baseline < 1e-12:
            continue

        beat_snr = 10.0 * np.log10(p_qrs / p_baseline)
        snr_per_beat.append(beat_snr)

    if not snr_per_beat:
        return np.nan

    return float(np.median(snr_per_beat))  # median is robust to outlier beats


def detect_r_peaks(filtered, fs):
    if len(filtered)<int(0.6*fs): return np.array([],dtype=int)
    nyq=0.5*fs; b,a=butter(3,[5.0/nyq,15.0/nyq],btype="band")
    win=max(1,int(0.12*fs))
    env=np.convolve(filtfilt(b,a,filtered)**2,np.ones(win)/win,mode="same")
    pks=np.array([],dtype=int)
    for mult in [0.5,0.3,0.15]:
        thr=np.mean(env)+mult*np.std(env)
        pks,_=find_peaks(env,height=thr,distance=max(1,int(0.25*fs)))
        if len(pks)>=2: break
    if len(pks)<2: return np.array([],dtype=int)
    hw=max(1,int(0.10*fs)); refined=[]
    for p in pks:
        s=max(0,p-hw); e=min(len(filtered),p+hw); seg=filtered[s:e]
        if len(seg)==0: continue
        refined.append(s+int(np.argmax(np.abs(seg))))
    if not refined: return np.array([],dtype=int)
    refined=np.unique(refined).astype(int)
    min_d=int(0.20*fs); keep=[refined[0]]
    for idx in refined[1:]:
        if idx-keep[-1]>=min_d: keep.append(idx)
    return np.array(keep,dtype=int)


def _pick(segs):
    return segs[int(np.argmin([np.sum(w**2) for w in segs]))] if segs else None


def extract_features(sig, r_peaks, t_arr, fs):
    f={}
    if len(r_peaks)<2: return f
    rr=np.diff(t_arr[r_peaks]); rr=rr[(rr>0.28)&(rr<2.5)]
    if len(rr)==0: return f
    f["HR"]=float(60.0/np.mean(rr))
    f["RMSSD"]=float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else np.nan
    Ps,Qs,Ts,STs=[],[],[],[]; Qd,Qdur=[],[]; N=len(sig)
    for r in r_peaks:
        p0,p1=r-int(0.26*fs),r-int(0.08*fs)
        q0,q1=r-int(0.05*fs),r+int(0.07*fs)
        st0,st1=r+int(0.07*fs),r+int(0.14*fs)
        t0,t1=r+int(0.14*fs),r+int(0.44*fs)
        if 0<p0 and p1<N: Ps.append(sig[p0:p1])
        if 0<q0 and q1<N:
            Qs.append(sig[q0:q1])
            pre=sig[q0:r]
            if len(pre)>0 and np.min(pre)<0:
                neg=np.where(pre<0)[0]
                Qd.append(float(abs(np.min(pre)))); Qdur.append(float(len(neg)/fs))
        if 0<st0 and st1<N: STs.append(sig[st0:st1])
        if 0<t0  and t1<N:  Ts.append(sig[t0:t1])
    repP,repQ,repT,repST=_pick(Ps),_pick(Qs),_pick(Ts),_pick(STs)
    if repP  is not None:
        f["P_amp"]=float(np.max(repP)); f["P_dur"]=float(len(repP)/fs)
    if repQ  is not None:
        f["QRS_amp"]=float(np.max(repQ)); f["QRS_dur"]=float(len(repQ)/fs)
        if repP is not None: f["PR"]=float((len(repP)+int(0.08*fs))/fs)
    if repT  is not None:
        pk=int(np.argmax(np.abs(repT))); f["T_amp"]=float(repT[pk])
        qt=float((int(0.14*fs)+len(repT))/fs)
        f["QTc"]=qt/np.sqrt(float(np.mean(rr)))
    if repST is not None:
        bl=float(np.mean(sig[:max(1,int(0.05*fs))])); f["ST_dev"]=float(np.mean(repST)-bl)
    f["Q_depth"]=float(np.mean(Qd))   if Qd   else np.nan
    f["Q_dur"]  =float(np.mean(Qdur)) if Qdur else np.nan
    f["_repP"]=repP; f["_repQRS"]=repQ; f["_repT"]=repT
    f["_repST"]=repST; f["_rr"]=rr
    return f

# =============================================================================
#  ABNORMALITY DETECTION
# =============================================================================

def _nn(v):
    return v is not None and not (isinstance(v,float) and np.isnan(v))


def detect_abnormalities(feats, snr_wb=None, snr_qrs=None):
    abn=[]; g=lambda k:feats.get(k,np.nan); R=ECG_REF
    if snr_wb is not None:
        if np.isnan(snr_wb):
            # nan means DC/flat signal — ADC saturated or no ECG connected
            abn.append("No ECG signal detected — ADC reads flat DC "
                       "(check: gain setting, electrode connection, VREF_MID_VOLTS)")
        elif snr_wb < SNR_POOR_DB:
            abn.append(f"Poor hardware SNR={snr_wb:.1f}dB (check electrodes/circuit)")
    if snr_qrs is not None and _nn(snr_qrs) and snr_qrs < QRS_SNR_POOR_DB:
        abn.append(f"Poor clinical SNR={snr_qrs:.1f}dB (QRS detection unreliable)")
    HR=g("HR"); RMSSD=g("RMSSD"); P_amp=g("P_amp"); P_dur=g("P_dur"); PR=g("PR")
    QRS_d=g("QRS_dur"); QRS_a=g("QRS_amp"); T_amp=g("T_amp")
    QTc=g("QTc"); ST=g("ST_dev"); Qd=g("Q_depth"); Qdur=g("Q_dur")
    if _nn(HR):
        if   HR<R["HR"]["range"][0]: abn.append(f"Bradycardia ({HR:.0f} bpm)")
        elif HR>R["HR"]["range"][1]: abn.append(f"Tachycardia ({HR:.0f} bpm)")
    if _nn(P_amp):
        if   P_amp<R["P"]["amplitude"][0]: abn.append("Absent/low P wave")
        elif P_amp>R["P"]["amplitude"][1]: abn.append("Tall P wave (R. atrial enlargement)")
    if _nn(P_dur) and P_dur>R["P"]["duration"][1]:
        abn.append("Wide P wave (L. atrial enlargement)")
    if _nn(PR):
        if   PR<R["PR"]["duration"][0]: abn.append("Short PR (WPW/pre-excitation)")
        elif PR>R["PR"]["duration"][1]: abn.append("Prolonged PR (1st-degree AV block)")
    if _nn(QRS_d) and QRS_d>R["QRS"]["duration"][1]:
        abn.append("Wide QRS (BBB/ventricular rhythm)")
    if _nn(QRS_a):
        if   QRS_a>R["QRS"]["amplitude"][1]: abn.append("Tall R wave (LVH)")
        elif QRS_a<R["QRS"]["amplitude"][0]: abn.append("Low voltage QRS")
    if _nn(ST):
        if   ST>R["ST"]["deviation"][1]:  abn.append(f"ST elevation {ST:+.3f}mV (STEMI?)")
        elif ST<R["ST"]["deviation"][0]:  abn.append(f"ST depression {ST:+.3f}mV (ischaemia?)")
    if _nn(T_amp):
        if   T_amp>R["T"]["amplitude"][1]: abn.append("Tall T wave (hyperkalaemia)")
        elif T_amp<0:                       abn.append("T-wave inversion (ischaemia/PE)")
        elif T_amp<R["T"]["amplitude"][0]: abn.append("Flat T wave (ischaemia/hypokalaemia)")
    if _nn(QTc):
        if   QTc>R["QT"]["duration"][1]: abn.append(f"Prolonged QTc {QTc:.3f}s (TdP risk)")
        elif QTc<R["QT"]["duration"][0]: abn.append(f"Short QTc {QTc:.3f}s")
    if _nn(Qd) and _nn(Qdur) and Qd>0.1 and Qdur>0.04:
        abn.append("Pathological Q waves (prior MI?)")
    if _nn(RMSSD) and RMSSD>0.20:
        abn.append(f"High RR variability RMSSD={RMSSD*1000:.0f}ms (AF?)")
    return list(dict.fromkeys(abn))

# =============================================================================
#  CLINICAL SUMMARY TEXT
# =============================================================================

def _fmt_v(v):
    if v is None: return "---"
    av=abs(v)
    if av==0: return "0"
    if av>=10:  return f"{v:.0f}"
    if av>=1:   return f"{v:.1f}"
    if av>=0.1: return f"{v:.2f}"
    return f"{v:.3f}"


def _get_status(disp_val, nrange):
    if nrange is None: return "---"
    lo,hi=nrange
    if lo<=disp_val<=hi: return "NORMAL"
    bound=lo if disp_val<lo else hi
    pct=abs(disp_val-bound)/max(abs(bound),1e-9)*100
    if pct>75: return "CRITICAL"
    elif pct>40: return "HIGH"
    else: return "MODERATE"


def build_summary_text(feats, abn, loop_idx, snr_wb=None, snr_qrs=None):
    ts=time.strftime("%Y-%m-%d  %H:%M:%S")
    rows=[
        ("Heart Rate",      "HR",      "bpm", False, (60.0,  100.0)),
        ("P Amplitude",     "P_amp",   "mV",  False, (0.05,  0.25 )),
        ("P Duration",      "P_dur",   "ms",  True,  (80.0,  100.0)),
        ("PR Interval",     "PR",      "ms",  True,  (120.0, 200.0)),
        ("QRS Duration",    "QRS_dur", "ms",  True,  (60.0,  100.0)),
        ("QRS Amplitude",   "QRS_amp", "mV",  False, (0.5,   2.0  )),
        ("T Amplitude",     "T_amp",   "mV",  False, (-0.5,  0.5  )),
        ("QTc (Bazett)",    "QTc",     "ms",  True,  (350.0, 450.0)),
        ("ST Deviation",    "ST_dev",  "mV",  False, (-0.10, 0.10 )),
        ("Q-wave Depth",    "Q_depth", "mV",  False, None          ),
        ("Q-wave Duration", "Q_dur",   "ms",  True,  None          ),
        ("RR RMSSD",        "RMSSD",   "ms",  True,  None          ),
    ]
    HDR=(f"  {'Feature':<20}  {'Value':>8}  {'Unit':<4}"
         f"  {'Normal Range':<13}  {'Status':<10}")
    SEP="  "+"-"*65
    lines=["="*68,"  ECG REAL-TIME CLINICAL SUMMARY",
           f"  Loop {loop_idx:04d}     {ts}","="*68,"",HDR,SEP]
    for name,key,unit,to_ms,nrange in rows:
        raw=feats.get(key,np.nan)
        if not _nn(raw):
            nr=(f"{_fmt_v(nrange[0])}-{_fmt_v(nrange[1])}" if nrange else "---")
            lines.append(f"  {name:<20}  {'N/A':>8}  {unit:<4}  {nr:<13}  {'N/A':<10}")
            continue
        disp=raw*1000.0 if to_ms else raw
        nr=(f"{_fmt_v(nrange[0])}-{_fmt_v(nrange[1])}" if nrange else "---")
        status=_get_status(disp,nrange)
        lines.append(f"  {name:<20}  {disp:>8.2f}  {unit:<4}  {nr:<13}  {status:<10}")

    lines.append(SEP)
    # Dual SNR rows
    # Wideband SNR display — nan means flat/DC signal (no ECG detected)
    if snr_wb is None or (isinstance(snr_wb, float) and np.isnan(snr_wb)):
        wb_str  = "N/A"
        wb_stat = "NO SIGNAL"
    elif snr_wb >= 20: wb_str = f"{snr_wb:.1f}"; wb_stat = "GOOD"
    elif snr_wb >= 10: wb_str = f"{snr_wb:.1f}"; wb_stat = "MODERATE"
    else:              wb_str = f"{snr_wb:.1f}"; wb_stat = "POOR"    
    lines.append(
        f"  {'Wideband SNR':<20}  {wb_str:>8}  {'dB':<4}  {'>10=ok':<13}  {wb_stat:<10}")
    qrs_str  = f"{snr_qrs:.1f}" if (_nn(snr_qrs)) else "N/A"
    qrs_stat = ("GOOD" if _nn(snr_qrs) and snr_qrs>=20
                else "MODERATE" if _nn(snr_qrs) and snr_qrs>=15
                else "POOR")
    lines.append(
        f"  {'QRS SNR (clinical)':<20}  {qrs_str:>8}  {'dB':<4}  {'>20=good':<13}  {qrs_stat:<10}")

    lines+=["","  CLINICAL FINDINGS",SEP]
    if abn:
        for a in abn: lines.append(f"  [ABNORMAL]  {a}")
    else:
        lines.append("  [NORMAL]  All features within normal clinical ranges")
    lines+=["","="*68]
    return "\n".join(lines)

# =============================================================================
#  PLOT HELPERS
# =============================================================================

_DARK={"figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
       "axes.edgecolor":"#30363d","axes.labelcolor":"#8b949e",
       "xtick.color":"#8b949e","ytick.color":"#8b949e",
       "grid.color":"#21262d","text.color":"#c9d1d9","lines.linewidth":0.9}


def _style_ax(ax,title,xl="",yl=""):
    ax.set_facecolor("#161b22"); ax.set_title(title,color="#58a6ff",fontsize=9,pad=4)
    if xl: ax.set_xlabel(xl,fontsize=8)
    if yl: ax.set_ylabel(yl,fontsize=8)
    ax.grid(True,alpha=0.2,linewidth=0.5); ax.tick_params(labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d"); sp.set_linewidth(0.5)


def _spectrum(sig,fs):
    f=rfftfreq(len(sig),1.0/fs); m=np.abs(rfft(sig))/len(sig)
    return f,m,m**2


def _save_figure(fig, filepath: Path) -> str:
    try:
        fig.savefig(str(filepath),dpi=110,bbox_inches="tight",
                    facecolor=fig.get_facecolor(),format="png")
        plt.close(fig)
        if not filepath.exists(): return ""
        sz=filepath.stat().st_size
        if sz==0: return ""
        b64=base64.b64encode(filepath.read_bytes()).decode("utf-8")
        log.info(f"  Saved  {filepath.name}  ({sz//1024} KB)")
        return b64
    except Exception as e:
        log.error(f"  SAVE ERROR {filepath.name}: {e}")
        try: plt.close(fig)
        except: pass
        return ""

# =============================================================================
#  PLOT GENERATION + SAVING
# =============================================================================

def build_and_save_plots(t_arr, raw, filtered, r_peaks, feats, abn,
                         loop_idx, fs, snr_wb, snr_qrs, loop_folder: Path):
    plt.rcParams.update(_DARK)
    plots_b64={}; plot_names=[]
    txt=build_summary_text(feats,abn,loop_idx,snr_wb,snr_qrs)

    def _do(fig,key,name):
        path=loop_folder/f"{name}.png"
        b64=_save_figure(fig,path)
        if b64: plots_b64[key]=b64; plot_names.append(name)

    hr_txt=feats.get("HR",np.nan)
    hr_s=f"   HR:{hr_txt:.0f}bpm" if _nn(hr_txt) else "   HR:---"
    snr_s=f"   WB:{snr_wb:.1f}dB  QRS:{snr_qrs:.1f}dB" if (_nn(snr_qrs) and snr_wb is not None) else ""
    n_rp=len(r_peaks)

    # 1. Filtered ECG
    fig,ax=plt.subplots(figsize=(13,3.8),facecolor="#0d1117")
    ax.plot(t_arr,filtered,color="#00d26a",lw=0.65,rasterized=True)
    if n_rp:
        ax.scatter(t_arr[r_peaks],filtered[r_peaks],color="#f85149",
                   s=18,zorder=5,label=f"R-peaks ({n_rp})")
        ax.legend(fontsize=7,facecolor="#161b22",edgecolor="#30363d",labelcolor="#c9d1d9")
    _style_ax(ax,f"Filtered ECG  Loop {loop_idx}{hr_s}{snr_s}  [{n_rp} R-peaks]",
              "Time (s)","Amplitude (mV)")
    _do(fig,"ecg","ECG_Filtered")

    # 2. Raw vs Filtered
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(13,5),sharex=True,facecolor="#0d1117")
    ax0.plot(t_arr,raw,color="#888780",lw=0.5,rasterized=True)
    _style_ax(ax0,f"Raw ADC Signal  Loop {loop_idx}","","Amplitude (mV)")
    ax1.plot(t_arr,filtered,color="#00d26a",lw=0.65,rasterized=True)
    if n_rp: ax1.scatter(t_arr[r_peaks],filtered[r_peaks],color="#f85149",s=14,zorder=5)
    _style_ax(ax1,f"Filtered (0.5-40Hz + {NOTCH_FREQ_HZ:.0f}Hz notch)","Time (s)","Amplitude (mV)")
    fig.suptitle(f"Raw vs Filtered   WB-SNR:{snr_wb:.1f}dB" if snr_wb else "Raw vs Filtered",
                 color="#58a6ff",fontsize=10)
    _do(fig,"raw_vs_filt","Raw_vs_Filtered")

    # 3. Full FFT
    freq_arr,mag_arr,pwr_arr=_spectrum(filtered,fs); mask=freq_arr<=80
    fig,(axm,axp)=plt.subplots(1,2,figsize=(11,3.5),facecolor="#0d1117")
    axm.plot(freq_arr[mask],mag_arr[mask],color="#a371f7",lw=0.8)
    _style_ax(axm,"FFT Magnitude (0-80 Hz)","Hz","Magnitude")
    axp.semilogy(freq_arr[mask],pwr_arr[mask]+1e-20,color="#f0883e",lw=0.8)
    _style_ax(axp,"FFT Power Spectrum","Hz","Power")
    fig.suptitle(f"Full Signal FFT  Loop {loop_idx}",color="#58a6ff",fontsize=10)
    _do(fig,"fft","FFT_Full_Signal")

    # 4. Welch PSD
    npg=min(1024,max(8,len(filtered)//2)); fw,psd=welch(filtered,fs,nperseg=npg)
    fig,ax=plt.subplots(figsize=(8,3.5),facecolor="#0d1117")
    ax.semilogy(fw,psd+1e-20,color="#3fb950",lw=0.9)
    _style_ax(ax,f"Welch PSD  Loop {loop_idx}","Frequency (Hz)","PSD (V^2/Hz)")
    _do(fig,"welch","Welch_PSD")

    # 5-7. Per-wave FFTs
    for fkey,label,clr,pkey,dname in [
        ("_repP","P-Wave FFT","#79c0ff","p_fft","FFT_P_Wave"),
        ("_repQRS","QRS FFT","#f85149","qrs_fft","FFT_QRS"),
        ("_repT","T-Wave FFT","#56d364","t_fft","FFT_T_Wave"),
    ]:
        seg=feats.get(fkey)
        if seg is not None and len(seg)>4:
            wf,wm,_=_spectrum(seg,fs)
            fig,ax=plt.subplots(figsize=(6,3.2),facecolor="#0d1117")
            ax.plot(wf,wm,color=clr,lw=0.9)
            _style_ax(ax,f"{label}  Loop {loop_idx}","Hz","Magnitude")
            _do(fig,pkey,dname)

    # 8. ST segment
    repST=feats.get("_repST")
    if repST is not None and len(repST)>2:
        t_ms=np.arange(len(repST))/fs*1000
        fig,ax=plt.subplots(figsize=(7,3.2),facecolor="#0d1117")
        ax.plot(t_ms,repST,color="#ff7b72",lw=1.1)
        stv=feats.get("ST_dev",np.nan)
        if _nn(stv): ax.axhline(stv,color="#f0883e",ls="--",lw=1,label=f"Mean ST:{stv:+.3f}mV")
        ax.axhline(0.1,color="#f85149",ls=":",lw=0.8,alpha=0.7,label="+0.1mV")
        ax.axhline(-0.1,color="#79c0ff",ls=":",lw=0.8,alpha=0.7,label="-0.1mV")
        ax.axhline(0,color="#8b949e",ls="-",lw=0.4,alpha=0.4)
        ax.legend(fontsize=7,facecolor="#161b22",edgecolor="#30363d",labelcolor="#c9d1d9")
        _style_ax(ax,f"ST Segment  Loop {loop_idx}","Time (ms)","Amplitude (mV)")
        _do(fig,"st","ST_Segment")

    # 9. RR tachogram
    rr_arr=feats.get("_rr")
    if rr_arr is not None and len(rr_arr)>=2:
        fig,ax=plt.subplots(figsize=(9,3.2),facecolor="#0d1117")
        ax.plot(np.arange(1,len(rr_arr)+1),rr_arr*1000,"o-",
                color="#79c0ff",ms=3.5,lw=0.9,rasterized=True)
        ax.axhline(600,color="#f85149",ls="--",lw=0.8,alpha=0.6,label="100 bpm")
        ax.axhline(1000,color="#3fb950",ls="--",lw=0.8,alpha=0.6,label="60 bpm")
        ax.legend(fontsize=7,facecolor="#161b22",edgecolor="#30363d",labelcolor="#c9d1d9")
        rmssd=feats.get("RMSSD",np.nan)
        rs=f"   RMSSD:{rmssd*1000:.1f}ms" if _nn(rmssd) else ""
        _style_ax(ax,f"RR Tachogram  Loop {loop_idx}{rs}","Beat number","RR interval (ms)")
        _do(fig,"rr","RR_Tachogram")

    # 10. Clinical summary
    fig,ax=plt.subplots(figsize=(12,9.5),facecolor="#0d1117")
    ax.axis("off")
    ax.text(0.02,0.98,txt,transform=ax.transAxes,fontsize=7.2,va="top",ha="left",
            family="monospace",color="#c9d1d9",linespacing=1.4,
            bbox=dict(boxstyle="round",facecolor="#0d1117",edgecolor="#30363d",alpha=1.0))
    fig.suptitle(f"Clinical Summary  Loop {loop_idx}",
                 fontsize=12,fontweight="bold",color="#f85149")
    _do(fig,"summary","Clinical_Summary")

    (loop_folder/"clinical_summary.txt").write_text(txt,encoding="utf-8")
    n_saved=len(list(loop_folder.glob("*.png")))
    log.info(f"Loop {loop_idx} complete  →  {loop_folder.name}  ({n_saved} PNG files + clinical_summary.txt)")
    plots_b64["summary_txt"]=txt
    return plots_b64, plot_names, txt

# =============================================================================
#  TCP SERVER
# =============================================================================

def _encode_msg(obj):
    payload=json.dumps(obj,separators=(",",":")).encode("utf-8")
    return struct.pack(">I",len(payload))+payload


class TCPServer:
    def __init__(self,host=TCP_HOST,port=TCP_PORT):
        self._srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self._srv.bind((host,port)); self._srv.listen(1); self._srv.settimeout(1.0)
        self._conn=None; self._lock=threading.Lock(); self._stop=threading.Event()
        self._stream_dec=0; self._send_q: deque=deque(maxlen=400)
        log.info(f"TCP server listening  {host}:{port}")
        threading.Thread(target=self._accept_loop,daemon=True,name="TCPAccept").start()
        threading.Thread(target=self._send_loop,daemon=True,name="TCPSend").start()

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn,addr=self._srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                log.info(f"Laptop connected  {addr[0]}:{addr[1]}")
                with self._lock:
                    if self._conn:
                        try: self._conn.close()
                        except: pass
                    self._conn=conn
            except socket.timeout: continue
            except Exception as e:
                if not self._stop.is_set(): log.warning(f"Accept error: {e}")

    def _send_loop(self):
        while not self._stop.is_set():
            with self._lock:
                conn=self._conn; msgs=[]
                while self._send_q: msgs.append(self._send_q.popleft())
            if conn and msgs:
                try: conn.sendall(b"".join(msgs))
                except Exception as e:
                    log.warning(f"Send error: {e}")
                    with self._lock: self._conn=None
            else: time.sleep(0.005)

    def send_stream(self,t,v):
        self._stream_dec+=1
        if self._stream_dec%STREAM_DECIMATE!=0: return
        with self._lock:
            if self._conn is None: return
            self._send_q.append(_encode_msg({"type":"stream","t":t,"v":v}))

    def send_result(self,obj):
        with self._lock:
            if self._conn is None: return
            self._send_q.append(_encode_msg(obj))

    def stop(self):
        self._stop.set()
        try: self._srv.close()
        except: pass

# =============================================================================
#  MAIN LOOP
# =============================================================================

def run(source, tcp: TCPServer, session_folder: Path, csv_log: CSVLogger):
    buf_size=int(BUFFER_SECONDS*FS)
    t_buf: deque=deque(maxlen=buf_size); x_buf: deque=deque(maxlen=buf_size)
    loop_idx=0; t_last=time.monotonic()
    log.info("Analysis loop started — waiting for samples...")

    while True:
        for tv,av in source.read():
            t_buf.append(tv); x_buf.append(av); tcp.send_stream(tv,av)

        now=time.monotonic()
        if (now-t_last)<LOOP_INTERVAL_SEC: time.sleep(0.015); continue
        t_last=now; loop_idx+=1

        if len(t_buf)<int(3.0*FS):
            pct=100*len(t_buf)/buf_size
            log.info(f"Loop {loop_idx:04d}  buffering {pct:.0f}%"); continue

        t_arr=np.array(t_buf); raw=np.array(x_buf)

        try:    filtered=notch_filter(bandpass_filter(raw,FS),FS)
        except Exception as e: log.warning(f"Filter error: {e}"); filtered=raw.copy()

        snr_wb  = compute_snr_wideband(raw, filtered)
        # If flat DC detected (nan), log a clear warning immediately
        if snr_wb is not None and np.isnan(snr_wb):
            log.warning(f"Loop {loop_idx:04d}  FLAT DC SIGNAL — no ECG detected. "
                        f"Check: ADS_GAIN={ADS_GAIN} range=+-{4.096/ADS_GAIN:.3f}V, "
                        f"VREF_MID_VOLTS={VREF_MID_VOLTS}, electrode connections.")
        r_peaks = detect_r_peaks(filtered,FS)
        snr_qrs = compute_snr_qrs(filtered,r_peaks,FS)
        feats   = extract_features(filtered,r_peaks,t_arr,FS) if len(r_peaks)>=2 else {}
        feats["SNR_WB"]=snr_wb
        feats["SNR_QRS"]=snr_qrs if _nn(snr_qrs) else None
        abn=detect_abnormalities(feats,snr_wb,snr_qrs)

        # CSV
        hr_val=feats.get("HR",None)
        csv_log.write(t_arr,raw,filtered,r_peaks,loop_idx,hr_val,snr_wb)

        hr_s  =f"{feats.get('HR',np.nan):.1f}"      if _nn(feats.get('HR'))     else "---"
        st_s  =f"{feats.get('ST_dev',np.nan):+.3f}" if _nn(feats.get('ST_dev')) else "---"
        qtc_s =f"{feats.get('QTc',np.nan):.3f}"     if _nn(feats.get('QTc'))    else "---"
        qrs_s =f"{snr_qrs:.1f}"                      if _nn(snr_qrs)            else "---"
        wb_log=("DC/FLAT" if (snr_wb is not None and np.isnan(snr_wb))
                else f"{snr_wb:.1f}dB" if snr_wb is not None else "---")
        flag  ="  [ABN] "+" | ".join(abn[:2]) if abn else "  [OK]"
        log.info(f"Loop {loop_idx:04d}  HR:{hr_s}  ST:{st_s}  QTc:{qtc_s}"
                 f"  WB-SNR:{wb_log}  QRS-SNR:{qrs_s}dB  Rpeaks:{len(r_peaks)}{flag}")

        do_save=(loop_idx%SAVE_EVERY_N_LOOPS==0)
        if do_save:
            lf=create_loop_folder(session_folder,loop_idx)
            try:
                plots_b64,plot_names,txt=build_and_save_plots(
                    t_arr,raw,filtered,r_peaks,feats,abn,
                    loop_idx,FS,snr_wb,snr_qrs,lf)
            except Exception as e:
                log.error(f"Plot/save error: {e}")
                import traceback; traceback.print_exc()
                plots_b64,plot_names,txt={},[],build_summary_text(feats,abn,loop_idx,snr_wb,snr_qrs)
        else:
            txt=build_summary_text(feats,abn,loop_idx,snr_wb,snr_qrs)
            plots_b64={}; plot_names=[]

        clean={k:(None if (isinstance(v,float) and np.isnan(v)) else v)
               for k,v in feats.items() if not k.startswith("_")}
        tcp.send_result({
            "type":"result","loop":loop_idx,"ts":time.strftime("%H:%M:%S"),
            "feats":clean,"abn":abn,"plots":plots_b64,"summary_txt":txt,
            "plot_names":plot_names,"snr_wb":snr_wb,"snr_qrs":snr_qrs,
            "n_rpeaks":len(r_peaks),
        })

# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    print("\n"+"="*64)
    print("  ECG SYSTEM  --  RASPBERRY PI  (TCP SERVER)")
    print("="*64)
    print(f"  Source   :  {DATA_SOURCE}")
    print(f"  FS       :  {FS} Hz  (stream {FS//STREAM_DECIMATE} Hz to laptop)")
    print(f"  Gain     :  {ADS_GAIN}  (±{4.096/ADS_GAIN:.4f}V range)")
    print(f"  Buffer   :  {BUFFER_SECONDS}s   Analysis every {LOOP_INTERVAL_SEC}s")
    print(f"  Save     :  every {SAVE_EVERY_N_LOOPS} loops")
    print(f"  Port     :  {TCP_PORT}")
    print(f"  Output   :  {OUTPUT_BASE_DIR}")
    print("  Ctrl+C to stop.\n")

    out=Path(OUTPUT_BASE_DIR); out.mkdir(parents=True,exist_ok=True)
    _t=out/"_write_test.tmp"
    try:
        _t.write_text("ok"); _t.unlink()
        log.info(f"Output directory OK: {out}")
    except Exception as e:
        log.error(f"Cannot write to {out}: {e}"); sys.exit(1)

    session_folder=create_session_folder()

    if DATA_SOURCE=="ads1115":
        try:   source=ADS1115Source(); log.info("ADS1115 source ready")
        except Exception as e:
            log.error(f"ADS1115 init failed: {e}")
            log.warning("Falling back to DEMO mode")
            source=DemoSource()
    else:
        log.info("DEMO mode"); source=DemoSource()

    tcp=TCPServer()
    csv_log=CSVLogger(session_folder)

    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80)); ip=s.getsockname()[0]; s.close()
    except: ip="<pi-ip>"
    print(f"\n  On your laptop run:")
    print(f"  python3 ecg_laptop_client_final.py  {ip}\n")

    def _shutdown(sig_num,frame):
        log.info("Shutting down...")
        source.stop(); tcp.stop(); csv_log.close()
        print(f"\n  Results saved in:\n  {session_folder}\n"); sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    run(source,tcp,session_folder,csv_log)


if __name__=="__main__":
    main()