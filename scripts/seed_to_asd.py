#!/usr/bin/env python3
"""
seed_to_asd.py

Turn a raw SEED (or miniSEED) volume from the titanEA into calibrated
acceleration ASD curve(s), written in the same single-column,
no-header CSV convention used by the rest of the pipeline
(f_151.csv / s151_accel_ch1.csv, f_pcb.csv / PCB_accel_ch1.csv, etc.)
so they can be dropped straight into vibration_response.py or
rigid_coupling_floor.py.

Since we don't yet know exactly what's inside the file (how many
channels, whether the response is really embedded, sample rates,
gaps, etc.), the script always runs a discovery pass first and prints
everything it can find *before* doing any signal processing. Nothing
is assumed silently.

Pipeline:
  1. Read the SEED volume (waveform + whatever station/response info
     is embedded) and print a full inventory of what's in it.
  2. For each channel, one at a time: detrend, remove instrument
     response -> acceleration, Welch-average the ASD over the full
     trace, write it out, then release that channel's memory before
     moving to the next (so peak RAM tracks one channel, not all of
     them at once).
  3. Write f_<label>.csv and <label>_asd.csv per channel.

Requires: obspy, numpy, scipy
"""

from __future__ import annotations
from pathlib import Path
import ctypes
import gc
import platform
import resource

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, read_inventory, Stream, UTCDateTime
from scipy.signal import welch

# ==================== USER CONFIG ====================

SEED_PATH = Path("../raw_seed/S0001_titanEA-Slave_1977_20260808_153704.seed")   # <-- point this at your file
OUT_DIR   = Path("../data")                          # matches existing DATA_DIR convention
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR  = Path("../plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Per-channel StationXML response files. The SEED file itself is plain
# miniSEED (waveform only, no dataless/station blockettes) -- these
# were supplied separately, one per channel, and get merged into a
# single Inventory below.
RESPONSE_DIR = Path("../response")
RESPONSE_FILES = [
    "XX.S0001.HNX.xml",
    "XX.S0001.HNY.xml",
    "XX.S0001.HNZ.xml",
]

# The exported StationXML epochs are stamped with the export date
# (2026-08-18), which postdates this recording (2026-08-08). User has
# confirmed the unit was power-cycled and physically moved (but not
# reconfigured) between the recording and the export, and that the
# electronic calibration (sensitivity/poles-zeros/gain) is expected to
# survive a power cycle -- so the epoch alone gets backdated to cover
# the data. Physical orientation is a separate
# (see CHANNEL_LABELS) and is NOT assumed unchanged here.
BACKDATE_RESPONSE_EPOCH = True

# Once you've run this once and seen the discovery printout below,
# fill this in to map SEED channel IDs -> friendly labels that match
# your existing up_down / side_to_side naming convention. Anything not
# listed here just falls back to a sanitized version of the channel id,
# so the script works before you've mapped anything.
CHANNEL_LABELS = {
    # "XX.STA..HNZ": "titanEA_up_down",
    # "XX.STA..HNN": "titanEA_side_to_side_1",
    # "XX.STA..HNE": "titanEA_side_to_side_2",
}

OUTPUT_UNIT = "ACC"   # 'ACC' -> m/s^2, matches vibration_response.py convention

# Welch PSD settings. WINDOW_SEC sets your lowest resolvable frequency
# (~1/WINDOW_SEC Hz) and how much averaging/smoothing you get at high
# frequency. 3600 s -> ~2.8e-4 Hz resolution, decent averaging over a
# 24 hr file (~46 windows at 50% overlap).
WINDOW_SEC   = 3600.0
OVERLAP_FRAC = 0.5

# Pre-filter corners (f1, f2, f3, f4) in Hz applied before deconvolution
# to avoid blowing up noise outside the sensor's calibrated band.
# None -> ObsPy derives sensible defaults from the response itself;
# set explicitly once you know titanEA's passband (e.g. (0.001, 0.005, 40, 45)).
PRE_FILT = None
WATER_LEVEL = 60

# =======================================================


# -------------------- memory helpers --------------------

def release_memory() -> None:
    """Force a GC pass and, on Linux, ask glibc to actually hand freed
    heap back to the OS (gc.collect() alone frees Python objects but
    doesn't guarantee the C allocator returns that memory to the OS)."""
    gc.collect()
    if platform.system() == "Linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def report_peak_memory(note: str = "") -> None:
    """ru_maxrss is *peak* RSS so far (KB on Linux), monotonically
    increasing -- printing it after each channel lets you see how high
    memory climbed, even though it can't show it going back down."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  [mem] peak RSS so far: {kb / 1e6:.2f} GB{'  (' + note + ')' if note else ''}")


# -------------------- discovery --------------------

def load_stream(seed_path: Path) -> Stream:
    st = read(str(seed_path))
    st.merge(method=1, fill_value="interpolate")
    return st


def load_inventory(response_dir: Path, response_files: list[str],
                    epoch_start: "UTCDateTime | None" = None):
    """
    The SEED file has no embedded dataless/station blockettes (plain
    miniSEED), so response metadata comes from separate per-channel
    StationXML files instead. Each file covers one channel of the same
    station (XX.S0001); read them individually with read_inventory()
    and merge into a single Inventory covering all channels.

    If epoch_start is given, the network/station/channel start_date on
    every loaded epoch is backdated to that value when it postdates it,
    so response lookups against earlier-recorded data still match. Only
    call this with epoch_start set once the user has confirmed the
    calibration itself (not just the epoch timestamp) is valid back to
    that time -- see BACKDATE_RESPONSE_EPOCH.
    """
    inv = None
    for fname in response_files:
        path = response_dir / fname
        print(f"[info] reading response metadata from {path}")
        single = read_inventory(str(path))
        if epoch_start is not None:
            for net in single:
                if net.start_date is not None and net.start_date > epoch_start:
                    net.start_date = epoch_start
                for sta in net:
                    if sta.start_date is not None and sta.start_date > epoch_start:
                        sta.start_date = epoch_start
                    for cha in sta:
                        if cha.start_date is not None and cha.start_date > epoch_start:
                            print(f"  [WARN] backdating epoch start for "
                                  f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}: "
                                  f"{cha.start_date} -> {epoch_start} (exported epoch "
                                  f"postdates the recording; confirmed by user that "
                                  f"sensitivity/PZ/gain survive the power cycle in "
                                  f"between -- orientation is NOT assumed unchanged)")
                            cha.start_date = epoch_start
        inv = single if inv is None else inv + single
    if inv is None or len(inv) == 0:
        raise RuntimeError(f"No response metadata found in {response_dir}")
    return inv


def describe_seed_contents(st: Stream, inv) -> None:
    """Print everything we can determine about the file's contents
    before doing any processing, since the exact channel layout isn't
    known ahead of time."""

    print("\n" + "=" * 60)
    print("SEED CONTENTS - DISCOVERY PASS")
    print("=" * 60)

    print(f"\n[Waveform traces]  ({len(st)} total)")
    for tr in st:
        dur_hr = (tr.stats.endtime - tr.stats.starttime) / 3600.0
        print(f"  {tr.id:<20s} fs={tr.stats.sampling_rate:g} Hz  "
              f"npts={tr.stats.npts:>10d}  dtype={tr.data.dtype}  "
              f"calib={getattr(tr.stats, 'calib', 1.0)}  "
              f"duration={dur_hr:.2f} hr")
        print(f"      start={tr.stats.starttime}  end={tr.stats.endtime}")

    gaps = st.get_gaps()
    print(f"\n[Gaps / overlaps]  ({len(gaps)} found)")
    for g in gaps[:20]:
        net, sta, loc, cha, t1, t2, dur, samples = g
        kind = "GAP" if dur > 0 else "OVERLAP"
        print(f"  {kind}  {net}.{sta}.{loc}.{cha}  {t1} -> {t2}  ({dur:.3f} s)")
    if len(gaps) > 20:
        print(f"  ... and {len(gaps) - 20} more")

    print("\n[Inventory contents]")
    if len(inv) == 0:
        print("  (empty -- no station/response metadata recovered)")
    else:
        try:
            contents = inv.get_contents()
            for key, vals in contents.items():
                print(f"  {key}: {vals}")
        except Exception as e:
            print(f"  [could not summarize inventory: {e}]")

        print("\n[Per-channel response summary]")
        for net in inv:
            for sta in net:
                for cha in sta:
                    chan_id = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                    print(f"  {chan_id}")
                    print(f"      sample_rate={cha.sample_rate} Hz  "
                          f"start={cha.start_date}  end={cha.end_date}")
                    sensor_desc = None
                    try:
                        sensor_desc = cha.sensor.description or cha.sensor.type
                    except Exception:
                        pass
                    if sensor_desc:
                        print(f"      sensor: {sensor_desc}")
                    if cha.response is not None:
                        try:
                            sens = cha.response.instrument_sensitivity
                            print(f"      overall sensitivity: {sens.value:.6g} "
                                  f"counts per {sens.input_units} at {sens.frequency} Hz")
                        except Exception:
                            print("      [no overall instrument sensitivity found]")
                        n_stages = len(cha.response.response_stages)
                        print(f"      {n_stages} response stage(s)")
                        if n_stages:
                            first, last = cha.response.response_stages[0], cha.response.response_stages[-1]
                            print(f"      stage 1 input units:  {getattr(first, 'input_units', '?')}")
                            print(f"      stage {n_stages} output units: {getattr(last, 'output_units', '?')}")
                    else:
                        print("      [no response attached to this channel]")

    print("=" * 60 + "\n")


# -------------------- processing --------------------

def remove_response_safe(st: Stream, inv) -> Stream:
    st.detrend("linear")
    st.remove_response(
        inventory=inv,
        output=OUTPUT_UNIT,
        water_level=WATER_LEVEL,
        pre_filt=PRE_FILT,
        zero_mean=True,
        taper=True,
        taper_fraction=0.02,
    )
    return st


def welch_asd(tr, window_sec: float = WINDOW_SEC, overlap_frac: float = OVERLAP_FRAC):
    """
    Average Welch periodograms over the full trace -> a single, smooth
    ASD curve (units/sqrt(Hz)) matching the shape of f_151.csv /
    s151_accel_ch1.csv, ready for splice_spectra() / asd_to_rms().
    """
    fs = tr.stats.sampling_rate
    nperseg = int(window_sec * fs)
    nperseg = max(min(nperseg, tr.stats.npts), 256)
    noverlap = int(nperseg * overlap_frac)

    f, pxx = welch(
        tr.data, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        detrend="linear", scaling="density",
    )
    asd = np.sqrt(pxx)

    mask = f > 0  # drop DC bin, undefined on log-log axes
    return f[mask], asd[mask]


def write_single_col(path: Path, arr: np.ndarray, header: str = "") -> None:
    np.savetxt(path, arr, delimiter=",", header=header, comments="# ")


def plot_asd(f: np.ndarray, asd: np.ndarray, label: str, calibrated: bool,
             unit: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(f, asd)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"ASD ({unit}/sqrt(Hz))")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    if calibrated:
        ax.set_title(f"{label} - calibrated ASD")
    else:
        ax.set_title(f"{label} - RAW COUNTS, UNCALIBRATED")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print(f"Reading {SEED_PATH} ...")
    st = load_stream(SEED_PATH)
    report_peak_memory("after read")

    print("Loading instrument response from StationXML files ...")
    epoch_start = min(tr.stats.starttime for tr in st) if BACKDATE_RESPONSE_EPOCH else None
    inv = load_inventory(RESPONSE_DIR, RESPONSE_FILES, epoch_start=epoch_start)

    describe_seed_contents(st, inv)

    n_channels = len(st)
    try:
        # Iterate over a snapshot of the trace list, since we remove
        # traces from `st` as we finish with them.
        for i, tr in enumerate(list(st.traces), start=1):
            label = CHANNEL_LABELS.get(tr.id, tr.id.replace(".", "_"))
            print(f"\n[{i}/{n_channels}] Processing {tr.id} -> label '{label}'")

            sub = Stream([tr])
            try:
                remove_response_safe(sub, inv)
                calibrated = True
                unit = "m/s^2" if OUTPUT_UNIT == "ACC" else OUTPUT_UNIT
            except Exception as e:
                # Stage 1 fallback: no matching response for this channel
                # (shouldn't happen now that all 3 StationXML files are
                # loaded, but kept as a safety net per the hard rule that
                # we never silently present raw counts as calibrated data).
                print(f"  [WARN] response removal failed for {tr.id}: {e}")
                print("  Falling back to raw-counts ASD (UNCALIBRATED) for this channel.")
                sub.detrend("linear")
                calibrated = False
                unit = "counts"
                label = f"{label}_RAW_UNCALIBRATED"

            f, asd = welch_asd(tr)

            f_path = OUT_DIR / f"f_{label}.csv"
            a_path = OUT_DIR / f"{label}_asd.csv"
            header = (f"ASD, {unit}/sqrt(Hz)" if calibrated
                       else f"RAW COUNTS/sqrt(Hz), UNCALIBRATED - {unit}/sqrt(Hz)")
            write_single_col(f_path, f, header="frequency, Hz")
            write_single_col(a_path, asd, header=header)

            plot_path = PLOT_DIR / f"{label}_asd.png"
            plot_asd(f, asd, label, calibrated, unit, plot_path)

            print(f"  wrote {f_path}   ({len(f)} pts, f=[{f.min():.3g}, {f.max():.3g}] Hz)")
            tag = "" if calibrated else " [UNCALIBRATED]"
            print(f"  wrote {a_path}   median ASD = {np.median(asd):.3e} {unit}/sqrt(Hz){tag}")
            print(f"  wrote {plot_path}")

            # Release this channel's data before moving to the next one,
            # rather than holding every channel's array in memory at once.
            st.remove(tr)
            del tr, sub, f, asd
            release_memory()
            report_peak_memory(f"after channel {i}/{n_channels}")

    finally:
        # Final cleanup regardless of success/failure above.
        del st, inv
        release_memory()

    print("\nDone. These f_*.csv / *_asd.csv pairs are drop-in compatible with "
          "read_single_column_csv() and splice_spectra() in tf_tools.py.")


if __name__ == "__main__":
    main()