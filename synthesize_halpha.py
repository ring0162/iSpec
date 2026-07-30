#!/usr/bin/env python3
"""
synthesize_halpha.py
====================
Synthesize a model photospheric H-alpha line profile using iSpec and the
best-fit stellar parameters from a previous fit_harps_spectrum.py or
fit_espresso_spectrum.py run.

Scientific context
------------------
The synthetic H-alpha profile represents the purely photospheric contribution
with no chromospheric emission.  For a young active star, the observed H-alpha
core is partially filled in by chromospheric emission.  Comparing the observed
profile to this model gives:

    ΔEW = EW_observed − EW_model

which is the net chromospheric emission equivalent width — the standard
activity diagnostic (analogous to the Lλ/Lbol and log R'HK indices).

H-alpha in iSpec
----------------
H-alpha (6562.797 Å) is handled via a dedicated Hlinedata file internal to
each synthesis backend (turbospectrum, SPECTRUM, MOOG).  It is NOT listed in
the user atomic linelist (.tsv).  No special flags are needed; just set the
waveobs range to cover the line and call generate_spectrum() normally.

The synthesis window should be wide (±8 nm minimum) to capture the
Lorentzian pressure-broadened wings, which extend several nm either side of
line centre even in cool K-type stars.

Usage
-----
  # Minimal: reads stellar_params.json from a previous iSpec run
  python synthesize_halpha.py --results-dir ispec_results/

  # Overlay on observed spectrum
  python synthesize_halpha.py --results-dir ispec_results/ \\
      --observed ispec_results/observed_spectrum_normalized.fits

  # Override individual parameters (useful for exploring sensitivity)
  python synthesize_halpha.py --results-dir ispec_results/ \\
      --teff 4500 --logg 4.4 --mh 0.0

  # Specify parameters directly without a results directory
  python synthesize_halpha.py \\
      --teff 4400 --logg 4.3 --mh 0.0 --vmic 1.1 --vmac 2.5 --vsini 10.0 \\
      --resolution 115000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# iSpec bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ISPEC_DIR  = SCRIPT_DIR

sys.path.insert(0, str(ISPEC_DIR))

try:
    import ispec
except ImportError as e:
    sys.exit(
        f"ERROR: Cannot import ispec from {ISPEC_DIR}.\n"
        f"Details: {e}"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# H-alpha constants
# ---------------------------------------------------------------------------
HALPHA_NM   = 656.279          # vacuum rest wavelength (nm)
HALPHA_WING = 8.0              # half-width of synthesis window (nm)
WAVE_STEP   = 0.001            # synthesis step (nm) — ~0.46 km/s per pixel at Hα

ISPEC_INPUT = ISPEC_DIR / "input"


# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------

def load_params(results_dir: Path | None,
                cli_overrides: dict) -> dict:
    """
    Build the parameter dict from a stellar_params.json file, then apply
    any values explicitly given on the command line.

    Falls back to sensible defaults (quiet K dwarf) when no JSON is provided.

    Returns
    -------
    dict with keys: teff, logg, MH, alpha, vmic, vmac, vsini, resolution
    """
    defaults = {
        "teff": 5777.0, "logg": 4.44, "MH": 0.0, "alpha": 0.0,
        "vmic": 1.0,  "vmac": 3.0,  "vsini": 2.0,
        "resolution": 115_000,
    }

    params = dict(defaults)

    if results_dir is not None:
        json_path = results_dir / "stellar_params.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            fitted = data.get("params", {})
            for k in ("teff", "logg", "MH", "alpha", "vmic", "vmac", "vsini"):
                if k in fitted:
                    params[k] = float(fitted[k])
            log.info("Loaded parameters from %s", json_path)
            log.info(
                "  Teff=%.1f K  logg=%.3f  [M/H]=%.3f  "
                "vmic=%.2f  vmac=%.2f  vsini=%.2f km/s",
                params["teff"], params["logg"], params["MH"],
                params["vmic"], params["vmac"], params["vsini"],
            )
        else:
            log.warning("stellar_params.json not found in %s; using defaults.", results_dir)

    # CLI overrides
    for key, val in cli_overrides.items():
        if val is not None:
            params[key] = float(val)
            log.info("CLI override: %s = %s", key, val)

    # alpha from MH if not provided explicitly
    if cli_overrides.get("alpha") is None and results_dir is None:
        params["alpha"] = float(ispec.determine_abundance_enchancements(params["MH"]))

    return params


# ---------------------------------------------------------------------------
# iSpec data loading
# ---------------------------------------------------------------------------

def load_ispec_data(atmosphere_name: str = "MARCS.GES",
                    wave_min: float = HALPHA_NM - HALPHA_WING - 2.0,
                    wave_max: float = HALPHA_NM + HALPHA_WING + 2.0):
    """
    Load model atmospheres, linelist, solar abundances, and isotopes.

    The linelist only needs to cover the H-alpha window for nearby metal
    lines.  H-alpha itself is supplied internally by the synthesis backend
    via DATA/Hlinedata.
    """
    # --- Atmosphere ---------------------------------------------------------
    model_dir = ISPEC_INPUT / "atmospheres" / atmosphere_name
    if not model_dir.exists():
        for fallback in ("MARCS.GES", "MARCS"):
            candidate = ISPEC_INPUT / "atmospheres" / fallback
            if candidate.exists():
                log.warning("Atmosphere '%s' not found; using %s.", atmosphere_name, fallback)
                model_dir = candidate
                break
        else:
            raise FileNotFoundError(
                f"No atmosphere grid found under {ISPEC_INPUT / 'atmospheres'}.\n"
                "Download iSpec input data from https://www.blancocuaresma.com/s/iSpec"
            )

    log.info("Loading atmosphere grid from %s …", model_dir)
    modeled_layers_pack = ispec.load_modeled_layers_pack(str(model_dir))
    is_atlas = "ATLAS" in atmosphere_name.upper()

    # --- Solar abundances ---------------------------------------------------
    abund_name = "Grevesse.1998" if is_atlas else "Grevesse.2007"
    abund_file = ISPEC_INPUT / "abundances" / abund_name / "stdatom.dat"
    if not abund_file.exists():
        for sub in sorted((ISPEC_INPUT / "abundances").iterdir()):
            candidate = sub / "stdatom.dat"
            if candidate.exists():
                abund_file = candidate
                break
    solar_abundances = ispec.read_solar_abundances(str(abund_file))
    log.info("Solar abundances: %s", abund_file.parent.name)

    # --- Isotopes -----------------------------------------------------------
    isotope_file = ISPEC_INPUT / "isotopes" / "SPECTRUM.lst"
    isotopes = ispec.read_isotope_data(str(isotope_file))

    # --- Atomic linelist (for nearby metal lines) ---------------------------
    linelist_candidates = [
        ISPEC_INPUT / "linelists/transitions/GESv6_atom_hfs_iso.420_920nm/atomic_lines.tsv",
        ISPEC_INPUT / "linelists/transitions/VALD.300_1100nm/atomic_lines.tsv",
    ]
    linelist_file = next((p for p in linelist_candidates if p.exists()), None)
    if linelist_file is None:
        raise FileNotFoundError(
            "No atomic linelist found.  Looked for GESv6 and VALD under "
            f"{ISPEC_INPUT / 'linelists/transitions'}"
        )

    log.info("Loading linelist from %s (%.1f–%.1f nm) …", linelist_file.name, wave_min, wave_max)
    atomic_linelist = ispec.read_atomic_linelist(
        str(linelist_file), wave_base=wave_min, wave_top=wave_max
    )
    log.info("  %d metal lines in the H-alpha window.", len(atomic_linelist))

    return modeled_layers_pack, atomic_linelist, solar_abundances, isotopes


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize_halpha(params: dict,
                      modeled_layers_pack,
                      atomic_linelist,
                      solar_abundances,
                      isotopes,
                      *,
                      code: str = "spectrum",
                      wave_min: float = HALPHA_NM - HALPHA_WING,
                      wave_max: float = HALPHA_NM + HALPHA_WING,
                      wave_step: float = WAVE_STEP) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a spectrum over the H-alpha region.

    Parameters
    ----------
    params : dict
        Keys: teff, logg, MH, alpha, vmic, vmac, vsini, resolution.
    code : str
        Radiative transfer code ('spectrum', 'turbospectrum', 'moog').
    wave_min, wave_max : float
        Synthesis wavelength range (nm).  Default ±8 nm around Hα centre.
    wave_step : float
        Wavelength sampling (nm).

    Returns
    -------
    waveobs : ndarray  (nm)
    flux    : ndarray  (continuum-normalised, 0–1)
    """
    teff  = params["teff"]
    logg  = params["logg"]
    MH    = params["MH"]
    alpha = params["alpha"]
    vmic  = params["vmic"]
    vmac  = params["vmac"]
    vsini = params["vsini"]
    R     = int(params["resolution"])

    log.info(
        "Synthesising H-alpha:  Teff=%.0f K  logg=%.2f  [M/H]=%.2f  "
        "vmic=%.2f  vmac=%.2f  vsini=%.2f km/s  R=%d  code=%s",
        teff, logg, MH, vmic, vmac, vsini, R, code,
    )

    if not ispec.valid_atmosphere_target(
        modeled_layers_pack, {"teff": teff, "logg": logg, "MH": MH, "alpha": alpha}
    ):
        raise ValueError(
            f"Parameters (Teff={teff}, logg={logg}, [M/H]={MH}) are outside "
            "the atmosphere grid bounds."
        )

    atmosphere_layers = ispec.interpolate_atmosphere_layers(
        modeled_layers_pack,
        {"teff": teff, "logg": logg, "MH": MH, "alpha": alpha},
        code=code,
    )

    waveobs = np.arange(wave_min, wave_max + wave_step, wave_step)

    log.info("Calling generate_spectrum over %.3f–%.3f nm (%d pixels) …",
             wave_min, wave_max, len(waveobs))

    flux = ispec.generate_spectrum(
        waveobs,
        atmosphere_layers,
        teff, logg, MH, alpha,
        atomic_linelist, isotopes, solar_abundances,
        fixed_abundances=None,
        microturbulence_vel=vmic,
        macroturbulence=vmac,
        vsini=vsini,
        limb_darkening_coeff=0.6,
        R=R,
        regions=None,
        verbose=0,
        code=code,
    )

    log.info("Synthesis complete.")
    return waveobs, flux


# ---------------------------------------------------------------------------
# Line inventory in the synthesis window
# ---------------------------------------------------------------------------

def report_lines_in_window(waveobs: np.ndarray, flux: np.ndarray,
                           atomic_linelist,
                           win_min: float = 655.7,
                           win_max: float = 656.8,
                           depth_threshold: float = 0.05,
                           measure_window_nm: float = 0.05,
                           out_file: Path | None = None) -> None:
    """
    Print (and optionally write) a table of every line in [win_min, win_max]
    whose synthesised depth exceeds depth_threshold.

    H-alpha (H I 656.279 nm) is injected explicitly because it lives in the
    internal Hlinedata file rather than the user atomic linelist.

    Parameters
    ----------
    waveobs, flux     : synthesised spectrum arrays (nm, normalised)
    atomic_linelist   : iSpec recarray from read_atomic_linelist
    win_min, win_max  : wavelength window to inspect (nm)
    depth_threshold   : minimum depth (1 − F_min) to include in the table
    measure_window_nm : half-width of the window used to find F_min per line (nm)
    out_file          : if given, the table is appended to this file as well
    """
    # --- Collect candidate lines from the atomic linelist -------------------
    mask = (atomic_linelist["wave_nm"] >= win_min) & \
           (atomic_linelist["wave_nm"] <= win_max)
    candidates = atomic_linelist[mask]

    rows = []
    for line in candidates:
        wl            = float(line["wave_nm"])
        species       = str(line["element"]).strip()
        theo_depth    = float(line["theoretical_depth"])

        # Require the solar theoretical depth to pass the threshold first.
        # This prevents weak lines near the H-alpha core from inheriting
        # H-alpha's measured depth due to an overlapping measurement window.
        if theo_depth < depth_threshold:
            continue

        w_lo = wl - measure_window_nm
        w_hi = wl + measure_window_nm
        pix  = (waveobs >= w_lo) & (waveobs <= w_hi)
        if pix.sum() < 2:
            continue
        depth = float(1.0 - np.min(flux[pix]))
        if depth >= depth_threshold:
            rows.append((species, wl, depth, theo_depth))

    # --- Inject H-alpha explicitly (not in the atomic .tsv linelist) --------
    halpha_wl = HALPHA_NM
    if win_min <= halpha_wl <= win_max:
        pix = (waveobs >= halpha_wl - measure_window_nm) & \
              (waveobs <= halpha_wl + measure_window_nm)
        if pix.sum() >= 2:
            halpha_depth = float(1.0 - np.min(flux[pix]))
            # Remove any atomic-linelist entry that accidentally matched Hα
            rows = [(s, w, d, td) for s, w, d, td in rows if abs(w - halpha_wl) > 0.01]
            if halpha_depth >= depth_threshold:
                rows.append(("H I", halpha_wl, halpha_depth, float("nan")))

    # --- Sort by wavelength -------------------------------------------------
    rows.sort(key=lambda r: r[1])

    # --- Format table -------------------------------------------------------
    header    = f"\n  Lines with depth > {depth_threshold:.2f} in {win_min:.3f}–{win_max:.3f} nm\n"
    separator = "  " + "─" * 56
    col_hdr   = f"  {'Species':<10}  {'Wave (nm)':>10}  {'Meas. depth':>12}  {'Theo. depth':>12}"
    divider   = "  " + "-" * 56

    lines_out = [header, separator, col_hdr, divider]
    if rows:
        for species, wl, depth, theo_depth in rows:
            theo_str = f"{theo_depth:>12.4f}" if np.isfinite(theo_depth) else f"{'—':>12}"
            lines_out.append(f"  {species:<10}  {wl:>10.4f}  {depth:>12.4f}  {theo_str}")
    else:
        lines_out.append("  (no lines above threshold)")
    lines_out.append(separator + "\n")

    text = "\n".join(lines_out)
    print(text)

    if out_file is not None:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------------------
# Equivalent-width measurement
# ---------------------------------------------------------------------------

def measure_ew(waveobs: np.ndarray, flux: np.ndarray,
               centre_nm: float = HALPHA_NM,
               window_nm: float = 1.5) -> float:
    """
    Measure the equivalent width of a line by direct integration.

        EW = ∫ (1 − F/Fc) dλ

    Parameters
    ----------
    waveobs   : wavelength array (nm)
    flux      : continuum-normalised flux (Fc = 1)
    centre_nm : line centre (nm)
    window_nm : half-width of integration window (nm)

    Returns
    -------
    EW : float, in nm  (positive for absorption)
    """
    mask = (waveobs >= centre_nm - window_nm) & (waveobs <= centre_nm + window_nm)
    if mask.sum() < 3:
        return np.nan
    dλ = np.diff(waveobs[mask])
    integrand = 1.0 - flux[mask]
    # Trapezoidal integration
    ew = float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dλ))
    return ew


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_synthetic_spectrum(waveobs, flux, output_path: Path) -> None:
    """Save the synthetic H-alpha spectrum as an iSpec-format FITS file."""
    synth = ispec.create_spectrum_structure(waveobs, flux)
    ispec.write_spectrum(synth, str(output_path))
    log.info("Synthetic spectrum saved to %s", output_path)


def save_ascii(waveobs, flux, output_path: Path) -> None:
    """Save as a plain two-column ASCII file (nm  flux)."""
    with open(output_path, "w") as f:
        f.write("# waveobs_nm\tflux\n")
        for w, fl in zip(waveobs, flux):
            f.write(f"{w:.6f}\t{fl:.8f}\n")
    log.info("ASCII spectrum saved to %s", output_path)


# ---------------------------------------------------------------------------
# Optional: load observed spectrum for comparison
# ---------------------------------------------------------------------------

def load_observed(obs_path: str,
                  centre_nm: float = HALPHA_NM,
                  window_nm: float = HALPHA_WING) -> tuple | None:
    """
    Load an observed spectrum and trim to the H-alpha region.

    Accepts any format iSpec can read: FITS (WCS or BinTable) or text.

    Returns
    -------
    (waveobs, flux) or None if loading fails.
    """
    try:
        spec = ispec.read_spectrum(obs_path)
        mask = (spec["waveobs"] >= centre_nm - window_nm) & \
               (spec["waveobs"] <= centre_nm + window_nm)
        spec = spec[mask]
        if len(spec) == 0:
            log.warning(
                "Observed spectrum has no pixels in the H-alpha window "
                "(%.1f–%.1f nm). Is the spectrum already rest-frame and normalised?",
                centre_nm - window_nm, centre_nm + window_nm,
            )
            return None
        log.info(
            "Observed spectrum: %d pixels in %.3f–%.3f nm window.",
            len(spec), float(spec["waveobs"].min()), float(spec["waveobs"].max()),
        )
        return spec["waveobs"], spec["flux"]
    except Exception as exc:
        log.warning("Could not read observed spectrum (%s): %s", obs_path, exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Sources of stellar parameters ------------------------------------
    p.add_argument(
        "--results-dir", "-r", type=Path, default=None,
        metavar="DIR",
        help="Directory containing stellar_params.json from a previous fit run. "
             "Parameters can be overridden individually below."
    )

    # ---- Stellar parameter overrides --------------------------------------
    pg = p.add_argument_group("Stellar parameter overrides",
                               "Any of these override the value loaded from stellar_params.json.")
    pg.add_argument("--teff",       type=float, default=None, help="Effective temperature (K)")
    pg.add_argument("--logg",       type=float, default=None, help="Surface gravity log g (dex)")
    pg.add_argument("--mh",         type=float, default=None, help="Metallicity [M/H] (dex)")
    pg.add_argument("--alpha",      type=float, default=None, help="Alpha enhancement [alpha/Fe] (dex)")
    pg.add_argument("--vmic",       type=float, default=None, help="Microturbulence (km/s)")
    pg.add_argument("--vmac",       type=float, default=None, help="Macroturbulence (km/s)")
    pg.add_argument("--vsini",      type=float, default=None, help="Projected rotation (km/s)")
    pg.add_argument("--resolution", type=int,   default=None,
                    help="Instrumental resolving power R (default: from JSON, else 115000)")

    # ---- Synthesis options ------------------------------------------------
    p.add_argument(
        "--code", choices=["spectrum", "turbospectrum", "moog"],
        default="spectrum",
        help="Radiative transfer code (default: spectrum)"
    )
    p.add_argument(
        "--atmosphere", default="MARCS.GES",
        help="Atmosphere grid sub-directory under input/atmospheres/ (default: MARCS.GES)"
    )
    p.add_argument(
        "--wing", type=float, default=HALPHA_WING, metavar="NM",
        help=f"Half-width of H-alpha synthesis window in nm (default: {HALPHA_WING}). "
             "Increase if you want to inspect wider pressure-broadened wings."
    )
    p.add_argument(
        "--wave-step", type=float, default=WAVE_STEP, metavar="NM",
        help=f"Synthesis wavelength step in nm (default: {WAVE_STEP})"
    )
    p.add_argument(
        "--ew-window", type=float, default=1.5, metavar="NM",
        help="Half-width of EW integration window in nm (default: 1.5)"
    )

    # ---- Line inventory ---------------------------------------------------
    p.add_argument(
        "--line-win-min", type=float, default=655.7, metavar="NM",
        help="Blue edge of the line-inventory window in nm (default: 655.7)"
    )
    p.add_argument(
        "--line-win-max", type=float, default=656.8, metavar="NM",
        help="Red edge of the line-inventory window in nm (default: 656.8)"
    )
    p.add_argument(
        "--line-depth-min", type=float, default=0.05, metavar="DEPTH",
        help="Minimum synthesised line depth to include in the inventory table (default: 0.05)"
    )

    # ---- Comparison spectrum ----------------------------------------------
    p.add_argument(
        "--observed", "-obs", default=None,
        metavar="FITS_OR_TXT",
        help="Observed spectrum for overlay.  If this is the full observed spectrum "
             "(not pre-trimmed to Hα), it must already be RV-corrected and continuum-"
             "normalised.  Provide ispec_results/observed_spectrum_normalized.fits "
             "from a previous fit run, or any compatible file."
    )

    # ---- Output -----------------------------------------------------------
    p.add_argument(
        "--output-dir", "-o", type=Path, default=Path("halpha_model"),
        help="Output directory (default: halpha_model/)"
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Assemble CLI overrides dict (only non-None values)
    cli_overrides = {
        "teff": args.teff, "logg": args.logg, "MH": args.mh,
        "alpha": args.alpha, "vmic": args.vmic, "vmac": args.vmac,
        "vsini": args.vsini, "resolution": args.resolution,
    }

    # ---- 1. Load parameters ------------------------------------------------
    params = load_params(args.results_dir, cli_overrides)

    wave_min = HALPHA_NM - args.wing
    wave_max = HALPHA_NM + args.wing

    log.info("H-alpha synthesis window: %.3f–%.3f nm", wave_min, wave_max)

    # ---- 2. Set up output directory and log file --------------------------
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_handler = logging.FileHandler(output_dir / "halpha_synth.log",
                                      mode="w", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(log_handler)

    # ---- 3. Check synthesis code availability -----------------------------
    code = args.code
    code_checks = {
        "turbospectrum": ispec.is_turbospectrum_support_enabled,
        "moog":          ispec.is_moog_support_enabled,
        "spectrum":      ispec.is_spectrum_support_enabled,
    }
    if code in code_checks and not code_checks[code]():
        log.warning("Code '%s' not available; falling back to 'spectrum'.", code)
        code = "spectrum"

    # ---- 4. Load iSpec data -----------------------------------------------
    modeled_layers_pack, atomic_linelist, solar_abundances, isotopes = \
        load_ispec_data(
            atmosphere_name=args.atmosphere,
            wave_min=wave_min - 2.0,   # extra margin for metal lines near wings
            wave_max=wave_max + 2.0,
        )

    # ---- 5. Synthesise H-alpha -------------------------------------------
    waveobs, flux = synthesize_halpha(
        params,
        modeled_layers_pack,
        atomic_linelist,
        solar_abundances,
        isotopes,
        code=code,
        wave_min=wave_min,
        wave_max=wave_max,
        wave_step=args.wave_step,
    )

    # ---- 6. Line inventory in the Hα window --------------------------------
    report_lines_in_window(
        waveobs, flux, atomic_linelist,
        win_min=args.line_win_min,
        win_max=args.line_win_max,
        depth_threshold=args.line_depth_min,
        out_file=output_dir / "halpha_synth.log",
    )

    # ---- 7. Measure equivalent width -------------------------------------
    ew_nm = measure_ew(waveobs, flux, centre_nm=HALPHA_NM,
                       window_nm=args.ew_window)
    ew_ang = ew_nm * 10.0   # nm → Å
    log.info(
        "Photospheric H-alpha EW (±%.1f nm window): %.4f nm = %.4f Å",
        args.ew_window, ew_nm, ew_ang,
    )

    # ---- 8. Print results -------------------------------------------------
    separator = "─" * 60
    result_lines = [
        "",
        separator,
        "  Photospheric H-alpha Model — Results",
        separator,
        f"  Teff   = {params['teff']:7.1f} K",
        f"  log g  = {params['logg']:7.3f} dex",
        f"  [M/H]  = {params['MH']:+7.3f} dex",
        f"  vmic   = {params['vmic']:7.2f} km/s",
        f"  vmac   = {params['vmac']:7.2f} km/s",
        f"  vsini  = {params['vsini']:7.2f} km/s",
        f"  R      = {int(params['resolution']):>7d}",
        "",
        f"  Synthesis window:  {wave_min:.3f}–{wave_max:.3f} nm",
        f"  Line centre:       {HALPHA_NM:.3f} nm  (H-alpha)",
        f"  EW (±{args.ew_window:.1f} nm):      {ew_nm:.4f} nm  =  {ew_ang:.4f} Å",
        separator,
        "",
    ]
    text = "\n".join(result_lines)
    print(text)
    with open(output_dir / "halpha_synth.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")

    # ---- 9. Save outputs -------------------------------------------------
    # FITS
    fits_path = output_dir / "halpha_model.fits"
    save_synthetic_spectrum(waveobs, flux, fits_path)

    # ASCII
    ascii_path = output_dir / "halpha_model.txt"
    save_ascii(waveobs, flux, ascii_path)

    # JSON summary
    summary = {
        "params":     {k: float(v) for k, v in params.items()},
        "halpha_nm":  HALPHA_NM,
        "wave_min_nm": float(wave_min),
        "wave_max_nm": float(wave_max),
        "ew_nm":      round(ew_nm, 6) if np.isfinite(ew_nm) else None,
        "ew_ang":     round(ew_ang, 6) if np.isfinite(ew_ang) else None,
        "ew_window_nm": args.ew_window,
        "code":       code,
        "atmosphere": args.atmosphere,
    }

    # Also measure EW in observed spectrum if provided, for direct comparison
    obs_result = None
    if args.observed:
        obs_result = load_observed(args.observed,
                                   centre_nm=HALPHA_NM, window_nm=args.wing)
        if obs_result is not None:
            obs_wave, obs_flux = obs_result
            ew_obs_nm = measure_ew(obs_wave, obs_flux,
                                   centre_nm=HALPHA_NM, window_nm=args.ew_window)
            ew_obs_ang = ew_obs_nm * 10.0
            delta_ew_ang = ew_obs_ang - ew_ang
            log.info(
                "Observed Hα EW (±%.1f nm): %.4f Å  |  "
                "ΔEW (obs−model) = %+.4f Å",
                args.ew_window, ew_obs_ang, delta_ew_ang,
            )
            summary["ew_obs_nm"]    = round(ew_obs_nm, 6)  if np.isfinite(ew_obs_nm)  else None
            summary["ew_obs_ang"]   = round(ew_obs_ang, 6) if np.isfinite(ew_obs_ang) else None
            summary["delta_ew_ang"] = round(delta_ew_ang, 6)

            obs_ascii = output_dir / "halpha_observed_trimmed.txt"
            with open(obs_ascii, "w") as f:
                f.write("# waveobs_nm\tflux\n")
                for w, fl in zip(obs_wave, obs_flux):
                    f.write(f"{w:.6f}\t{fl:.8f}\n")
            log.info("Trimmed observed H-alpha saved to %s", obs_ascii)

    import json as _json
    with open(output_dir / "halpha_results.json", "w") as f:
        _json.dump(summary, f, indent=2)
    log.info("JSON summary saved to %s", output_dir / "halpha_results.json")

    log.info("Done.  Output in %s/", output_dir)


if __name__ == "__main__":
    main()
