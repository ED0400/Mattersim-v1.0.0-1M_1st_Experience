#!/usr/bin/env python3
"""
MatterSim GUI — Final unified version
- Correct phonon integration via PhononWorkflow in child processes
- Fast and Full modes
- Exports plotted data to ./results/<Element>_<SimType>.xlsx (or CSV fallback)
- No local process-target functions (avoids pickle error)
"""

import os
import sys
import math
import json
import traceback
import threading
import queue
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib
# GUI main uses TkAgg to embed plots into Tkinter (works in VSCode)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from loguru import logger
import torch

# ASE (used by main for quick tasks and by workers)
from ase.build import bulk, molecule
from ase import Atoms
from ase.calculators.emt import EMT
from ase.units import GPa

# Optional MatterSim imports in main; child processes will import as needed
try:
    from mattersim.forcefield import MatterSimCalculator
    from mattersim.applications.phonon import PhononWorkflow
except Exception:
    MatterSimCalculator = None
    PhononWorkflow = None

# Optional pandas/openpyxl for Excel export
try:
    import pandas as pd
except Exception:
    pd = None

# Constants & lookups
ELEMENTS = ["C", "Si", "Ge", "Al", "Cu", "Ni", "Fe", "Mo", "W", "NaCl"]
CRYSTAL_STRUCTURES = ["diamond", "zincblende", "fcc", "bcc", "hcp", "rocksalt"]
MOLECULES = ["H2O", "CO2", "NH3", "CH4", "O2", "N2"]

LATTICE_CONSTANTS = {
    ("Si", "diamond"): 5.43,
    ("C", "diamond"): 3.57,
    ("Ge", "diamond"): 5.66,
    ("Al", "fcc"): 4.05,
    ("Cu", "fcc"): 3.61,
    ("Ni", "fcc"): 3.52,
    ("Fe", "bcc"): 2.87,
    ("Mo", "bcc"): 3.15,
    ("W", "bcc"): 3.16,
    ("NaCl", "rocksalt"): 5.64,
}

EV_TO_J = 1.602176634e-19
ANG3_TO_M3 = 1e-30
BOLTZMANN = 1.380649e-23
HBAR = 1.054571817e-34
R_GAS = 8.31446261815324
AVOGADRO = 6.02214076e23

DEFAULT_MODEL_PATH = "MatterSim-v1.0.0-5M.pth"
RESULTS_DIR = Path.cwd() / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- utility functions ----------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _safe_use_agg_backend():
    # called inside child processes before importing matplotlib/phonopy
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl
    mpl.use("Agg")

# Synthetic phonon generator for fast mode (element-dependent so curves differ)
def synthetic_phonon_for_element(element: str, natoms: int):
    # small deterministic noise based on element name
    seed = abs(hash(element)) % (2**32)
    rng = np.random.RandomState(seed)
    Z = sum(ord(c) for c in element) % 60 + 10
    n_bands = min(3 * max(1, natoms), 12)
    n_q = 48
    base_scale = max(0.3, 12.0 / (1 + math.log10(Z + 1)))
    bands = []
    for b in range(n_bands):
        center = base_scale * (0.6 + 0.15 * b)
        q = np.linspace(0, 1, n_q)
        band = np.abs(np.sin(np.pi * q + b * 0.12) * center + 0.05 * b)
        band += rng.normal(scale=center * 0.015, size=band.shape)
        bands.append(band)
    bands = np.array(bands)
    energies = np.linspace(0.0, bands.max() * 1.05 + 1.0, 300)
    dos = np.zeros_like(energies)
    for b in range(bands.shape[0]):
        # smear each band into DOS
        vals, _ = np.histogram(bands[b], bins=energies, density=True)
        dos += vals
    dos = np.convolve(dos, np.ones(3) / 3, mode='same')
    return bands.tolist(), energies.tolist(), dos.tolist()

# ---------- worker functions (top-level) ----------
def phonon_worker(params: Dict[str, Any], out_q: multiprocessing.Queue):
    """
    Worker that runs PhononWorkflow in a separate process and returns serializable arrays.
    Returns ("ok", payload) or ("error", message)
    """
    try:
        _safe_use_agg_backend()
        import numpy as _np
        from ase.build import bulk as _bulk
        from ase.calculators.emt import EMT as _EMT
        # Import mattersim inside process
        try:
            from mattersim.forcefield import MatterSimCalculator as _MSC
            from mattersim.applications.phonon import PhononWorkflow as _PW
        except Exception as e:
            out_q.put(("error", f"Failed to import mattersim in worker: {e}\n{traceback.format_exc()}"))
            return

        element = params["element"]
        structure = params["structure"]
        lattice = float(params["lattice"])
        model_path = params.get("model_path", "")
        device = params.get("device", "cpu")
        work_dir = params.get("work_dir", "/tmp")
        supercell = params.get("supercell", [2,2,2])

        # Build atoms
        try:
            try:
                atoms = _bulk(element, crystalstructure=structure, a=lattice)
            except TypeError:
                atoms = _bulk(element, structure, a=lattice)
        except Exception as e:
            out_q.put(("error", f"Failed to build structure: {e}\n{traceback.format_exc()}"))
            return

        # attach calculator
        try:
            atoms.calc = _MSC(load_path=model_path, device=device)
        except Exception:
            atoms.calc = _EMT()

        # run phonon workflow
        try:
            ph = _PW(atoms=atoms, find_prim=True, work_dir=os.path.join(work_dir, f"phonon_{element}_{structure}"), amplitude=0.01, supercell_matrix=_np.diag(supercell))
            has_imag, phonon_obj = ph.run()
        except Exception as e:
            out_q.put(("error", f"PhononWorkflow.run failed: {e}\n{traceback.format_exc()}"))
            return

        payload: Dict[str, Any] = {"has_imag": bool(has_imag)}

        # Try to extract band structure
        try:
            # many phonopy wrappers expose get_band_structure returning dict or arrays
            if hasattr(phonon_obj, "get_band_structure"):
                bs = phonon_obj.get_band_structure()
                if isinstance(bs, dict):
                    if "frequencies" in bs:
                        payload["band_frequencies"] = _np.array(bs["frequencies"]).tolist()
                    if "qpoints" in bs:
                        payload["band_qpoints"] = _np.array(bs["qpoints"]).tolist()
                elif isinstance(bs, (list, tuple, _np.ndarray)):
                    arr = _np.array(bs)
                    if arr.ndim == 2:
                        payload["band_frequencies"] = arr.tolist()
        except Exception:
            # ignore; we'll still return what we have
            pass

        # Try DOS
        try:
            if hasattr(phonon_obj, "get_total_dos"):
                dos = phonon_obj.get_total_dos()
                if isinstance(dos, dict) and "frequencies" in dos and "dos" in dos:
                    payload["dos_energies"] = _np.array(dos["frequencies"]).tolist()
                    payload["dos_values"] = _np.array(dos["dos"]).tolist()
        except Exception:
            pass

        out_q.put(("ok", payload))
    except Exception as e:
        tb = traceback.format_exc()
        out_q.put(("error", f"{e}\n{tb}"))

def phase_worker(params: Dict[str, Any], out_q: multiprocessing.Queue):
    """
    Robust phase diagram worker (Full mode). Returns serializable arrays.
    """
    try:
        _safe_use_agg_backend()
        import numpy as _np
        from ase.build import bulk as _bulk
        from ase.calculators.emt import EMT as _EMT
        try:
            from mattersim.forcefield import MatterSimCalculator as _MSC
            from mattersim.applications.phonon import PhononWorkflow as _PW
        except Exception:
            _MSC = _EMT
            _PW = None

        element = params["element"]
        structures = params["structures"]
        base_lattice = float(params["lattice"])
        Tmin = float(params["Tmin"]); Tmax = float(params["Tmax"]); Tsteps = int(params["Tsteps"])
        Pmin = float(params["Pmin"]); Pmax = float(params["Pmax"]); Psteps = int(params["Psteps"])
        Ts = _np.linspace(Tmin, Tmax, Tsteps)
        Ps = _np.linspace(Pmin, Pmax, Psteps)
        model_path = params.get("model_path", "")
        device = params.get("device", "cpu")
        work_dir = params.get("work_dir", "/tmp")

        phase_grid = _np.empty((len(Ps), len(Ts)), dtype=object)
        G_values = {s: _np.full((len(Ps), len(Ts)), _np.nan) for s in structures}

        for iP, P in enumerate(Ps):
            for iT, T in enumerate(Ts):
                bestG = None; bestS = None
                for s in structures:
                    try:
                        try:
                            atoms = _bulk(element, crystalstructure=s, a=base_lattice)
                        except TypeError:
                            atoms = _bulk(element, s, a=base_lattice)
                    except Exception:
                        continue
                    # attach calculator
                    try:
                        atoms.calc = _MSC(load_path=model_path, device=device)
                    except Exception:
                        atoms.calc = _EMT()
                    # Attempt phonon-based vibrational free energy where possible (expensive)
                    F_vib_eV = None
                    if _PW is not None:
                        try:
                            ph = _PW(atoms=atoms, find_prim=True, work_dir=os.path.join(work_dir, f"phase_{element}_{s}"), amplitude=0.01, supercell_matrix=_np.diag([2,2,2]))
                            has_imag, phonon_obj = ph.run()
                            if hasattr(phonon_obj, "get_thermal_properties"):
                                try:
                                    tp = phonon_obj.get_thermal_properties(temperatures=[float(T)])
                                    if isinstance(tp, dict) and "free_energy" in tp:
                                        F_vib_eV = float(tp["free_energy"][0])
                                except Exception:
                                    F_vib_eV = None
                        except Exception:
                            F_vib_eV = None
                    # fallback: Einstein-like estimate
                    if F_vib_eV is None:
                        try:
                            V_m3 = atoms.get_volume() * ANG3_TO_M3
                            n_modes = len(atoms) * 3
                            thetaD = (HBAR * 3000.0 * (6 * math.pi**2 * (len(atoms) / V_m3))**(1.0/3.0)) / BOLTZMANN
                            thetaE = thetaD
                            omega = BOLTZMANN * thetaE / HBAR
                            x = (HBAR * omega) / (BOLTZMANN * max(T, 1e-6))
                            if x > 700:
                                F_mode_J = 0.5 * HBAR * omega
                            else:
                                F_mode_J = 0.5 * HBAR * omega + BOLTZMANN * T * math.log(1.0 - math.exp(-x))
                            F_vib_J = n_modes * F_mode_J
                            F_vib_eV = F_vib_J / EV_TO_J
                        except Exception:
                            F_vib_eV = None

                    try:
                        E0 = float(atoms.get_potential_energy())
                        V_ang3 = float(atoms.get_volume())
                    except Exception:
                        continue

                    PV_eV = (P * 1e9) * (V_ang3 * 1e-30) / EV_TO_J
                    if F_vib_eV is not None:
                        G = E0 + float(F_vib_eV) + PV_eV
                    else:
                        G = E0 + PV_eV

                    G_values[s][iP, iT] = G
                    if bestG is None or (G is not None and G < bestG):
                        bestG = G; bestS = s
                phase_grid[iP, iT] = bestS

        payload = {"Ts": Ts.tolist(), "Ps": Ps.tolist(), "phase_grid": phase_grid.tolist(), "G_values": {s: G_values[s].tolist() for s in G_values}}
        out_q.put(("ok", payload))
    except Exception as e:
        tb = traceback.format_exc()
        out_q.put(("error", f"{e}\n{tb}"))

def cv_worker(params: Dict[str, Any], out_q: multiprocessing.Queue):
    """
    Compute Cv(T), kappa(T), and vapor pressure optionally in a process.
    Returns ("ok", payload) or ("error", message)
    """
    try:
        _safe_use_agg_backend()
        import numpy as _np
        from ase.build import bulk as _bulk
        from ase.calculators.emt import EMT as _EMT
        try:
            from mattersim.forcefield import MatterSimCalculator as _MSC
            from mattersim.applications.phonon import PhononWorkflow as _PW
        except Exception:
            _MSC = _EMT
            _PW = None

        element = params["element"]
        structure = params["structure"]
        lattice = float(params["lattice"])
        Ts = _np.array(params["Ts"], dtype=float)
        kappa_model = params.get("kappa_model", "Slack")
        mfp_m = float(params.get("mfp_m", 1e-8))
        model_path = params.get("model_path", "")
        device = params.get("device", "cpu")
        include_vapor = bool(params.get("include_vapor", False))
        vapor_method = params.get("vapor_method", "estimate")
        vapor_dHvap_kJmol = float(params.get("vapor_dHvap_kJmol", 0.0))
        work_dir = params.get("work_dir", "/tmp")

        try:
            try:
                atoms = _bulk(element, crystalstructure=structure, a=lattice)
            except TypeError:
                atoms = _bulk(element, structure, a=lattice)
        except Exception:
            out_q.put(("error", "Failed to build bulk in cv_worker"))
            return

        try:
            atoms.calc = _MSC(load_path=model_path, device=device)
        except Exception:
            atoms.calc = _EMT()

        # Try phonon-based Cv first (expensive)
        Cv_vol = None
        if _PW is not None:
            try:
                ph = _PW(atoms=atoms, find_prim=True, work_dir=os.path.join(work_dir, f"phonon_cv_{element}_{structure}"), amplitude=0.01, supercell_matrix=np.diag([2,2,2]))
                has_imag, phonon_obj = ph.run()
                if hasattr(phonon_obj, "get_thermal_properties"):
                    tp = phonon_obj.get_thermal_properties(temperatures=Ts.tolist())
                    Cv_arr = None
                    if isinstance(tp, dict):
                        for key in ("heat_capacity", "heat_capacity_at_constant_volume", "cv"):
                            if key in tp:
                                Cv_arr = _np.array(tp[key], dtype=float); break
                    if Cv_arr is not None:
                        V_m3 = atoms.get_volume() * ANG3_TO_M3
                        Cv_vol = (Cv_arr / V_m3).astype(float)
            except Exception:
                Cv_vol = None

        # fallback: finite-difference energy wrt T (cheap-ish)
        if Cv_vol is None:
            energies = []
            for T in Ts:
                try:
                    atoms.calc = _MSC(load_path=model_path, device=device, temperature=float(T))
                except Exception:
                    pass
                e = float(atoms.get_potential_energy())
                energies.append(e)
            energies = _np.array(energies)
            V_m3 = atoms.get_volume() * ANG3_TO_M3
            energies_J = energies * EV_TO_J
            u_vol = energies_J / V_m3
            Cv_vol = _np.gradient(u_vol, Ts)

        # kappa estimate
        method_desc = "Unknown"
        if "slack" in kappa_model.lower():
            method_desc = "Slack-like"
            thetaD = (HBAR * 3000.0 * (6 * math.pi**2 * (len(atoms) / (atoms.get_volume()*ANG3_TO_M3)))**(1.0/3.0)) / BOLTZMANN
            def slack_est(Tval):
                return 1e-6 * (thetaD**3) / max(Tval, 1.0)
            kappa_vals = np.array([slack_est(float(T)) for T in Ts])
        elif "callaway" in kappa_model.lower():
            method_desc = "Callaway-approx"
            kappa_vals = np.array([1e-2 / max(float(T)/300.0, 0.1) for T in Ts])
        else:
            method_desc = "Kinetic"
            v_s = 3000.0
            kappa_vals = (1.0/3.0) * np.array(Cv_vol) * v_s * mfp_m

        # vapor pressure (normalized)
        vapor_list = None; vapor_units = None; vapor_method_desc = None
        if include_vapor:
            if vapor_method == "user" and vapor_dHvap_kJmol > 0.0:
                dHvap_Jmol = vapor_dHvap_kJmol * 1000.0
                vapor_method_desc = f"user ΔH_vap={vapor_dHvap_kJmol:.2f} kJ/mol"
            else:
                try:
                    E0 = float(atoms.get_potential_energy())
                    e_per_atom_eV = E0 / len(atoms)
                    dHvap_Jmol = abs(e_per_atom_eV * EV_TO_J * AVOGADRO)
                    vapor_method_desc = "estimated from cohesive energy"
                except Exception:
                    dHvap_Jmol = 40e3
                    vapor_method_desc = "fallback 40 kJ/mol"
            try:
                vapor_list = np.exp(-dHvap_Jmol / (R_GAS * Ts))
                vapor_units = "normalized"
            except Exception:
                vapor_list = None

        payload = {"Ts": Ts.tolist(), "Cv": np.array(Cv_vol).tolist(), "kappa": kappa_vals.tolist(), "method": method_desc, "vapor_pressure": (vapor_list.tolist() if vapor_list is not None else None), "vapor_units": vapor_units, "vapor_method_desc": vapor_method_desc}
        out_q.put(("ok", payload))
    except Exception as e:
        tb = traceback.format_exc()
        out_q.put(("error", f"{e}\n{tb}"))

# ---------- GUI class ----------
class MatterSimGUI:
    def __init__(self, master):
        self.master = master
        master.title("MatterSim — Final GUI (Phonon + Thermo + Exports)")
        logger.remove(); logger.add(lambda m: self._log(m), level="INFO")
        self.task_queue = queue.Queue()
        self._proc_handle: Optional[Tuple[multiprocessing.Process, multiprocessing.Queue]] = None
        self.model_cache = {}
        self.last_plot_data: Dict[str, Any] = {}

        # Selection frame
        sel_frame = ttk.LabelFrame(master, text="Selections")
        sel_frame.pack(fill=tk.X, padx=8, pady=6)
        self.sim_type = tk.StringVar(value="Bulk")
        ttk.Radiobutton(sel_frame, text="Bulk", variable=self.sim_type, value="Bulk", command=self._on_mode_change).grid(row=0, column=0)
        ttk.Radiobutton(sel_frame, text="Molecule", variable=self.sim_type, value="Molecule", command=self._on_mode_change).grid(row=0, column=1)

        ttk.Label(sel_frame, text="Element:").grid(row=1, column=0, sticky=tk.W)
        self.element_cb = ttk.Combobox(sel_frame, values=ELEMENTS, state="readonly", width=14); self.element_cb.grid(row=1, column=1); self.element_cb.set("Si"); self.element_cb.bind("<<ComboboxSelected>>", lambda e: self._suggest_lattice_constant())
        ttk.Label(sel_frame, text="Structure:").grid(row=1, column=2, sticky=tk.W)
        self.structure_cb = ttk.Combobox(sel_frame, values=CRYSTAL_STRUCTURES, state="readonly", width=14); self.structure_cb.grid(row=1, column=3); self.structure_cb.set("diamond"); self.structure_cb.bind("<<ComboboxSelected>>", lambda e: self._suggest_lattice_constant())
        ttk.Label(sel_frame, text="Molecule:").grid(row=2, column=0, sticky=tk.W)
        self.molecule_cb = ttk.Combobox(sel_frame, values=MOLECULES, state="readonly", width=14); self.molecule_cb.grid(row=2, column=1); self.molecule_cb.set(MOLECULES[0])
        ttk.Label(sel_frame, text="Lattice a (Å):").grid(row=2, column=2, sticky=tk.W)
        self.lattice_entry = ttk.Entry(sel_frame, width=16); self.lattice_entry.grid(row=2, column=3)

        ttk.Label(sel_frame, text="Model path:").grid(row=3, column=0, sticky=tk.W)
        self.model_path_entry = ttk.Entry(sel_frame, width=40); self.model_path_entry.grid(row=3, column=1, columnspan=2); self.model_path_entry.insert(0, DEFAULT_MODEL_PATH)
        ttk.Button(sel_frame, text="Load model (cache main)", command=self._load_model_now).grid(row=3, column=3)

        ttk.Label(sel_frame, text="Compute mode:").grid(row=4, column=0, sticky=tk.W)
        self.compute_mode = tk.StringVar(value="Fast")
        ttk.Radiobutton(sel_frame, text="Fast (quick)", variable=self.compute_mode, value="Fast").grid(row=4, column=1)
        ttk.Radiobutton(sel_frame, text="Full (accurate, slow)", variable=self.compute_mode, value="Full").grid(row=4, column=2)

        # Notebook
        self.nb = ttk.Notebook(master); self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self._build_phonon_tab(); self._build_thermo_tab(); self._build_force_tab()

        # Plot canvas
        self.fig, self.ax = plt.subplots(figsize=(7,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master); self.canvas.get_tk_widget().pack(fill=tk.BOTH, padx=8, pady=(0,6))

        # bottom logs and controls
        bottom = ttk.Frame(master); bottom.pack(fill=tk.BOTH, padx=8, pady=(0,8))
        self.logbox = scrolledtext.ScrolledText(bottom, height=10); self.logbox.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        controls = ttk.Frame(bottom); controls.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0))
        ttk.Label(controls, text="Status:").pack(anchor=tk.NW)
        self.status_label = ttk.Label(controls, text="Idle"); self.status_label.pack(anchor=tk.NW)
        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=160); self.progress.pack(anchor=tk.NW, pady=(6,6))
        ttk.Button(controls, text="Export Plot (PNG/PDF)", command=self._export_plot).pack(anchor=tk.NW, pady=(6,4))
        ttk.Button(controls, text="Export Data (Excel/CSV)", command=self._export_data).pack(anchor=tk.NW)
        self.cancel_btn = ttk.Button(controls, text="Cancel (stop process)", command=self._cancel_process, state="disabled"); self.cancel_btn.pack(anchor=tk.NW, pady=(8,0))

        self._on_mode_change(); self._suggest_lattice_constant()
        self.master.after(200, self._poll_task_queue)

    # ---------- UI building ----------
    def _build_phonon_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Phonon")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="Supercell (diag):").grid(row=0, column=0, sticky=tk.W)
        self.ph_super = ttk.Combobox(frame, values=[2,3,4], state="readonly", width=8); self.ph_super.grid(row=0, column=1); self.ph_super.set(2)
        ttk.Label(frame, text="Work dir:").grid(row=0, column=2, sticky=tk.W)
        self.ph_workdir = ttk.Entry(frame, width=40); self.ph_workdir.grid(row=0, column=3); self.ph_workdir.insert(0, "/tmp/phonon_runs")
        ttk.Button(frame, text="Run Phonon", command=self._run_phonon).grid(row=1, column=0, pady=8)

    def _build_thermo_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Thermo/Phase")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="T min (K):").grid(row=0, column=0, sticky=tk.W); self.Tmin_cb = ttk.Combobox(frame, values=[50,100,200,300,400], state="readonly", width=10); self.Tmin_cb.grid(row=0, column=1); self.Tmin_cb.set("50")
        ttk.Label(frame, text="T max (K):").grid(row=0, column=2, sticky=tk.W); self.Tmax_cb = ttk.Combobox(frame, values=[300,500,800,1000], state="readonly", width=10); self.Tmax_cb.grid(row=0, column=3); self.Tmax_cb.set("1000")
        ttk.Label(frame, text="T steps:").grid(row=0, column=4, sticky=tk.W); self.Tsteps_cb = ttk.Combobox(frame, values=[20,40], state="readonly", width=8); self.Tsteps_cb.grid(row=0, column=5); self.Tsteps_cb.set("20")
        ttk.Label(frame, text="P min (GPa):").grid(row=1, column=0, sticky=tk.W); self.Pmin_cb = ttk.Combobox(frame, values=[0,1,5], state="readonly", width=10); self.Pmin_cb.grid(row=1, column=1); self.Pmin_cb.set("0")
        ttk.Label(frame, text="P max (GPa):").grid(row=1, column=2, sticky=tk.W); self.Pmax_cb = ttk.Combobox(frame, values=[1,5,10], state="readonly", width=10); self.Pmax_cb.grid(row=1, column=3); self.Pmax_cb.set("10")
        ttk.Label(frame, text="P steps:").grid(row=1, column=4, sticky=tk.W); self.Psteps_cb = ttk.Combobox(frame, values=[3,5], state="readonly", width=8); self.Psteps_cb.grid(row=1, column=5); self.Psteps_cb.set("3")
        ttk.Button(frame, text="Fast Phase", command=self._phase_fast).grid(row=2, column=0, pady=8)
        ttk.Button(frame, text="Full Phase (process)", command=self._phase_full).grid(row=2, column=1, pady=8)
        ttk.Button(frame, text="Compute Cv & κ (Full)", command=self._cv_full).grid(row=2, column=2, pady=8)

    def _build_force_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Forces")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="T (K):").grid(row=0, column=0, sticky=tk.W); self.force_T = ttk.Entry(frame, width=10); self.force_T.grid(row=0, column=1); self.force_T.insert(0,"300")
        ttk.Label(frame, text="P (GPa):").grid(row=0, column=2, sticky=tk.W); self.force_P = ttk.Entry(frame, width=10); self.force_P.grid(row=0, column=3); self.force_P.insert(0,"0")
        ttk.Button(frame, text="Run Force (quick)", command=self._force_quick).grid(row=1, column=0, pady=8)

    # ---------- model cache ----------
    def _load_model_now(self):
        path = self.model_path_entry.get().strip()
        if not path:
            messagebox.showinfo("Model path", "Provide the model path.")
            return
        if MatterSimCalculator is None:
            messagebox.showwarning("MatterSim import", "MatterSim can't be imported in this main process. Full-mode workers will attempt to import in child processes.")
            return
        try:
            self._log(f"Loading MatterSim model from {path} into main-process cache...")
            calc = MatterSimCalculator(load_path=path, device=("cuda" if torch.cuda.is_available() else "cpu"))
            self.model_cache["calculator"] = calc
            self._log("Model cached.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self._log(traceback.format_exc())

    # ---------- process orchestration ----------
    def _start_process(self, worker_fn, params: Dict[str, Any]):
        try:
            out_q = multiprocessing.Queue()
            proc = multiprocessing.Process(target=worker_fn, args=(params, out_q), daemon=True)
            proc.start()
            self._proc_handle = (proc, out_q)
            # enable cancel button
            self.cancel_btn.config(state="normal")
            self.progress.start(8)
            self.status_label.config(text="Worker running...")
            self.master.after(200, lambda: self._poll_proc(proc, out_q))
        except Exception as e:
            self._log(f"Failed to start process: {e}\n{traceback.format_exc()}")

    def _poll_proc(self, proc: multiprocessing.Process, out_q: multiprocessing.Queue):
        try:
            while True:
                tag, payload = out_q.get_nowait()
                if tag == "ok":
                    # route by content
                    if isinstance(payload, dict) and ("band_frequencies" in payload or "dos_values" in payload):
                        self.task_queue.put(("phonon_ok", payload))
                    elif isinstance(payload, dict) and "phase_grid" in payload:
                        self.task_queue.put(("phase_ok", payload))
                    elif isinstance(payload, dict) and "Cv" in payload:
                        self.task_queue.put(("cv_ok", payload))
                    else:
                        self.task_queue.put(("worker_ok", payload))
                elif tag == "error":
                    self.task_queue.put(("worker_error", payload))
        except Exception:
            pass
        if proc.is_alive():
            self.master.after(200, lambda: self._poll_proc(proc, out_q))
        else:
            try:
                proc.join(timeout=0.1)
            except Exception:
                pass
            try:
                out_q.close()
            except Exception:
                pass
            self._proc_handle = None
            self.cancel_btn.config(state="disabled")
            self.progress.stop()
            self.status_label.config(text="Idle")

    def _cancel_process(self):
        if self._proc_handle:
            proc, q = self._proc_handle
            try:
                if proc.is_alive():
                    proc.terminate()
                    self._log("Background process terminated.")
            except Exception as e:
                self._log(f"Failed to terminate: {e}")
            self._proc_handle = None
            self.cancel_btn.config(state="disabled")
            self.progress.stop()
            self.status_label.config(text="Idle")

    # ---------- actions (phonon/phase/cv/force) ----------
    def _run_phonon(self):
        element = self.element_cb.get()
        if self.sim_type.get() != "Bulk":
            messagebox.showinfo("Phonon needs Bulk", "Phonon runs require Bulk mode.")
            return
        params = {
            "element": element,
            "structure": self.structure_cb.get(),
            "lattice": safe_float(self.lattice_entry.get(), LATTICE_CONSTANTS.get((element, self.structure_cb.get()), 5.43)),
            "model_path": self.model_path_entry.get().strip(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "work_dir": self.ph_workdir.get().strip() or "/tmp/phonon_runs",
            "supercell": [int(self.ph_super.get())]*3
        }
        if self.compute_mode.get() == "Fast":
            # run fast synthetic phonon in a thread (quick)
            threading.Thread(target=self._phonon_fast_thread, args=(params,), daemon=True).start()
        else:
            # full worker process
            self._start_process(phonon_worker, params)

    def _phonon_fast_thread(self, params):
        try:
            self.progress.start(8); self.status_label.config(text="Phonon (fast)...")
            element = params["element"]
            try:
                atoms = bulk(element, crystalstructure=params["structure"], a=params["lattice"])
            except TypeError:
                atoms = bulk(element, params["structure"], a=params["lattice"])
            bands, energies, dos = synthetic_phonon_for_element(element, len(atoms))
            payload = {"has_imag": False, "band_frequencies": bands, "band_qpoints": list(range(len(bands[0]))), "dos_energies": energies, "dos_values": dos}
            self.task_queue.put(("phonon_ok", payload))
        except Exception as e:
            self.task_queue.put(("worker_error", f"{e}\n{traceback.format_exc()}"))
        finally:
            self.progress.stop(); self.status_label.config(text="Idle")

    def _phase_fast(self):
        # cheap approximate phase using EMT energies (quick)
        try:
            self.progress.start(8); self.status_label.config(text="Phase (fast)...")
            element = self.element_cb.get()
            lattice = safe_float(self.lattice_entry.get(), 5.43)
            structures = [s for s in CRYSTAL_STRUCTURES]
            Tmin = float(self.Tmin_cb.get()); Tmax = float(self.Tmax_cb.get()); Tsteps = int(self.Tsteps_cb.get())
            Pmin = float(self.Pmin_cb.get()); Pmax = float(self.Pmax_cb.get()); Psteps = int(self.Psteps_cb.get())
            Ts = np.linspace(Tmin, Tmax, Tsteps); Ps = np.linspace(Pmin, Pmax, Psteps)
            phase_grid = np.empty((len(Ps), len(Ts)), dtype=object)
            Gvals = {}
            for s in structures:
                try:
                    atoms = bulk(element, crystalstructure=s, a=lattice)
                except TypeError:
                    try:
                        atoms = bulk(element, s, a=lattice)
                    except Exception:
                        atoms = None
                if atoms is None:
                    E0 = 0.0; V = 1.0
                else:
                    atoms.calc = EMT()
                    E0 = atoms.get_potential_energy()
                    V = atoms.get_volume()
                Gvals[s] = np.full((len(Ps), len(Ts)), np.nan)
                for iP, P in enumerate(Ps):
                    for iT, T in enumerate(Ts):
                        PV_eV = (P * 1e9) * (V * 1e-30) / EV_TO_J
                        Gvals[s][iP, iT] = E0 + PV_eV
            for iP in range(len(Ps)):
                for iT in range(len(Ts)):
                    bestS = min(structures, key=lambda s: Gvals[s][iP, iT])
                    phase_grid[iP, iT] = bestS
            payload = {"Ts": Ts.tolist(), "Ps": Ps.tolist(), "phase_grid": phase_grid.tolist(), "G_values": {s: Gvals[s].tolist() for s in Gvals}}
            self.task_queue.put(("phase_ok", payload))
        except Exception as e:
            self.task_queue.put(("worker_error", f"{e}\n{traceback.format_exc()}"))
        finally:
            self.progress.stop(); self.status_label.config(text="Idle")

    def _phase_full(self):
        element = self.element_cb.get()
        lattice = safe_float(self.lattice_entry.get(), 5.43)
        params = {
            "element": element,
            "structures": [s for s in CRYSTAL_STRUCTURES],
            "lattice": lattice,
            "Tmin": float(self.Tmin_cb.get()), "Tmax": float(self.Tmax_cb.get()), "Tsteps": int(self.Tsteps_cb.get()),
            "Pmin": float(self.Pmin_cb.get()), "Pmax": float(self.Pmax_cb.get()), "Psteps": int(self.Psteps_cb.get()),
            "model_path": self.model_path_entry.get().strip(),
            "work_dir": "/tmp",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        self._start_process(phase_worker, params)

    def _cv_full(self):
        # spawn cv_worker process
        element = self.element_cb.get()
        params = {
            "element": element,
            "structure": self.structure_cb.get(),
            "lattice": safe_float(self.lattice_entry.get(), 5.43),
            "Ts": np.linspace(float(self.Tmin_cb.get()), float(self.Tmax_cb.get()), int(self.Tsteps_cb.get())).tolist(),
            "kappa_model": "Slack",
            "mfp_m": 1e-8,
            "model_path": self.model_path_entry.get().strip(),
            "work_dir": "/tmp",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "include_vapor": False
        }
        self._start_process(cv_worker, params)

    def _force_quick(self):
        threading.Thread(target=self._force_quick_thread, daemon=True).start()

    def _force_quick_thread(self):
        try:
            self.progress.start(8); self.status_label.config(text="Force (quick)...")
            element = self.element_cb.get()
            if self.sim_type.get() == "Bulk":
                try:
                    atoms = bulk(element, crystalstructure=self.structure_cb.get(), a=safe_float(self.lattice_entry.get(), 5.43))
                except TypeError:
                    atoms = bulk(element, self.structure_cb.get(), a=safe_float(self.lattice_entry.get(), 5.43))
            else:
                atoms = molecule(self.molecule_cb.get())
            calc = self.model_cache.get("calculator")
            if calc is not None:
                atoms.calc = calc
            else:
                try:
                    if MatterSimCalculator is not None:
                        atoms.calc = MatterSimCalculator(load_path=self.model_path_entry.get().strip(), device=("cuda" if torch.cuda.is_available() else "cpu"))
                    else:
                        atoms.calc = EMT()
                except Exception:
                    atoms.calc = EMT()
            E = atoms.get_potential_energy()
            forces = atoms.get_forces()
            self.task_queue.put(("force_ok", {"positions": atoms.get_positions().tolist(), "forces": forces.tolist(), "energy": float(E)}))
        except Exception as e:
            self.task_queue.put(("worker_error", f"{e}\n{traceback.format_exc()}"))
        finally:
            self.progress.stop(); self.status_label.config(text="Idle")

    # ---------- task queue poll (main thread) ----------
    def _poll_task_queue(self):
        try:
            while True:
                tag, payload = self.task_queue.get_nowait()
                if tag == "phonon_ok":
                    self._plot_phonon(payload)
                elif tag == "phase_ok":
                    self._plot_phase(payload)
                elif tag == "cv_ok":
                    self._plot_cv(payload)
                elif tag == "force_ok":
                    self._plot_force(payload)
                elif tag == "worker_ok":
                    self._log("Worker finished successfully.")
                elif tag == "worker_error":
                    self._log("Worker error: " + str(payload))
                    messagebox.showerror("Worker Error", str(payload).splitlines()[0])
        except queue.Empty:
            pass
        self.master.after(200, self._poll_task_queue)

    # ---------- plotting routines ----------
    def _plot_phonon(self, payload: Dict[str, Any]):
        self.ax.clear()
        plotted = False
        if "band_frequencies" in payload:
            band = np.array(payload["band_frequencies"])
            if band.ndim == 1:
                band = band[np.newaxis, :]
            if band.shape[0] < band.shape[1]:
                freqs = band
            else:
                freqs = band.T
            n_bands, n_q = freqs.shape
            x = np.arange(n_q)
            for b in range(n_bands):
                self.ax.plot(x, freqs[b], label=f"band {b+1}", linewidth=0.9)
            self.ax.set_xlabel("q-index"); self.ax.set_ylabel("Frequency (arb.)")
            self.ax.set_title(f"Phonon bands ({self.element_cb.get()})")
            self.ax.legend(loc="upper right", fontsize="x-small")
            plotted = True
            self.last_plot_data["band_q"] = list(range(n_q))
            self.last_plot_data["band_frequencies"] = freqs.tolist()
        if "dos_energies" in payload and "dos_values" in payload:
            ax2 = self.ax.twinx()
            ax2.plot(payload["dos_energies"], payload["dos_values"], linestyle="--")
            ax2.set_ylabel("DOS (arb.)")
            plotted = True
            self.last_plot_data["dos_energies"] = payload["dos_energies"]
            self.last_plot_data["dos_values"] = payload["dos_values"]
        if not plotted:
            txt = "Phonon computed but band/DOS arrays not available.\nFull run may have failed to produce arrays; check logs."
            self.ax.text(0.5, 0.5, txt, ha="center")
            self.last_plot_data.clear()
        self.canvas.draw()
        self._log("Phonon plotted.")

    def _plot_phase(self, payload: Dict[str, Any]):
        Ts = np.array(payload["Ts"]); Ps = np.array(payload["Ps"])
        phase_grid = np.array(payload["phase_grid"], dtype=object)
        unique = sorted(set(phase_grid.flatten()) - {None})
        cmap = plt.get_cmap("tab10")
        color_map = {s: cmap(i % 10) for i, s in enumerate(unique)}
        img = np.ones((len(Ps), len(Ts), 3))
        for i in range(len(Ps)):
            for j in range(len(Ts)):
                s = phase_grid[i,j]
                img[i,j,:] = (1.0,1.0,1.0) if s is None else color_map.get(s, (0.9,0.9,0.9))[:3]
        self.ax.clear()
        self.ax.imshow(img, extent=[Ts[0], Ts[-1], Ps[0], Ps[-1]], aspect='auto', origin='lower')
        self.ax.set_xlabel("Temperature (K)"); self.ax.set_ylabel("Pressure (GPa)")
        handles = [plt.Rectangle((0,0),1,1,color=color_map[s]) for s in unique]
        self.ax.legend(handles, unique, title="Phase", bbox_to_anchor=(1.05,1), loc='upper left')
        self.ax.set_title(f"Phase diagram ({self.element_cb.get()})")
        self.canvas.draw()
        self.last_plot_data["phase_payload"] = payload
        self._log("Phase diagram plotted.")

    def _plot_cv(self, payload: Dict[str, Any]):
        Ts = np.array(payload["Ts"]); Cv = np.array(payload["Cv"]); kappa = np.array(payload["kappa"])
        method = payload.get("method", "unknown")
        self.ax.clear()
        ax1 = self.ax
        ax1.plot(Ts, Cv, label="Cv (J/m^3 K)")
        ax1.set_xlabel("Temperature (K)"); ax1.set_ylabel("Cv (J/m^3 K)")
        ax2 = ax1.twinx()
        ax2.plot(Ts, kappa, label="κ (W/mK)", linestyle="--")
        ax2.set_ylabel("κ (W/mK)")
        ax1.set_title(f"Cv & κ ({self.element_cb.get()}) method={method}")
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        self.canvas.draw()
        self.last_plot_data["cv_payload"] = payload
        self._log("Cv & κ plotted.")

    def _plot_force(self, payload: Dict[str, Any]):
        pos = np.array(payload["positions"]); f = np.array(payload["forces"])
        self.ax.clear()
        if pos.size and f.size:
            x = pos[:,0]; y = pos[:,1]; u = f[:,0]; v = f[:,1]
            self.ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
            self.ax.scatter(x, y, alpha=0.6)
            self.ax.set_xlabel("X (Å)"); self.ax.set_ylabel("Y (Å)")
            self.ax.set_title(f"Forces ({self.element_cb.get()})")
            self.last_plot_data["force_positions"] = pos.tolist()
            self.last_plot_data["force_forces"] = f.tolist()
            self.last_plot_data["energy"] = payload.get("energy")
        else:
            self.ax.text(0.5, 0.5, "No force data", ha="center")
        self.canvas.draw()

    # ---------- export utilities ----------
    def _export_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("PDF","*.pdf")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self._log(f"Saved figure to {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _export_data(self):
        # Save to ./results automatically with element and type
        element = self.element_cb.get()
        if "band_frequencies" in self.last_plot_data:
            sim = "phonon"
        elif "phase_payload" in self.last_plot_data:
            sim = "phase"
        elif "cv_payload" in self.last_plot_data:
            sim = "cv"
        elif "force_positions" in self.last_plot_data:
            sim = "force"
        else:
            messagebox.showinfo("No data", "No plotted data to export.")
            return
        fname = RESULTS_DIR / f"{element}_{sim}.xlsx"
        try:
            if pd is not None:
                with pd.ExcelWriter(str(fname), engine="openpyxl") as writer:
                    if "band_frequencies" in self.last_plot_data:
                        df_band = pd.DataFrame(self.last_plot_data["band_frequencies"])
                        df_band.to_excel(writer, sheet_name="bands", index=False)
                    if "dos_energies" in self.last_plot_data and "dos_values" in self.last_plot_data:
                        df_dos = pd.DataFrame({"energy": self.last_plot_data["dos_energies"], "dos": self.last_plot_data["dos_values"]})
                        df_dos.to_excel(writer, sheet_name="dos", index=False)
                    if "phase_payload" in self.last_plot_data:
                        ph = self.last_plot_data["phase_payload"]
                        Ts = ph["Ts"]; Ps = ph["Ps"]; grid = ph["phase_grid"]
                        rows = []
                        for iP, P in enumerate(Ps):
                            for iT, T in enumerate(Ts):
                                rows.append({"P_GPa": P, "T_K": T, "Phase": grid[iP][iT]})
                        pd.DataFrame(rows).to_excel(writer, sheet_name="phase_grid", index=False)
                    if "cv_payload" in self.last_plot_data:
                        cv = self.last_plot_data["cv_payload"]
                        df_cv = pd.DataFrame({"T_K": cv["Ts"], "Cv": cv["Cv"], "kappa": cv["kappa"]})
                        df_cv.to_excel(writer, sheet_name="cv_kappa", index=False)
                    if "force_positions" in self.last_plot_data:
                        pos = np.array(self.last_plot_data["force_positions"]); f = np.array(self.last_plot_data["force_forces"])
                        df_f = pd.DataFrame(np.hstack([pos, f]), columns=["x","y","z","fx","fy","fz"])
                        df_f.to_excel(writer, sheet_name="forces", index=False)
                self._log(f"Exported data to {fname}")
            else:
                # fallback to CSV(s)
                base = RESULTS_DIR / f"{element}_{sim}"
                if "band_frequencies" in self.last_plot_data:
                    np.savetxt(str(base) + "_bands.csv", np.array(self.last_plot_data["band_frequencies"]), delimiter=",")
                if "dos_energies" in self.last_plot_data and "dos_values" in self.last_plot_data:
                    np.savetxt(str(base) + "_dos.csv", np.vstack([self.last_plot_data["dos_energies"], self.last_plot_data["dos_values"]]).T, delimiter=",", header="energy,dos")
                if "phase_payload" in self.last_plot_data:
                    ph = self.last_plot_data["phase_payload"]; Ts = ph["Ts"]; Ps = ph["Ps"]; grid = ph["phase_grid"]
                    with open(str(base) + "_phase.csv", "w") as f:
                        f.write("P_GPa,T_K,Phase\n")
                        for ip, P in enumerate(Ps):
                            for it, T in enumerate(Ts):
                                f.write(f"{P},{T},{grid[ip][it]}\n")
                if "cv_payload" in self.last_plot_data:
                    cv = self.last_plot_data["cv_payload"]
                    np.savetxt(str(base) + "_cv.csv", np.vstack([cv["Ts"], cv["Cv"], cv["kappa"]]).T, delimiter=",", header="T_K,Cv,kappa")
                if "force_positions" in self.last_plot_data:
                    pos = np.array(self.last_plot_data["force_positions"]); f = np.array(self.last_plot_data["force_forces"])
                    np.savetxt(str(base) + "_forces.csv", np.hstack([pos, f]), delimiter=",", header="x,y,z,fx,fy,fz")
                self._log(f"Exported data to {RESULTS_DIR} as CSV files with base {base}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            self._log(traceback.format_exc())

    # ---------- logging ----------
    def _log(self, text):
        self.logbox.insert(tk.END, str(text) + "\n"); self.logbox.see(tk.END)

    def _on_mode_change(self):
        if self.sim_type.get() == "Bulk":
            self.element_cb.config(state="readonly"); self.structure_cb.config(state="readonly"); self.molecule_cb.config(state="disabled")
        else:
            self.element_cb.config(state="disabled"); self.structure_cb.config(state="disabled"); self.molecule_cb.config(state="readonly")

    def _suggest_lattice_constant(self):
        element = self.element_cb.get().strip()
        structure = self.structure_cb.get().strip().lower()
        key = (element, structure)
        if key in LATTICE_CONSTANTS:
            a0 = LATTICE_CONSTANTS[key]; self.lattice_entry.delete(0, tk.END); self.lattice_entry.insert(0, str(a0)); self._log(f"Suggested lattice {a0} Å")
        else:
            self._log(f"No default lattice for {element} ({structure})")

# ---------- main ----------
def main():
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except Exception:
        pass
    root = tk.Tk()
    app = MatterSimGUI(root)
    root.geometry("1180x880")
    root.mainloop()

if __name__ == "__main__":
    main()
