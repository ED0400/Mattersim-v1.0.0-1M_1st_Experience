#!/usr/bin/env python3
"""
MatterSim GUI — multiprocessing-safe phonopy / thermodynamics version.

- Runs phonon / thermo / phase-diagram tasks in separate processes to avoid
  Matplotlib GUI creation in non-main threads (fixes "Fail to allocate bitmap").
- Main GUI uses TkAgg; child workers set MPLBACKEND=Agg before importing matplotlib/phonopy.
- Includes dropdown-only selectors, lattice auto-suggestion, Slack & Callaway models,
  caching, export, tooltips, and progress/queue plumbing.

Requirements:
 - Python 3.8+
 - tkinter, matplotlib, ase, torch, phonopy (optional), loguru
 - mattersim package (your local package)
"""

# Standard libs
import os
import sys
import json
import math
import pickle
import traceback
from pathlib import Path
import multiprocessing
import queue
import threading
import time

# Scientific libs
import numpy as np
import matplotlib
# Keep default GUI backend for main process (TkAgg) - do NOT switch here.
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# GUI libs
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from loguru import logger
import torch

# ASE and fallback calculator
from ase.build import bulk, molecule
from ase import Atoms
from ase.calculators.emt import EMT
from ase.units import GPa

# mattersim imports — these will be imported in workers as well
# (we import the calculator here for force evaluations)
from mattersim.forcefield import MatterSimCalculator
# PhononWorkflow will be used in workers which import it there.

# ---------- Constants & lookups ----------
ELEMENTS = ["C", "Si", "Ge", "Al", "Cu", "Ni", "Fe", "Mo", "W", "NaCl", "GaAs", "SiC"]
CRYSTAL_STRUCTURES = ["diamond", "zincblende", "fcc", "bcc", "hcp", "rocksalt"]
MOLECULES = ["H2O", "CO2", "NH3", "CH4", "O2", "N2", "H2", "CO"]

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
    ("GaAs", "zincblende"): 5.65,
    ("SiC", "zincblende"): 4.36,
}

# conversions/constants
EV_TO_J = 1.602176634e-19
ANG3_TO_M3 = 1e-30
BOLTZMANN = 1.380649e-23
HBAR = 1.054571817e-34

# phonon cache file
CACHE_PATH = Path.home() / ".mattersim_phonon_cache.pkl"

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ---------- cache helpers ----------
def load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

# ---------- Slack / Callaway heuristics ----------
def estimate_debye_temperature(atoms):
    V_m3 = atoms.get_volume() * ANG3_TO_M3
    n = len(atoms)
    avg_mass = atoms.get_masses().mean()
    if avg_mass < 50:
        v_s = 4000.0
    elif avg_mass > 150:
        v_s = 2000.0
    else:
        v_s = 3000.0
    kD = (6 * math.pi**2 * (n / V_m3))**(1.0/3.0)
    thetaD = (HBAR * v_s * kD) / BOLTZMANN
    return thetaD

def slack_kappa_estimate(T, atoms, gamma=1.5, theta_D=None):
    V_m3 = atoms.get_volume() * ANG3_TO_M3
    n = len(atoms)
    mean_mass_amu = atoms.get_masses().mean()
    M_bar = mean_mass_amu * 1.66053906660e-27
    if theta_D is None:
        theta_D = estimate_debye_temperature(atoms)
    V_a_m3 = V_m3 / n
    A = 2.0e-6
    kappa = A * (M_bar) * (theta_D**3) * (V_a_m3**(1.0/3.0)) / (gamma**2 * max(T,1.0))
    return float(kappa)

def callaway_approx_kappa(T, atoms, mfp=10e-9, gamma=1.5, theta_D=None):
    V_m3 = atoms.get_volume() * ANG3_TO_M3
    n = len(atoms)
    if theta_D is None:
        theta_D = estimate_debye_temperature(atoms)
    omega_avg = BOLTZMANN * theta_D / HBAR
    Cv_per_cell = 3 * n * BOLTZMANN * (1.0 if T > theta_D / 5 else (T / max(theta_D,1.0)))
    Cv_vol = Cv_per_cell / V_m3
    avg_mass_amu = atoms.get_masses().mean()
    if avg_mass_amu < 50:
        v_s = 4000.0
    elif avg_mass_amu > 150:
        v_s = 2000.0
    else:
        v_s = 3000.0
    B_U = 1e-19
    tau_U_inv = B_U * (omega_avg**2) * T * math.exp(-theta_D / (3.0 * max(T,1.0)))
    tau_B_inv = v_s / max(mfp, 1e-12)
    tau_inv = tau_U_inv + tau_B_inv
    tau_eff = 1.0 / tau_inv if tau_inv > 0 else 1e-12
    kappa = (1.0/3.0) * Cv_vol * v_s**2 * tau_eff
    return float(kappa)

# ---------- worker helpers for processes ----------
def _child_use_agg_backend():
    # Ensure child process uses non-interactive backend
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")

# --- Top-level process workers (must be at module top-level) ---

def phase_process_worker(params, out_q):
    """
    Compute phase diagram in a child process and put ("ok", payload) or ("error", msg) into out_q.
    payload: {"Ts": [...], "Ps": [...], "phase_grid": [[...]], "G_values": {structure: [[...]]}}
    """
    try:
        _child_use_agg_backend()
        import numpy as _np
        from ase.build import bulk as _bulk
        from ase.calculators.emt import EMT as _EMT
        from mattersim.forcefield import MatterSimCalculator as _MSC
        from mattersim.applications.phonon import PhononWorkflow as _PhononWorkflow

        element = params["element"]
        structures = params["structures"]
        lattice = float(params["lattice"])
        Tmin = float(params["Tmin"]); Tmax = float(params["Tmax"]); Tsteps = int(params["Tsteps"])
        Pmin = float(params["Pmin"]); Pmax = float(params["Pmax"]); Psteps = int(params["Psteps"])
        Ts = _np.linspace(Tmin, Tmax, Tsteps)
        Ps = _np.linspace(Pmin, Pmax, Psteps)
        model_path = params.get("model_path", "")

        phase_grid = _np.empty((len(Ps), len(Ts)), dtype=object)
        G_values = {s: _np.full((len(Ps), len(Ts)), _np.nan) for s in structures}

        for iP, P in enumerate(Ps):
            for iT, T in enumerate(Ts):
                bestG = None; bestS = None
                for s in structures:
                    try:
                        try:
                            atoms = _bulk(element, crystalstructure=s, a=lattice)
                        except TypeError:
                            atoms = _bulk(element, s, a=lattice)
                    except Exception:
                        continue
                    try:
                        atoms.calc = _MSC(load_path=model_path, device=params.get("device", "cpu"))
                    except Exception:
                        atoms.calc = _EMT()
                    F_vib = None
                    try:
                        ph = _PhononWorkflow(atoms=atoms, find_prim=True, work_dir=params.get("work_dir","/tmp"), amplitude=0.01, supercell_matrix=_np.diag([2,2,2]))
                        has_imag, phonon_obj = ph.run()
                        if hasattr(phonon_obj, "get_thermal_properties"):
                            try:
                                tp = phonon_obj.get_thermal_properties(temperatures=[float(T)])
                                if isinstance(tp, dict) and "free_energy" in tp:
                                    F_vib = float(tp["free_energy"][0])
                                elif isinstance(tp, (list, tuple, _np.ndarray)) and len(tp) > 0:
                                    try:
                                        F_vib = float(_np.array(tp[0])[0])
                                    except Exception:
                                        F_vib = None
                            except Exception:
                                F_vib = None
                    except Exception:
                        F_vib = None
                    try:
                        E0 = float(atoms.get_potential_energy())
                        V_ang3 = float(atoms.get_volume())
                    except Exception:
                        continue
                    if F_vib is not None:
                        # convert P (GPa) * V (Å^3) to eV: P*1e9 [Pa] * V_ang3*1e-30 [m3] = J -> /EV_TO_J -> eV
                        G = E0 + F_vib + (P * 1e9) * (V_ang3 * 1e-30) / EV_TO_J
                    else:
                        G = E0 + (P * 1e9) * (V_ang3 * 1e-30) / EV_TO_J
                    G_values[s][iP,iT] = G
                    if bestG is None or (G is not None and G < bestG):
                        bestG = G; bestS = s
                phase_grid[iP, iT] = bestS

        payload = {"Ts": Ts.tolist(), "Ps": Ps.tolist(), "phase_grid": phase_grid.tolist(), "G_values": {s: G_values[s].tolist() for s in G_values}}
        out_q.put(("ok", payload))
    except Exception as e:
        out_q.put(("error", str(e) + "\n" + traceback.format_exc()))

def cv_kappa_process_worker(params, out_q):
    """
    Compute Cv and kappa vs T in child process; put ("ok", payload) or ("error", msg).
    payload: {"Ts": [...], "Cv": [...], "kappa": [...], "method": "..."}
    """
    try:
        _child_use_agg_backend()
        import numpy as _np
        from ase.build import bulk as _bulk
        from ase.calculators.emt import EMT as _EMT
        from mattersim.forcefield import MatterSimCalculator as _MSC
        from mattersim.applications.phonon import PhononWorkflow as _PhononWorkflow

        element = params["element"]
        structure = params["structure"]
        lattice = float(params["lattice"])
        Ts = _np.array(params["Ts"], dtype=float)
        kappa_model = params.get("kappa_model", "Slack")
        mfp_m = float(params.get("mfp_m", 1e-8))
        model_path = params.get("model_path", "")
        try:
            atoms = _bulk(element, crystalstructure=structure, a=lattice)
        except TypeError:
            atoms = _bulk(element, structure, a=lattice)

        try:
            atoms.calc = _MSC(load_path=model_path, device=params.get("device","cpu"))
        except Exception:
            atoms.calc = _EMT()

        Cv_vol = None
        try:
            ph = _PhononWorkflow(atoms=atoms, find_prim=True, work_dir=params.get("work_dir","/tmp"), amplitude=0.01, supercell_matrix=_np.diag([2,2,2]))
            has_imag, phonon_obj = ph.run()
            if hasattr(phonon_obj, "get_thermal_properties"):
                tp = phonon_obj.get_thermal_properties(temperatures=Ts.tolist())
                Cv_array = None
                if isinstance(tp, dict):
                    for key in ["heat_capacity", "heat_capacity_at_constant_volume", "cv"]:
                        if key in tp:
                            Cv_array = _np.array(tp[key], dtype=float)
                            break
                elif isinstance(tp, (list, tuple, _np.ndarray)):
                    try:
                        Cv_array = _np.array(tp[0], dtype=float)
                    except Exception:
                        Cv_array = None
                if Cv_array is not None:
                    V_m3 = atoms.get_volume() * 1e-30
                    Cv_vol = (Cv_array / V_m3).astype(float)
        except Exception:
            Cv_vol = None

        if Cv_vol is None:
            energies = []
            for T in Ts:
                try:
                    atoms.calc = _MSC(load_path=model_path, device=params.get("device","cpu"), temperature=float(T))
                except Exception:
                    pass
                e = float(atoms.get_potential_energy())
                energies.append(e)
            energies = _np.array(energies)
            V_m3 = atoms.get_volume() * 1e-30
            energies_J = energies * EV_TO_J
            u_vol = energies_J / V_m3
            Cv_vol = _np.gradient(u_vol, Ts)

        # compute kappa according to selected model
        if kappa_model.startswith("Slack"):
            thetaD = estimate_debye_temperature(atoms)
            kappa_vals = _np.array([slack_kappa_estimate(float(T), atoms, gamma=1.5, theta_D=thetaD) for T in Ts])
            method = "Slack"
        elif kappa_model.startswith("Callaway"):
            kappa_vals = _np.array([callaway_approx_kappa(float(T), atoms, mfp=mfp_m) for T in Ts])
            method = "Callaway"
        else:
            avg_mass_amu = atoms.get_masses().mean()
            if avg_mass_amu < 50:
                v_s = 4000.0
            elif avg_mass_amu > 150:
                v_s = 2000.0
            else:
                v_s = 3000.0
            kappa_vals = (1.0/3.0) * _np.array(Cv_vol) * v_s * mfp_m
            method = "Kinetic"

        payload = {"Ts": Ts.tolist(), "Cv": _np.array(Cv_vol).tolist(), "kappa": _np.array(kappa_vals).tolist(), "method": method}
        out_q.put(("ok", payload))
    except Exception as e:
        out_q.put(("error", str(e) + "\n" + traceback.format_exc()))

# ---------- GUI / main-thread code ----------
class ToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self.tipwin = None
        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.unschedule)
        widget.bind("<ButtonPress>", self.unschedule)
    def schedule(self, event=None):
        self.unschedule()
        self._id = self.widget.after(self.delay, self.show)
    def unschedule(self, event=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        self.hide()
    def show(self):
        if self.tipwin or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, padx=4, pady=2)
        label.pack()
    def hide(self):
        if self.tipwin:
            self.tipwin.destroy()
        self.tipwin = None

class MatterSimGUI:
    def __init__(self, master):
        self.master = master
        master.title("MatterSim GUI (process-safe phonopy/thermo)")
        logger.remove(); logger.add(lambda m: self._log(m), level="INFO")
        self.task_queue = queue.Queue()
        self._proc_handle = None
        self._last_phase_boundaries = None
        self.cache = load_cache()

        # Top selection frame
        sel_frame = ttk.LabelFrame(master, text="Selections (click-only)")
        sel_frame.pack(fill=tk.X, padx=8, pady=6)

        self.sim_type = tk.StringVar(value="Bulk")
        ttk.Label(sel_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=6)
        ttk.Radiobutton(sel_frame, text="Bulk Material", variable=self.sim_type, value="Bulk", command=self._on_mode_change).grid(row=0,column=1,sticky=tk.W)
        ttk.Radiobutton(sel_frame, text="Molecule/Compound", variable=self.sim_type, value="Molecule", command=self._on_mode_change).grid(row=0,column=2,sticky=tk.W)

        ttk.Label(sel_frame, text="Element / Material:").grid(row=1,column=0,sticky=tk.W,padx=6)
        self.element_cb = ttk.Combobox(sel_frame, values=ELEMENTS, state="readonly", width=18)
        self.element_cb.grid(row=1,column=1,sticky=tk.W); self.element_cb.bind("<<ComboboxSelected>>", lambda e: self._on_element_or_structure_change())
        if "Si" in ELEMENTS: self.element_cb.set("Si")

        ttk.Label(sel_frame, text="Crystal structure:").grid(row=1,column=2,sticky=tk.W,padx=6)
        self.structure_cb = ttk.Combobox(sel_frame, values=CRYSTAL_STRUCTURES, state="readonly", width=15)
        self.structure_cb.grid(row=1,column=3,sticky=tk.W); self.structure_cb.set("diamond"); self.structure_cb.bind("<<ComboboxSelected>>", lambda e: self._on_element_or_structure_change())

        ttk.Label(sel_frame, text="Molecule:").grid(row=2,column=0,sticky=tk.W,padx=6)
        self.molecule_cb = ttk.Combobox(sel_frame, values=MOLECULES, state="readonly", width=18)
        self.molecule_cb.grid(row=2,column=1,sticky=tk.W); self.molecule_cb.set(MOLECULES[0])

        ttk.Label(sel_frame, text="Lattice a (Å):").grid(row=2,column=2,sticky=tk.W,padx=6)
        self.lattice_entry = ttk.Entry(sel_frame, width=16)
        self.lattice_entry.grid(row=2,column=3,sticky=tk.W)

        ttk.Label(sel_frame, text="MatterSim model path:").grid(row=3,column=0,sticky=tk.W,padx=6,pady=6)
        self.model_path_entry = ttk.Entry(sel_frame, width=40)
        self.model_path_entry.grid(row=3,column=1,columnspan=2,sticky=tk.W); self.model_path_entry.insert(0, "MatterSim-v1.0.0-5M.pth")
        self.device_label = ttk.Label(sel_frame, text="Device: auto")
        self.device_label.grid(row=3,column=3,sticky=tk.W)

        cfg_frame = ttk.Frame(sel_frame); cfg_frame.grid(row=4,column=0,columnspan=4,pady=(6,0),sticky=tk.W)
        ttk.Button(cfg_frame, text="Save Config", command=self.save_config).grid(row=0,column=0,padx=(0,6))
        ttk.Button(cfg_frame, text="Load Config", command=self.load_config).grid(row=0,column=1)

        ToolTip(self.element_cb, "Select element/material (click-only).")
        ToolTip(self.structure_cb, "Select crystal structure (click-only).")
        ToolTip(self.lattice_entry, "Auto-suggested from element+structure; editable for override.")

        # Notebook and tabs
        self.nb = ttk.Notebook(master); self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self._build_force_tab(); self._build_phonon_simple_tab(); self._build_phonon_advanced_tab(); self._build_thermo_tab()

        # Figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, padx=8, pady=(0,6))

        # Bottom area: logs and controls
        bottom_frame = ttk.Frame(master); bottom_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0,8))
        self.output = scrolledtext.ScrolledText(bottom_frame, height=10, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        right_ctrl = ttk.Frame(bottom_frame); right_ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0))
        ttk.Label(right_ctrl, text="Status:").pack(anchor=tk.NW)
        self.status_label = ttk.Label(right_ctrl, text="Idle"); self.status_label.pack(anchor=tk.NW, pady=(0,8))
        self.progress = ttk.Progressbar(right_ctrl, mode="indeterminate", length=180); self.progress.pack(anchor=tk.NW, pady=(0,8))
        ToolTip(self.progress, "Shows activity during long runs.")
        self.run_btn = ttk.Button(right_ctrl, text="Run Selected", command=self._on_run_clicked); self.run_btn.pack(anchor=tk.NW, pady=(0,6))
        self.cancel_btn = ttk.Button(right_ctrl, text="Cancel", command=self._on_cancel_clicked, state="disabled"); self.cancel_btn.pack(anchor=tk.NW)

        # init states
        self._on_mode_change(); self._suggest_lattice_constant()
        self.master.after(200, self._poll_task_queue)

    # ---------- Config save/load ----------
    def save_config(self):
        cfg = {
            "sim_type": self.sim_type.get(),
            "element": self.element_cb.get(),
            "structure": self.structure_cb.get(),
            "molecule": self.molecule_cb.get(),
            "lattice": self.lattice_entry.get(),
            "model_path": self.model_path_entry.get(),
        }
        path = filedialog.asksaveasfilename(title="Save configuration", defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._log(f"Config saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_config(self):
        path = filedialog.askopenfilename(title="Load configuration", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            if "sim_type" in cfg: self.sim_type.set(cfg["sim_type"])
            if "element" in cfg and cfg["element"] in ELEMENTS: self.element_cb.set(cfg["element"])
            if "structure" in cfg and cfg["structure"] in CRYSTAL_STRUCTURES: self.structure_cb.set(cfg["structure"])
            if "molecule" in cfg and cfg["molecule"] in MOLECULES: self.molecule_cb.set(cfg["molecule"])
            if "lattice" in cfg: self.lattice_entry.delete(0, tk.END); self.lattice_entry.insert(0, str(cfg["lattice"]))
            if "model_path" in cfg: self.model_path_entry.delete(0, tk.END); self.model_path_entry.insert(0, cfg["model_path"])
            self._on_mode_change(); self._log(f"Config loaded from {path}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ---------- Selection handlers ----------
    def _on_mode_change(self):
        mode = self.sim_type.get()
        if mode == "Bulk":
            self.element_cb.config(state="readonly"); self.structure_cb.config(state="readonly"); self.molecule_cb.config(state="disabled")
        else:
            self.element_cb.config(state="disabled"); self.structure_cb.config(state="disabled"); self.molecule_cb.config(state="readonly")
        self._suggest_lattice_constant()

    def _on_element_or_structure_change(self):
        self._suggest_lattice_constant()

    def _suggest_lattice_constant(self):
        if self.sim_type.get() != "Bulk":
            return
        el = (self.element_cb.get() or "").strip()
        struct = (self.structure_cb.get() or "").strip().lower()
        if not el or not struct:
            return
        key = (el, struct)
        if key in LATTICE_CONSTANTS:
            a0 = LATTICE_CONSTANTS[key]
            self.lattice_entry.delete(0, tk.END); self.lattice_entry.insert(0, str(a0))
            self._log(f"Suggested lattice: {el} ({struct}) -> {a0} Å")
            return
        # fallback message
        self._log(f"No default lattice for {el} ({struct}). Enter value or use fallback.")

    # ---------- Tabs build ----------
    def _build_force_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Force Simulation")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="Temperature (K):").grid(row=0,column=0,sticky=tk.W,padx=6)
        self.temp_entry = ttk.Entry(frame, width=12); self.temp_entry.grid(row=0,column=1,sticky=tk.W); self.temp_entry.insert(0,"300")
        ttk.Label(frame, text="Pressure (GPa):").grid(row=0,column=2,sticky=tk.W,padx=6)
        self.pressure_entry = ttk.Entry(frame,width=12); self.pressure_entry.grid(row=0,column=3,sticky=tk.W); self.pressure_entry.insert(0,"0")

    def _build_phonon_simple_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Phonon (Simple)")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="Supercell (diag):").grid(row=0,column=0,sticky=tk.W,padx=6)
        self.ph_simple_super = ttk.Combobox(frame, values=["2","3","4","5","6"], state="readonly", width=8); self.ph_simple_super.grid(row=0,column=1); self.ph_simple_super.set("4")
        ttk.Label(frame, text="Amplitude:").grid(row=0,column=2,sticky=tk.W,padx=6)
        self.ph_simple_amp = ttk.Entry(frame,width=12); self.ph_simple_amp.grid(row=0,column=3); self.ph_simple_amp.insert(0,"0.01")
        ttk.Label(frame, text="Work dir:").grid(row=1,column=0,sticky=tk.W,padx=6)
        self.ph_simple_dir = ttk.Entry(frame,width=40); self.ph_simple_dir.grid(row=1,column=1,columnspan=3,sticky=tk.W); self.ph_simple_dir.insert(0,"/tmp/phonon_simple")
        ttk.Button(frame, text="Run Phonon (Simple)", command=self._on_run_phonon_simple_process).grid(row=2,column=0,pady=8)

    def _build_phonon_advanced_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Phonon (Advanced)")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="Supercell (3 ints):").grid(row=0,column=0,sticky=tk.W,padx=6)
        self.ph_adv_super = ttk.Combobox(frame, values=["4,4,4","3,3,3","2,2,2","6,6,6"], state="readonly", width=12); self.ph_adv_super.grid(row=0,column=1); self.ph_adv_super.set("4,4,4")
        ttk.Label(frame, text="q-mesh:").grid(row=0,column=2,sticky=tk.W,padx=6)
        self.ph_adv_qmesh = ttk.Combobox(frame, values=["12,12,12","8,8,8","6,6,6"], state="readonly", width=12); self.ph_adv_qmesh.grid(row=0,column=3); self.ph_adv_qmesh.set("12,12,12")
        ttk.Label(frame, text="Amplitude:").grid(row=1,column=0,sticky=tk.W,padx=6)
        self.ph_adv_amp = ttk.Entry(frame,width=12); self.ph_adv_amp.grid(row=1,column=1); self.ph_adv_amp.insert(0,"0.01")
        ttk.Label(frame, text="Find primitive:").grid(row=1,column=2,sticky=tk.W,padx=6)
        self.ph_adv_find_prim = tk.BooleanVar(value=True); ttk.Checkbutton(frame, variable=self.ph_adv_find_prim).grid(row=1,column=3,sticky=tk.W)
        ttk.Label(frame, text="Work dir:").grid(row=2,column=0,sticky=tk.W,padx=6)
        self.ph_adv_dir = ttk.Entry(frame,width=40); self.ph_adv_dir.grid(row=2,column=1,columnspan=3,sticky=tk.W); self.ph_adv_dir.insert(0,"/tmp/phonon_adv")
        ttk.Button(frame, text="Run Phonon (Advanced)", command=self._on_run_phonon_adv_process).grid(row=3,column=0,pady=8)

    def _build_thermo_tab(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Thermodynamics / Phase Diagram")
        frame = ttk.Frame(tab); frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(frame, text="T min (K):").grid(row=0,column=0,sticky=tk.W,padx=6)
        self.thermo_Tmin = ttk.Combobox(frame, values=[str(x) for x in [50,100,200,300,400,500,800,1000]], state="readonly", width=10); self.thermo_Tmin.grid(row=0,column=1); self.thermo_Tmin.set("50")
        ttk.Label(frame, text="T max (K):").grid(row=0,column=2,sticky=tk.W,padx=6)
        self.thermo_Tmax = ttk.Combobox(frame, values=[str(x) for x in [300,500,800,1000,1500,2000]], state="readonly", width=10); self.thermo_Tmax.grid(row=0,column=3); self.thermo_Tmax.set("1000")
        ttk.Label(frame, text="T steps:").grid(row=0,column=4,sticky=tk.W,padx=6)
        self.thermo_Tsteps = ttk.Combobox(frame, values=["20","40","80","100"], state="readonly", width=8); self.thermo_Tsteps.grid(row=0,column=5); self.thermo_Tsteps.set("40")
        ttk.Label(frame, text="P min (GPa):").grid(row=1,column=0,sticky=tk.W,padx=6)
        self.thermo_Pmin = ttk.Combobox(frame, values=["0","1","5","10","20","50"], state="readonly", width=10); self.thermo_Pmin.grid(row=1,column=1); self.thermo_Pmin.set("0")
        ttk.Label(frame, text="P max (GPa):").grid(row=1,column=2,sticky=tk.W,padx=6)
        self.thermo_Pmax = ttk.Combobox(frame, values=["0","1","5","10","20","50","100"], state="readonly", width=10); self.thermo_Pmax.grid(row=1,column=3); self.thermo_Pmax.set("10")
        ttk.Label(frame, text="P steps:").grid(row=1,column=4,sticky=tk.W,padx=6)
        self.thermo_Psteps = ttk.Combobox(frame, values=["1","5","10"], state="readonly", width=8); self.thermo_Psteps.grid(row=1,column=5); self.thermo_Psteps.set("5")
        ttk.Label(frame, text="Include structures:").grid(row=2,column=0,sticky=tk.W,padx=6,pady=(6,0))
        self.struct_vars = {}
        col = 1
        for s in CRYSTAL_STRUCTURES:
            v = tk.BooleanVar(value=(s == self.structure_cb.get()))
            cb = ttk.Checkbutton(frame, text=s, variable=v)
            cb.grid(row=2, column=col, sticky=tk.W, padx=4, pady=(6,0))
            self.struct_vars[s] = v
            col += 1
        ttk.Label(frame, text="Mean free path l (nm):").grid(row=3,column=0,sticky=tk.W,padx=6,pady=(6,0))
        self.kappa_mfp = ttk.Combobox(frame, values=["1","5","10","50","100"], state="readonly", width=10); self.kappa_mfp.grid(row=3,column=1); self.kappa_mfp.set("10")
        ttk.Label(frame, text="Conductivity model:").grid(row=3,column=2,sticky=tk.W,padx=6)
        self.kappa_model = ttk.Combobox(frame, values=["Kinetic (1/3 C v l)","Slack (semi-empirical)","Callaway (approx)"], state="readonly", width=18); self.kappa_model.grid(row=3,column=3); self.kappa_model.set("Slack (semi-empirical)")
        ttk.Button(frame, text="Compute Phase Diagram", command=self._on_run_phase_process).grid(row=4,column=0,pady=8,padx=6)
        ttk.Button(frame, text="Compute Cv & κ (over T)", command=self._on_run_thermo_process).grid(row=4,column=1,pady=8,padx=6)
        ttk.Button(frame, text="Export current plot (PNG)", command=self._export_plot).grid(row=4,column=2,pady=8,padx=6)
        ttk.Button(frame, text="Export phase boundaries (CSV)", command=self._export_phase_boundaries).grid(row=4,column=3,pady=8,padx=6)
        ToolTip(frame, "Note: phase diagram & thermodynamic calculations use phonon workflows when available; otherwise fall back to approximations.")

    # ---------- Run / Cancel / Process integration ----------
    def _on_run_clicked(self):
        if self.run_btn["state"] == "disabled": return
        self._disable_controls_for_run()
        self.progress.start(10); self.status_label.config(text="Running (thread)...")
        current_tab = self.nb.index(self.nb.select())
        if current_tab == 0:
            # run force simulation in background thread (safe)
            t = threading.Thread(target=self._force_task_thread, daemon=True); t.start()
        else:
            self._log("Use the dedicated buttons in each tab for phonon/thermo operations.")
            self._enable_controls_after_run()

    def _on_cancel_clicked(self):
        self._log("Cancellation requested (best-effort).")
        # If a process is running, terminate it
        if self._proc_handle is not None:
            proc, out_q = self._proc_handle
            if proc.is_alive():
                try:
                    proc.terminate()
                    self._log("Terminated child process.")
                except Exception as e:
                    self._log(f"Failed to terminate process: {e}")
            self._proc_handle = None
        self.cancel_flag = True

    def _disable_controls_for_run(self):
        self.run_btn.config(state="disabled"); self.cancel_btn.config(state="normal")
        self.element_cb.config(state="disabled"); self.structure_cb.config(state="disabled"); self.molecule_cb.config(state="disabled"); self.model_path_entry.config(state="disabled")

    def _enable_controls_after_run(self):
        self.run_btn.config(state="normal"); self.cancel_btn.config(state="disabled")
        self._on_mode_change(); self.model_path_entry.config(state="normal"); self.progress.stop(); self.status_label.config(text="Idle")

    def _start_process_and_poll(self, worker_fn, params):
        """
        Start a process worker and poll its multiprocessing.Queue for results.
        Results are forwarded into the main-thread task_queue for plotting/logging.
        """
        out_q = multiprocessing.Queue()
        proc = multiprocessing.Process(target=worker_fn, args=(params, out_q), daemon=True)
        proc.start()
        self._proc_handle = (proc, out_q)
        self.master.after(200, lambda: self._poll_process_queue(proc, out_q))

    def _poll_process_queue(self, proc, out_q):
        try:
            while True:
                tag, payload = out_q.get_nowait()
                if tag == "ok":
                    # Decide what payload type is: phase or cv_kappa
                    if "phase_grid" in payload:
                        self.task_queue.put(("plot_phase", payload))
                        # extract boundaries for export
                        boundaries = self._extract_phase_boundaries(np.array(payload["phase_grid"], dtype=object), payload["Ps"], payload["Ts"])
                        self._last_phase_boundaries = boundaries
                        self.task_queue.put(("log", "Phase diagram ready (from process)."))
                    elif "Cv" in payload or "kappa" in payload:
                        self.task_queue.put(("plot_cv_kappa", payload))
                        self.task_queue.put(("log", "Cv & κ ready (from process)."))
                    elif "phonon_done" in payload:
                        self.task_queue.put(("plot_phonon", payload))
                        self.task_queue.put(("log", "Phonon run finished (from process)."))
                else:
                    self.task_queue.put(("error", payload))
        except Exception:
            pass
        if proc.is_alive():
            self.master.after(200, lambda: self._poll_process_queue(proc, out_q))
        else:
            try:
                proc.join(timeout=0.1)
            except Exception:
                pass
            self._proc_handle = None
            self._enable_controls_after_run()

    # ---------- Thread-based force simulation (keeps GUI responsive) ----------
    def _force_task_thread(self):
        try:
            element = self.element_cb.get(); structure = self.structure_cb.get()
            lattice = safe_float(self.lattice_entry.get(), 5.43)
            mode = self.sim_type.get()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = self.model_path_entry.get().strip()
            self._log(f"Force sim: {element},{structure}, a={lattice} on {device}")
            if mode == "Bulk":
                try:
                    atoms = bulk(element, crystalstructure=structure, a=lattice)
                except TypeError:
                    atoms = bulk(element, structure, a=lattice)
            else:
                mol = self.molecule_cb.get(); atoms = molecule(mol)
            try:
                atoms.calc = MatterSimCalculator(load_path=model_path, device=device)
            except Exception:
                atoms.calc = EMT(); self._log("Using EMT fallback calculator.")
            energy = atoms.get_potential_energy(); forces = atoms.get_forces()
            self.task_queue.put(("plot_force", (atoms.get_positions().tolist(), forces.tolist())))
            self.task_queue.put(("log", f"Energy: {energy:.6f} eV"))
        except Exception as e:
            self.task_queue.put(("error", str(e) + "\n" + traceback.format_exc()))

    # ---------- Process-based phonon & thermo starters ----------
    def _on_run_phonon_simple_process(self):
        if self.run_btn["state"] == "disabled": return
        self._disable_controls_for_run(); self.progress.start(10); self.status_label.config(text="Running phonon simple (process)...")
        params = {
            "element": self.element_cb.get(),
            "structure": self.structure_cb.get(),
            "lattice": safe_float(self.lattice_entry.get(), 5.43),
            "super_n": int(self.ph_simple_super.get()),
            "amplitude": float(self.ph_simple_amp.get()),
            "work_dir": self.ph_simple_dir.get().strip() or "/tmp",
            "model_path": self.model_path_entry.get().strip(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        # For phonon we can reuse phase_process_worker style but simpler: run a phonon workflow to check has_imag
        def phonon_simple_worker_wrapper(params, out_q):
            try:
                _child_use_agg_backend()
                import numpy as _np
                from ase.build import bulk as _bulk
                from ase.calculators.emt import EMT as _EMT
                from mattersim.forcefield import MatterSimCalculator as _MSC
                from mattersim.applications.phonon import PhononWorkflow as _PhononWorkflow
                el = params["element"]; struct = params["structure"]; a = float(params["lattice"])
                try:
                    atoms = _bulk(el, crystalstructure=struct, a=a)
                except TypeError:
                    atoms = _bulk(el, struct, a=a)
                try:
                    atoms.calc = _MSC(load_path=params.get("model_path",""), device=params.get("device","cpu"))
                except Exception:
                    atoms.calc = _EMT()
                ph = _PhononWorkflow(atoms=atoms, find_prim=False, work_dir=params.get("work_dir","/tmp"), amplitude=float(params.get("amplitude",0.01)), supercell_matrix=_np.diag([int(params.get("super_n",4))]*3))
                has_imag, phonon_obj = ph.run()
                out_q.put(("ok", {"phonon_done": True, "has_imag": bool(has_imag)}))
            except Exception as e:
                out_q.put(("error", str(e) + "\n" + traceback.format_exc()))
        # launch process
        self._start_process_and_poll(phonon_simple_worker_wrapper, params)

    def _on_run_phonon_adv_process(self):
        if self.run_btn["state"] == "disabled": return
        self._disable_controls_for_run(); self.progress.start(10); self.status_label.config(text="Running phonon advanced (process)...")
        super_vals = [int(x.strip()) for x in self.ph_adv_super.get().split(",") if x.strip()]
        params = {
            "element": self.element_cb.get(),
            "structure": self.structure_cb.get(),
            "lattice": safe_float(self.lattice_entry.get(), 5.43),
            "super_vals": super_vals,
            "qmesh": self.ph_adv_qmesh.get(),
            "amplitude": safe_float(self.ph_adv_amp.get(), 0.01),
            "find_prim": bool(self.ph_adv_find_prim.get()),
            "work_dir": self.ph_adv_dir.get().strip() or "/tmp",
            "model_path": self.model_path_entry.get().strip(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        def phonon_adv_worker(params, out_q):
            try:
                _child_use_agg_backend()
                import numpy as _np
                from ase.build import bulk as _bulk
                from ase.calculators.emt import EMT as _EMT
                from mattersim.forcefield import MatterSimCalculator as _MSC
                from mattersim.applications.phonon import PhononWorkflow as _PhononWorkflow
                el = params["element"]; struct = params["structure"]; a = float(params["lattice"])
                try:
                    atoms = _bulk(el, crystalstructure=struct, a=a)
                except TypeError:
                    atoms = _bulk(el, struct, a=a)
                try:
                    atoms.calc = _MSC(load_path=params.get("model_path",""), device=params.get("device","cpu"))
                except Exception:
                    atoms.calc = _EMT()
                ph = _PhononWorkflow(atoms=atoms, work_dir=params.get("work_dir","/tmp"), find_prim=params.get("find_prim",True), amplitude=params.get("amplitude",0.01), supercell_matrix=_np.diag(params.get("super_vals",[4,4,4])), qpoints_mesh=_np.array([int(x) for x in params.get("qmesh","12,12,12").split(",")]))
                has_imag, phonon_obj = ph.run()
                out_q.put(("ok", {"phonon_done": True, "has_imag": bool(has_imag)}))
            except Exception as e:
                out_q.put(("error", str(e) + "\n" + traceback.format_exc()))
        self._start_process_and_poll(phonon_adv_worker, params)

    def _on_run_phase_process(self):
        if self.run_btn["state"] == "disabled": return
        self._disable_controls_for_run(); self.progress.start(10); self.status_label.config(text="Running phase diagram (process)...")
        structs = [s for s,v in self.struct_vars.items() if v.get()]
        if not structs:
            messagebox.showinfo("No structures", "Select at least one crystal structure to include in phase diagram.")
            self._enable_controls_after_run(); return
        params = {
            "element": self.element_cb.get(),
            "structures": structs,
            "lattice": safe_float(self.lattice_entry.get(), 5.43),
            "Tmin": float(self.thermo_Tmin.get()), "Tmax": float(self.thermo_Tmax.get()), "Tsteps": int(self.thermo_Tsteps.get()),
            "Pmin": float(self.thermo_Pmin.get()), "Pmax": float(self.thermo_Pmax.get()), "Psteps": int(self.thermo_Psteps.get()),
            "model_path": self.model_path_entry.get().strip(),
            "work_dir": "/tmp",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        self._start_process_and_poll(phase_process_worker, params)

    def _on_run_thermo_process(self):
        if self.run_btn["state"] == "disabled": return
        self._disable_controls_for_run(); self.progress.start(10); self.status_label.config(text="Running Cv & κ (process)...")
        Tmin = float(self.thermo_Tmin.get()); Tmax = float(self.thermo_Tmax.get()); Tsteps = int(self.thermo_Tsteps.get())
        Ts = np.linspace(Tmin, Tmax, Tsteps).tolist()
        params = {
            "element": self.element_cb.get(),
            "structure": self.structure_cb.get(),
            "lattice": safe_float(self.lattice_entry.get(), 5.43),
            "Ts": Ts,
            "kappa_model": self.kappa_model.get(),
            "mfp_m": float(self.kappa_mfp.get()) * 1e-9,
            "model_path": self.model_path_entry.get().strip(),
            "work_dir": "/tmp",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        self._start_process_and_poll(cv_kappa_process_worker, params)

    # ---------- task queue polling & plotting in main thread ----------
    def _poll_task_queue(self):
        try:
            while True:
                item = self.task_queue.get_nowait()
                tag, payload = item
                if tag == "log":
                    self._log(payload)
                elif tag == "plot_force":
                    pos, forces = payload; self._plot_force(np.array(pos), np.array(forces))
                elif tag == "plot_phonon":
                    self._plot_phonon_placeholder(payload)
                elif tag == "plot_phase":
                    self._plot_phase_diagram(payload)
                elif tag == "plot_cv_kappa":
                    self._plot_cv_kappa(payload)
                elif tag == "error":
                    self._log("Error:"); self._log(payload); messagebox.showerror("Task error", str(payload).splitlines()[0] if payload else "Error")
                else:
                    self._log(f"Unknown task_queue tag: {tag}")
        except queue.Empty:
            pass
        finally:
            self.master.after(200, self._poll_task_queue)

    # ---------- plotting utilities ----------
    def _plot_force(self, positions, forces):
        self.ax.clear()
        if positions.size and forces.size:
            x = positions[:,0]; y = positions[:,1]; u = forces[:,0]; v = forces[:,1]
            self.ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
            self.ax.scatter(x, y, alpha=0.6)
            self.ax.set_xlabel("X (Å)"); self.ax.set_ylabel("Y (Å)")
            self.ax.set_title("Force Projection (XY)")
        else:
            self.ax.text(0.5,0.5,"No data to plot",ha="center")
        self.canvas.draw()

    def _plot_phonon_placeholder(self, info):
        self.ax.clear()
        txt = "Phonon run complete.\n"
        txt += "Imaginary modes detected.\n" if info.get("has_imag") else "No imaginary modes detected.\n"
        self.ax.text(0.5,0.5,txt,ha="center")
        self.canvas.draw()

    def _plot_phase_diagram(self, payload):
        Ts = np.array(payload["Ts"]); Ps = np.array(payload["Ps"]); phase_grid = np.array(payload["phase_grid"], dtype=object)
        unique_structs = list(sorted(set(phase_grid.flatten()) - {None}))
        if not unique_structs:
            self.ax.clear(); self.ax.text(0.5,0.5,"No phases to plot",ha="center"); self.canvas.draw(); return
        cmap = plt.get_cmap("tab10"); color_map = {s: cmap(i%10) for i,s in enumerate(unique_structs)}
        img = np.zeros((len(Ps), len(Ts), 3))
        for i in range(len(Ps)):
            for j in range(len(Ts)):
                s = phase_grid[i,j]
                if s is None:
                    c = (1.0,1.0,1.0)
                else:
                    c = color_map.get(s, (0.9,0.9,0.9))
                img[i,j,:] = c[:3]
        self.ax.clear()
        self.ax.imshow(img, extent=[Ts[0], Ts[-1], Ps[0], Ps[-1]], aspect='auto', origin='lower')
        self.ax.set_xlabel("Temperature (K)"); self.ax.set_ylabel("Pressure (GPa)")
        handles = [plt.Rectangle((0,0),1,1,color=color_map[s]) for s in unique_structs]
        self.ax.legend(handles, unique_structs, title="Phase (min G)", bbox_to_anchor=(1.05,1), loc='upper left')
        self.ax.set_title("Phase Diagram (approx.)")
        self.canvas.draw()

    def _plot_cv_kappa(self, payload):
        Ts = np.array(payload["Ts"]); Cv = np.array(payload["Cv"]); kappa = np.array(payload["kappa"])
        method = payload.get("method", "unknown")
        self.ax.clear()
        ax1 = self.ax
        ax1.plot(Ts, Cv, label="Cv (J/m^3 K)")
        ax1.set_xlabel("Temperature (K)"); ax1.set_ylabel("Cv (J/m^3 K)")
        ax2 = ax1.twinx()
        ax2.plot(Ts, kappa, label="κ (W/m K)", linestyle="--")
        ax2.set_ylabel("κ (W/m K)")
        ax1.set_title(f"Heat Capacity & Thermal Conductivity (method={method})")
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
        self.canvas.draw()

    # ---------- phase boundary extraction & export ----------
    def _extract_phase_boundaries(self, phase_grid, Ps, Ts):
        boundaries = {}
        Pvals = np.array(Ps); Tvals = np.array(Ts)
        rows, cols = phase_grid.shape
        for iP in range(rows):
            row = phase_grid[iP]
            prev = row[0]
            for j in range(1, cols):
                cur = row[j]
                if cur != prev:
                    pair = tuple(sorted([str(prev), str(cur)]))
                    boundaries.setdefault(pair, []).append((float(Pvals[iP]), float(0.5*(Tvals[j-1]+Tvals[j]))))
                    prev = cur
        return boundaries

    def _export_plot(self):
        path = filedialog.asksaveasfilename(title="Save plot as PNG", defaultextension=".png", filetypes=[("PNG","*.png"),("PDF","*.pdf")])
        if not path: return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self._log(f"Plot saved to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _export_phase_boundaries(self):
        boundaries = getattr(self, "_last_phase_boundaries", None)
        if not boundaries:
            messagebox.showinfo("No data", "No phase boundaries available. Run a phase diagram first.")
            return
        path = filedialog.asksaveasfilename(title="Save boundaries CSV", defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["phase_pair", "P_GPa", "T_K"])
                for pair, pts in boundaries.items():
                    for P,T in pts:
                        writer.writerow(["|".join(pair), P, T])
            self._log(f"Phase boundaries exported to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # ---------- logging ----------
    def _log(self, text):
        self.output.insert(tk.END, text + "\n"); self.output.see(tk.END)

# ---------- main ----------
def main():
    root = tk.Tk()
    app = MatterSimGUI(root)
    root.geometry("1200x900")
    root.mainloop()

if __name__ == "__main__":
    multiprocessing.freeze_support()  # for Windows executables
    main()
