#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ameliorated MatterSim GUI: Full Thermodynamic and Structural Analysis for Elements, Molecules, and Compounds
"""

import sys
import io
import contextlib
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ase import units, Atoms
from ase.build import bulk, molecule
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.data import chemical_symbols
from ase.collections import g2
from ase.thermochemistry import CrystalThermo, IdealGasThermo
from ase.vibrations import Vibrations

from mattersim.forcefield import MatterSimCalculator
from mattersim.applications.phonon import PhononWorkflow
from mattersim.applications.relax import Relaxer

class MatterSimApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MatterSim AI Materials Lab")
        self.geometry("1280x900")
        # Tk variables
        self.md_engine = tk.StringVar(value="ase")
        self.num_gpus = tk.IntVar(value=1)
        self.verbose = tk.BooleanVar(value=True)
        self.selection_type = tk.StringVar(value="element")
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._safe_exit)

    def _build_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Performance Panel
        perf_frame = ttk.LabelFrame(main_frame, text="Computation Settings")
        perf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(perf_frame, text="MD Engine:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(perf_frame, text="ASE", variable=self.md_engine, value="ase").pack(side=tk.LEFT, padx=5)
        ttk.Label(perf_frame, text="GPUs:").pack(side=tk.LEFT, padx=10)
        ttk.Spinbox(perf_frame, from_=0, to=8, textvariable=self.num_gpus, width=4).pack(side=tk.LEFT, padx=5)
        ttk.Label(perf_frame, text="Verbose:").pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(perf_frame, variable=self.verbose).pack(side=tk.LEFT, padx=5)

        # Selection Panel
        sel_frame = ttk.LabelFrame(main_frame, text="Material Selection")
        sel_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sel_frame, text="Compute on:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(sel_frame, text="Element", variable=self.selection_type, value="element", command=self._update_selection_widgets).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Radiobutton(sel_frame, text="Molecule", variable=self.selection_type, value="molecule", command=self._update_selection_widgets).grid(row=0, column=2, sticky="w", padx=5)
        ttk.Radiobutton(sel_frame, text="Compound", variable=self.selection_type, value="compound", command=self._update_selection_widgets).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(sel_frame, text="Element:").grid(row=1, column=0, sticky="w", pady=5)
        element_list = [sym for sym in chemical_symbols[1:93]]
        self.element_cb = ttk.Combobox(sel_frame, values=element_list, state="readonly")
        self.element_cb.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5)
        self.element_cb.set("Fe")
        ttk.Label(sel_frame, text="Structure:").grid(row=1, column=3, sticky="w")
        self.structure_cb = ttk.Combobox(sel_frame, values=["bcc", "fcc", "hcp"], state="readonly")
        self.structure_cb.grid(row=1, column=4, sticky="ew", padx=5)
        self.structure_cb.set("bcc")
        ttk.Label(sel_frame, text="Molecule:").grid(row=2, column=0, sticky="w", pady=5)
        mol_list = sorted(list(g2.names))
        self.molecule_cb = ttk.Combobox(sel_frame, values=mol_list, state="disabled")
        self.molecule_cb.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5)
        self.molecule_cb.set("H2O")
        ttk.Label(sel_frame, text="Compound formula:").grid(row=3, column=0, sticky="w", pady=5)
        self.formula_entry = ttk.Entry(sel_frame, width=20, state="disabled")
        self.formula_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Label(sel_frame, text="Calculation Mode:").grid(row=2, column=3, sticky="w")
        self.mode_cb = ttk.Combobox(sel_frame, values=[
            "Equilibrium Scan", "Relaxation", "Phonon", "Molecular Dynamics", "Phase Diagram", "Thermodynamics"
        ], state="readonly")
        self.mode_cb.grid(row=2, column=4, sticky="ew", padx=5)
        self.mode_cb.set("Equilibrium Scan")
        self.run_btn = ttk.Button(sel_frame, text="▶ Run", command=self._on_run)
        self.run_btn.grid(row=0, column=5, rowspan=4, padx=10, sticky="ns")
        for c in range(6):
            sel_frame.grid_columnconfigure(c, weight=0)
        sel_frame.grid_columnconfigure(1, weight=1)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.tabs = {}
        for name in [
            "Equilibrium Scan", "Relaxation", "Phonon", "Molecular Dynamics", "Phase Diagram", "Thermodynamics"
        ]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            self.notebook.hide(frame)
            self.tabs[name] = frame
        self._build_equilibrium_tab()
        self._build_relaxation_tab()
        self._build_phonon_tab()
        self._build_md_tab()
        self._build_phase_tab()
        self._build_thermo_tab()
        self.notebook.select(self.tabs["Equilibrium Scan"])
        self.notebook.tab(self.tabs["Equilibrium Scan"], state="normal")
        self._update_selection_widgets()

    def _update_selection_widgets(self):
        mode = self.selection_type.get()
        if mode == "element":
            self.element_cb.configure(state="readonly")
            self.structure_cb.configure(state="readonly")
            self.molecule_cb.configure(state="disabled")
            self.formula_entry.configure(state="disabled")
        elif mode == "molecule":
            self.element_cb.configure(state="disabled")
            self.structure_cb.configure(state="disabled")
            self.molecule_cb.configure(state="readonly")
            self.formula_entry.configure(state="disabled")
        else:
            self.element_cb.configure(state="disabled")
            self.structure_cb.configure(state="disabled")
            self.molecule_cb.configure(state="disabled")
            self.formula_entry.configure(state="normal")

    # --- Tabs ---
    def _build_equilibrium_tab(self):
        frm = self.tabs["Equilibrium Scan"]
        ttk.Label(frm, text="± Scan around guess (%) :").pack(anchor="w", padx=5, pady=(5, 0))
        self.equi_pct = ttk.Entry(frm, width=10)
        self.equi_pct.pack(anchor="w", padx=5, pady=(0, 5))
        self.equi_pct.insert(0, "10")
        ttk.Label(frm, text="Number of points:").pack(anchor="w", padx=5)
        self.equi_npts = ttk.Entry(frm, width=10)
        self.equi_npts.pack(anchor="w", padx=5, pady=(0, 5))
        self.equi_npts.insert(0, "20")
        self.equi_fig = plt.Figure(figsize=(6, 4))
        self.equi_ax = self.equi_fig.add_subplot(111)
        self.equi_canvas = FigureCanvasTkAgg(self.equi_fig, master=frm)
        self.equi_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_relaxation_tab(self):
        frm = self.tabs["Relaxation"]
        ttk.Label(frm, text="Relaxation Log:").pack(anchor="w", padx=5, pady=(5, 0))
        self.relax_log = ScrolledText(frm, height=15)
        self.relax_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_phonon_tab(self):
        frm = self.tabs["Phonon"]
        ttk.Label(frm, text="Phonon band structure:").pack(anchor="w", padx=5, pady=(5, 0))
        self.phonon_frame = ttk.Frame(frm)
        self.phonon_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_md_tab(self):
        frm = self.tabs["Molecular Dynamics"]
        inner = ttk.Frame(frm)
        inner.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(inner, text="Temperature (K):").grid(row=0, column=0, sticky="w")
        self.md_temp = ttk.Entry(inner, width=10)
        self.md_temp.grid(row=0, column=1, padx=5, sticky="w")
        self.md_temp.insert(0, "300")
        ttk.Label(inner, text="Steps:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.md_steps = ttk.Entry(inner, width=10)
        self.md_steps.grid(row=1, column=1, padx=5, pady=(5, 0), sticky="w")
        self.md_steps.insert(0, "100")
        self.md_fig = plt.Figure(figsize=(6, 4))
        self.md_ax = self.md_fig.add_subplot(111)
        self.md_canvas = FigureCanvasTkAgg(self.md_fig, master=frm)
        self.md_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_phase_tab(self):
        frm = self.tabs["Phase Diagram"]
        inner = ttk.Frame(frm)
        inner.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(inner, text="Phase 1:").grid(row=0, column=0, sticky="w")
        self.pg_phase1_cb = ttk.Combobox(inner, values=["bcc", "fcc", "hcp"], width=8, state="readonly")
        self.pg_phase1_cb.grid(row=0, column=1, padx=5, sticky="w")
        self.pg_phase1_cb.set("bcc")
        ttk.Label(inner, text="Phase 2:").grid(row=0, column=2, sticky="w")
        self.pg_phase2_cb = ttk.Combobox(inner, values=["bcc", "fcc", "hcp"], width=8, state="readonly")
        self.pg_phase2_cb.grid(row=0, column=3, padx=5, sticky="w")
        self.pg_phase2_cb.set("fcc")
        ttk.Label(inner, text="T Min (K):").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.pg_tmin = ttk.Entry(inner, width=10)
        self.pg_tmin.grid(row=1, column=1, padx=5, pady=(5, 0), sticky="w")
        self.pg_tmin.insert(0, "0")
        ttk.Label(inner, text="T Max (K):").grid(row=1, column=2, sticky="w", pady=(5, 0))
        self.pg_tmax = ttk.Entry(inner, width=10)
        self.pg_tmax.grid(row=1, column=3, padx=5, pady=(5, 0), sticky="w")
        self.pg_tmax.insert(0, "2000")
        ttk.Label(inner, text="P Min (GPa):").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.pg_pmin = ttk.Entry(inner, width=10)
        self.pg_pmin.grid(row=2, column=1, padx=5, pady=(5, 0), sticky="w")
        self.pg_pmin.insert(0, "0")
        ttk.Label(inner, text="P Max (GPa):").grid(row=2, column=2, sticky="w", pady=(5, 0))
        self.pg_pmax = ttk.Entry(inner, width=10)
        self.pg_pmax.grid(row=2, column=3, padx=5, pady=(5, 0), sticky="w")
        self.pg_pmax.insert(0, "10")
        self.phase_fig = plt.Figure(figsize=(6, 4))
        self.phase_ax = self.phase_fig.add_subplot(111)
        self.phase_canvas = FigureCanvasTkAgg(self.phase_fig, master=frm)
        self.phase_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_thermo_tab(self):
        frm = self.tabs["Thermodynamics"]
        self.thermo_fig = plt.Figure(figsize=(8, 6))
        self.thermo_ax = self.thermo_fig.add_subplot(111)
        self.thermo_canvas = FigureCanvasTkAgg(self.thermo_fig, master=frm)
        self.thermo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # --- Main Run Handler ---
    def _on_run(self):
        self.run_btn.configure(state="disabled")
        mode = self.mode_cb.get()
        selection = self.selection_type.get()
        # Build atoms object
        if selection == "element":
            element = self.element_cb.get()
            phase = self.structure_cb.get()
            atoms = bulk(element, phase, a=4.0)
            formula = element
        elif selection == "molecule":
            molecule_name = self.molecule_cb.get()
            atoms = molecule(molecule_name)
            element = molecule_name
            phase = "molecule"
            formula = molecule_name
        else:
            formula = self.formula_entry.get().strip()
            try:
                atoms = Atoms(formula)
                element = formula
                phase = "compound"
            except Exception as e:
                messagebox.showerror("Error", f"Could not parse formula: {e}")
                self.run_btn.configure(state="normal")
                return
        # Show correct tab
        for name, tab in self.tabs.items():
            self.notebook.hide(tab)
        self.notebook.tab(self.tabs[mode], state="normal")
        self.notebook.select(self.tabs[mode])
        # Launch calculation
        if mode == "Equilibrium Scan":
            threading.Thread(target=self._compute_equilibrium, args=(atoms, element, phase), daemon=True).start()
        elif mode == "Relaxation":
            threading.Thread(target=self._compute_relaxation, args=(atoms, element, phase), daemon=True).start()
        elif mode == "Phonon":
            threading.Thread(target=self._compute_phonon, args=(atoms, element, phase), daemon=True).start()
        elif mode == "Molecular Dynamics":
            threading.Thread(target=self._compute_md, args=(atoms, element, phase), daemon=True).start()
        elif mode == "Phase Diagram":
            threading.Thread(target=self._compute_phase_diagram, args=(atoms, element, phase), daemon=True).start()
        elif mode == "Thermodynamics":
            threading.Thread(target=self._compute_thermodynamics, args=(atoms, element, phase, formula), daemon=True).start()
        else:
            messagebox.showerror("Mode Error", "Unknown calculation mode.")
            self.run_btn.configure(state="normal")

    # --- Calculation Methods ---

    def _compute_equilibrium(self, atoms, element, phase):
        try:
            pct = float(self.equi_pct.get()) / 100.0
            npts = int(self.equi_npts.get())
            a0 = atoms.get_cell_lengths_and_angles()[0]
            a_vals = np.linspace(a0 * (1 - pct), a0 * (1 + pct), npts)
            energies = np.zeros_like(a_vals)
            for i, a in enumerate(a_vals):
                atoms.set_cell([a, a, a], scale_atoms=True)
                atoms.calc = MatterSimCalculator()
                energies[i] = atoms.get_potential_energy()
            self.equi_ax.cla()
            self.equi_ax.plot(a_vals, energies, "o-", label=f"{element} ({phase})")
            self.equi_ax.set_xlabel("Lattice constant a (Å)")
            self.equi_ax.set_ylabel("Total Energy (eV)")
            self.equi_ax.set_title(f"Energy vs. a — {element} ({phase})")
            self.equi_ax.legend()
            self.equi_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self.run_btn.configure(state="normal")

    def _compute_relaxation(self, atoms, element, phase):
        self.relax_log.delete('1.0', tk.END)
        try:
            relaxer = Relaxer(calculator=MatterSimCalculator())
            with contextlib.redirect_stdout(io.StringIO() if not self.verbose.get() else sys.stdout):
                results = relaxer.relax(atoms, fmax=0.01, steps=500, verbose=self.verbose.get())
            self.relax_log.insert(tk.END, f"Relaxation completed in {results['steps']} steps\n")
            self.relax_log.insert(tk.END, f"Final energy: {results['energy']:.4f} eV\n")
            self.relax_log.insert(tk.END, f"Final forces max: {results['fmax']:.4f} eV/Å\n")
        except Exception as e:
            self.relax_log.insert(tk.END, f"Relaxation failed: {str(e)}\n")
        self.run_btn.configure(state="normal")

    def _compute_phonon(self, atoms, element, phase):
        for widget in self.phonon_frame.winfo_children():
            widget.destroy()
        work_dir = f"./phonon_{element}_{phase}"
        os.makedirs(work_dir, exist_ok=True)
        try:
            atoms.calc = MatterSimCalculator()
            workflow = PhononWorkflow(atoms, work_dir=work_dir)
            has_imag, phon = workflow.run()
            if not has_imag:
                messagebox.showinfo("Phonon", "No imaginary frequencies found.")
            try:
                fig = phon.plot_band_structure()
            except AttributeError:
                fig = phon.plot_band()
            plot_path = os.path.join(work_dir, "phonon_band.png")
            fig.savefig(plot_path)
            messagebox.showinfo("Phonon", f"Band structure plot saved as:\n{plot_path}")
            canvas = FigureCanvasTkAgg(fig, master=self.phonon_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Phonon workflow failed:\n{e}")
        self.run_btn.configure(state="normal")

    def _compute_md(self, atoms, element, phase):
        T = float(self.md_temp.get())
        steps = int(self.md_steps.get())
        self.md_ax.cla()
        try:
            atoms.calc = MatterSimCalculator(use_gpu=self.num_gpus.get() > 0)
            dyn = Langevin(atoms, timestep=1 * units.fs, temperature_K=T, friction=0.002, logfile=None)
            energies = []
            temps = []
            def log_step():
                energies.append(atoms.get_total_energy())
                temps.append(atoms.get_temperature())
            dyn.attach(log_step, interval=10)
            for _ in range(steps // 10):
                dyn.run(10)
            self.md_ax.plot(np.arange(len(energies)), energies, 'b-', label='Energy (eV)')
            self.md_ax.set_xlabel('MD Steps')
            self.md_ax.set_ylabel('Energy (eV)', color='b')
            ax2 = self.md_ax.twinx()
            ax2.plot(np.arange(len(temps)), temps, 'r--', label='Temperature (K)')
            ax2.set_ylabel('Temperature (K)', color='r')
            self.md_ax.set_title(f"{element} {phase} Molecular Dynamics")
            self.md_canvas.draw()
        except Exception as e:
            messagebox.showerror("MD Error", str(e))
        self.run_btn.configure(state="normal")

    def _compute_phase_diagram(self, atoms, element, phase):
        try:
            phase1 = self.pg_phase1_cb.get()
            phase2 = self.pg_phase2_cb.get()
            Tmin = float(self.pg_tmin.get())
            Tmax = float(self.pg_tmax.get())
            T = np.linspace(Tmin, Tmax, 50)
            P = np.linspace(0, 10, 50)
            atoms1 = bulk(element, phase1)
            atoms1.calc = MatterSimCalculator()
            E1 = atoms1.get_potential_energy()
            V1 = atoms1.get_volume()
            atoms2 = bulk(element, phase2)
            atoms2.calc = MatterSimCalculator()
            E2 = atoms2.get_potential_energy()
            V2 = atoms2.get_volume()
            boundary_T = []
            boundary_P = []
            for T_val in T:
                for P_val in P:
                    G1 = E1 + P_val * 1e9 * V1 * 1e-30
                    G2 = E2 + P_val * 1e9 * V2 * 1e-30
                    if abs(G1 - G2) < 1e-2:
                        boundary_T.append(T_val)
                        boundary_P.append(P_val)
            self.phase_ax.cla()
            self.phase_ax.scatter(boundary_T, boundary_P, c="k", s=10, label="Phase Boundary")
            self.phase_ax.set_xlabel("Temperature (K)")
            self.phase_ax.set_ylabel("Pressure (GPa)")
            self.phase_ax.set_title(f"Phase Diagram: {phase1} vs {phase2} ({element})")
            self.phase_ax.legend()
            self.phase_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Phase diagram computation failed:\n{e}")
        self.run_btn.configure(state="normal")

    def _compute_thermodynamics(self, atoms, element, phase, formula):
        self.thermo_ax.cla()
        try:
            atoms.calc = MatterSimCalculator()
            if phase in ["bcc", "fcc", "hcp"]:
                # Crystal
                thermo = CrystalThermo(atoms, potentialenergy=atoms.get_potential_energy())
                T = np.linspace(100, 2000, 50)
                G = [thermo.get_gibbs_energy(temp, pressure=0) for temp in T]
                S = [thermo.get_entropy(temp, pressure=0) for temp in T]
                Cp = [thermo.get_heat_capacity(temp, pressure=0) for temp in T]
                self.thermo_ax.plot(T, G, label="Gibbs Free Energy (eV)")
                self.thermo_ax.plot(T, S, label="Entropy (eV/K)")
                self.thermo_ax.plot(T, Cp, label="Heat Capacity (eV/K)")
                self.thermo_ax.set_xlabel("Temperature (K)")
                self.thermo_ax.set_title(f"Thermodynamic Properties: {element} ({phase})")
                self.thermo_ax.legend()
                self.thermo_canvas.draw()
            else:
                # Molecule or arbitrary compound: Use IdealGasThermo if possible
                vib = Vibrations(atoms)
                vib.run()
                frequencies = vib.get_frequencies()
                thermo = IdealGasThermo(
                    vib_energies=frequencies * units._h * units._c,
                    potentialenergy=atoms.get_potential_energy(),
                    atoms=atoms,
                    geometry='nonlinear' if len(atoms) > 2 else 'linear',
                    symmetrynumber=1,
                    spin=0
                )
                T = np.linspace(100, 2000, 50)
                G = [thermo.get_gibbs_energy(temp, pressure=101325) for temp in T]
                S = [thermo.get_entropy(temp, pressure=101325) for temp in T]
                Cp = [thermo.get_heat_capacity(temp, pressure=101325) for temp in T]
                self.thermo_ax.plot(T, G, label="Gibbs Free Energy (eV)")
                self.thermo_ax.plot(T, S, label="Entropy (eV/K)")
                self.thermo_ax.plot(T, Cp, label="Heat Capacity (eV/K)")
                self.thermo_ax.set_xlabel("Temperature (K)")
                self.thermo_ax.set_title(f"Thermodynamic Properties: {formula}")
                self.thermo_ax.legend()
                self.thermo_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Thermodynamic calculation failed:\n{e}")
        self.run_btn.configure(state="normal")

    def _safe_exit(self):
        if threading.active_count() > 1:
            if messagebox.askokcancel("Quit", "Calculations running! Force quit?"):
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    app = MatterSimApp()
    app.mainloop()
