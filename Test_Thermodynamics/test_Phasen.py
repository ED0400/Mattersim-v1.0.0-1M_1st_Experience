#!/usr/bin/env python3
"""
Standalone Phase Diagram Plotter (Fe-C and Al-Si)

- Plots approximate Fe-C and Al-Si phase diagrams that visually resemble textbook charts.
- Interactive small GUI: choose system, composition range, resolution.
- Export plotted data to Excel (.xlsx) if pandas/openpyxl are installed; otherwise export CSVs.
- Saves files into ./results/

Notes:
- Curves are semi-empirical approximations chosen to reproduce the visual features:
  Fe-C: liquidus, A3 (ferrite->austenite), A1 (eutectoid), eutectic line (liquidus at 4.3 wt% C)
  Al-Si: liquidus curve fit to points (0% Si, 660°C), (12.6% Si, 577°C), (100% Si, 1414°C)
- This is meant for visualization / educational use, not precise thermodynamic predictions.
"""

import os
from pathlib import Path
import math
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # run in VSCode/Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Try pandas/openpyxl for Excel export
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

RESULTS_DIR = Path.cwd() / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- Approximated curve functions ----------

def fe_c_liquidus(wtC: np.ndarray) -> np.ndarray:
    """Approximate Fe-C liquidus (°C) from 0 to ~6.7 wt% C.
    Linear interpolation between (0,1538) and (4.3,1147) and a gentle rise after.
    """
    wtC = np.array(wtC, dtype=float)
    # linear between 0 and 4.3
    slope = (1538.0 - 1147.0) / 4.3  # ≈ 90.93 deg per wt%
    T = 1538.0 - slope * wtC
    # for >4.3 wt% C, gently increase toward ~1320 at 6.7 wt% (cast-iron region)
    mask = wtC > 4.3
    if mask.any():
        # map [4.3, 6.7] to [1147, 1320] smoothly
        T[mask] = np.interp(wtC[mask], [4.3, 6.7], [1147.0, 1320.0])
    # avoid negatives
    return T

def fe_c_a3_line(wtC: np.ndarray) -> np.ndarray:
    """Approximate A3 (α -> γ) line (°C). For wtC <= 0.76 decreases from 912 to 727.
       For wtC > 0.76 we clamp to A1 (727°C).
    """
    wtC = np.array(wtC, dtype=float)
    T = np.full_like(wtC, 727.0)
    mask = wtC <= 0.76
    if mask.any():
        # linear from (0,912) to (0.76,727)
        T[mask] = 912.0 - ((912.0 - 727.0) / 0.76) * wtC[mask]
    return T

def fe_c_a1_line(wtC: np.ndarray) -> np.ndarray:
    """A1 (eutectoid) horizontal line at 727°C for relevant range."""
    return np.full_like(wtC, 727.0)

def fe_c_eutectic_line(wtC: np.ndarray) -> np.ndarray:
    """Eutectic temperature (1147°C) around 4.3 wt% C; plotted as horizontal at 1147."""
    return np.full_like(wtC, 1147.0)

def fe_c_cementite_line(wtC: np.ndarray) -> np.ndarray:
    """A rough estimate of Fe3C boundary line (for display only).
       We'll draw a line that separates austenite region appearance.
    """
    # For visualization: ramp from ~738°C at 0.02% C to lower values near eutectoid
    return 738.0 - 10.0 * (wtC / (0.8 + 1e-9))

# Al-Si approximations -----------------------------------------------------

def al_si_liquidus(wtSi: np.ndarray) -> np.ndarray:
    """Quadratic fit for Al-Si liquidus (°C) passing through:
       (0, 660.32), (12.6, 577), (100, 1414)
       We fit a quadratic T(x) = ax^2 + bx + c where x = wt% Si.
    """
    x = np.array(wtSi, dtype=float)
    # Fit coefficients based on points:
    # Solve for a,b,c with three points:
    pts_x = np.array([0.0, 12.6, 100.0])
    pts_T = np.array([660.32, 577.0, 1414.0])
    # compute quadratic coefficients once:
    A = np.vstack([pts_x**2, pts_x, np.ones_like(pts_x)]).T
    a, b, c = np.linalg.solve(A, pts_T)
    T = a * x**2 + b * x + c
    return T

def al_si_solidus(wtSi: np.ndarray) -> np.ndarray:
    """Approximate solidus: for low Si it's near 577°C until a moderate Si,
       then rises toward Si melting. We'll make a modest curve."""
    x = np.array(wtSi, dtype=float)
    # Use a blended curve: low-x flat at ~577 up to ~12.6 then increase
    T = np.where(x <= 12.6, 577.0, 577.0 + (1414.0 - 577.0) * ((x - 12.6) / (100.0 - 12.6))**0.9)
    return T

# ---------- plotting and GUI ----------

class PhasePlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("Phase Diagram Plotter — Fe–C & Al–Si (standalone)")

        # Controls frame
        frm = ttk.Frame(master)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="System:").grid(row=0, column=0, sticky=tk.W)
        self.system_var = tk.StringVar(value="Fe-C")
        self.system_cb = ttk.Combobox(frm, textvariable=self.system_var, values=["Fe-C", "Al-Si"], state="readonly", width=10)
        self.system_cb.grid(row=0, column=1, padx=4)
        self.system_cb.bind("<<ComboboxSelected>>", lambda e: self.on_system_change())

        ttk.Label(frm, text="Composition min:").grid(row=0, column=2, sticky=tk.W, padx=(12,0))
        self.comp_min_var = tk.DoubleVar(value=0.0)
        self.comp_min_entry = ttk.Entry(frm, textvariable=self.comp_min_var, width=8)
        self.comp_min_entry.grid(row=0, column=3, padx=4)

        ttk.Label(frm, text="Composition max:").grid(row=0, column=4, sticky=tk.W, padx=(12,0))
        self.comp_max_var = tk.DoubleVar(value=6.7)
        self.comp_max_entry = ttk.Entry(frm, textvariable=self.comp_max_var, width=8)
        self.comp_max_entry.grid(row=0, column=5, padx=4)

        ttk.Label(frm, text="Points:").grid(row=0, column=6, sticky=tk.W, padx=(12,0))
        self.res_var = tk.IntVar(value=400)
        self.res_entry = ttk.Entry(frm, textvariable=self.res_var, width=6)
        self.res_entry.grid(row=0, column=7, padx=4)

        ttk.Button(frm, text="Plot", command=self.plot).grid(row=0, column=8, padx=(12,0))
        ttk.Button(frm, text="Export Data", command=self.export_data).grid(row=0, column=9, padx=(8,0))

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(9,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Log box
        self.log = tk.Text(master, height=6)
        self.log.pack(fill=tk.BOTH, padx=8, pady=(0,8))
        self._log("Ready. Choose a system and click Plot.")

        # set UI defaults
        self.on_system_change()
        self.plotted_payload = None  # will hold arrays for export

    def on_system_change(self):
        sys = self.system_var.get()
        if sys == "Fe-C":
            self.comp_min_var.set(0.0)
            self.comp_max_var.set(6.7)
            self.res_var.set(600)
        else:
            self.comp_min_var.set(0.0)
            self.comp_max_var.set(100.0)
            self.res_var.set(600)
        self._log(f"Selected system: {sys}")

    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def plot(self):
        try:
            sys = self.system_var.get()
            cmin = float(self.comp_min_var.get())
            cmax = float(self.comp_max_var.get())
            npoints = int(self.res_var.get())
            if cmin >= cmax:
                messagebox.showerror("Invalid composition range", "min must be < max")
                return
            x = np.linspace(cmin, cmax, npoints)

            self.ax.clear()
            if sys == "Fe-C":
                # Plot Liquidus
                Tliq = fe_c_liquidus(x)
                self.ax.plot(x, Tliq, color="red", lw=2, label="Liquidus")

                # Plot A3 and A1
                Ta3 = fe_c_a3_line(x)
                self.ax.plot(x, Ta3, color="blue", ls="-", lw=1.5, label="A3 (α/γ boundary)")
                Ta1 = fe_c_a1_line(x)
                self.ax.plot(x, Ta1, color="brown", ls="--", lw=1.2, label="A1 (eutectoid)")

                # Eutectic horizontal
                x_eut = np.array([0.0, 6.7])
                Teut = fe_c_eutectic_line(x_eut)
                self.ax.plot(x_eut, Teut, color="darkred", ls=":", lw=1.5, label="Eutectic ~1147°C")

                # Cementite-ish display line
                Tcement = fe_c_cementite_line(x)
                self.ax.plot(x, Tcement, color="green", ls="-.", lw=1, label="Fe₃C boundary (schematic)")

                self.ax.set_xlim(cmin, cmax)
                self.ax.set_ylim(20, 1600)
                self.ax.set_xlabel("Carbon (wt%)")
                self.ax.set_ylabel("Temperature (°C)")
                self.ax.set_title("Approximate Fe–C Phase Diagram (schematic)")
                self.ax.legend(loc="upper right", fontsize="small")
                self.ax.grid(alpha=0.3)

                # Add vertical lines at key compositions
                self.ax.axvline(0.76, color="gray", lw=0.8, ls="--")  # eutectoid composition
                self.ax.text(0.76 + 0.05*(cmax-cmin), 740, "0.76% (eutectoid)", color="gray", fontsize=8)

                # record payload
                self.plotted_payload = {
                    "system": "Fe-C",
                    "x_wt": x.tolist(),
                    "liquidus_C": Tliq.tolist(),
                    "A3_C": Ta3.tolist(),
                    "A1_C": Ta1.tolist(),
                    "cementite_line_C": Tcement.tolist()
                }

            else:  # Al-Si
                Tliq = al_si_liquidus(x)
                Tsolid = al_si_solidus(x)
                self.ax.plot(x, Tliq, color="red", lw=2, label="Liquidus")
                self.ax.plot(x, Tsolid, color="blue", lw=1.5, label="Solidus (approx)")
                # Eutectic marker at 12.6 wt% (577°C)
                self.ax.axvline(12.6, color="gray", ls="--", lw=0.8)
                self.ax.plot([12.6], [577.0], marker="o", color="black")
                self.ax.text(12.6+0.8, 590, "Eutectic 12.6% @ 577°C", fontsize=8)
                self.ax.set_xlim(cmin, cmax)
                self.ax.set_ylim(300, 1500)
                self.ax.set_xlabel("Silicon (wt%)")
                self.ax.set_ylabel("Temperature (°C)")
                self.ax.set_title("Approximate Al–Si Phase Diagram (schematic)")
                self.ax.legend(loc="upper left", fontsize="small")
                self.ax.grid(alpha=0.3)

                self.plotted_payload = {
                    "system": "Al-Si",
                    "x_wt": x.tolist(),
                    "liquidus_C": Tliq.tolist(),
                    "solidus_C": Tsolid.tolist()
                }

            self.canvas.draw()
            self._log(f"Plotted {sys} from {cmin} to {cmax} wt%, {npoints} pts.")
        except Exception as e:
            self._log("Plot error: " + str(e))
            self._log(traceback.format_exc())
            messagebox.showerror("Plot error", str(e))

    def export_data(self):
        if not self.plotted_payload:
            messagebox.showinfo("No data", "Please Plot first, then Export.")
            return
        sys = self.plotted_payload["system"]
        fname = RESULTS_DIR / f"{sys.replace('-', '_')}_phase"
        # prefer excel
        if pd is not None:
            outpath = str(fname) + ".xlsx"
            try:
                writer = pd.ExcelWriter(outpath, engine="openpyxl")
                if sys == "Fe-C":
                    df = pd.DataFrame({
                        "wt% C": self.plotted_payload["x_wt"],
                        "Liquidus_C": self.plotted_payload["liquidus_C"],
                        "A3_C": self.plotted_payload["A3_C"],
                        "A1_C": self.plotted_payload["A1_C"],
                        "Cementite_line_C": self.plotted_payload["cementite_line_C"]
                    })
                    df.to_excel(writer, sheet_name="Fe-C_data", index=False)
                else:
                    df = pd.DataFrame({
                        "wt% Si": self.plotted_payload["x_wt"],
                        "Liquidus_C": self.plotted_payload["liquidus_C"],
                        "Solidus_C": self.plotted_payload["solidus_C"]
                    })
                    df.to_excel(writer, sheet_name="Al-Si_data", index=False)
                writer.save(); writer.close()
                self._log(f"Exported data to {outpath}")
                messagebox.showinfo("Export complete", f"Data exported to\n{outpath}")
                return
            except Exception as exc:
                self._log("Excel export failed: " + str(exc))
                self._log(traceback.format_exc())
                # fall through to CSV fallback

        # CSV fallback
        outpath = str(fname) + ".csv"
        try:
            if sys == "Fe-C":
                arr = np.vstack([
                    np.array(self.plotted_payload["x_wt"]),
                    np.array(self.plotted_payload["liquidus_C"]),
                    np.array(self.plotted_payload["A3_C"]),
                    np.array(self.plotted_payload["A1_C"]),
                    np.array(self.plotted_payload["cementite_line_C"])
                ]).T
                header = "wt%_C,liquidus_C,A3_C,A1_C,cementite_C"
                np.savetxt(outpath, arr, delimiter=",", header=header, comments="")
            else:
                arr = np.vstack([
                    np.array(self.plotted_payload["x_wt"]),
                    np.array(self.plotted_payload["liquidus_C"]),
                    np.array(self.plotted_payload["solidus_C"])
                ]).T
                header = "wt%_Si,liquidus_C,solidus_C"
                np.savetxt(outpath, arr, delimiter=",", header=header, comments="")
            self._log(f"Exported CSV to {outpath}")
            messagebox.showinfo("Export complete", f"Data exported to\n{outpath}")
        except Exception as exc:
            self._log("CSV export failed: " + str(exc))
            messagebox.showerror("Export failed", str(exc))


def main():
    root = tk.Tk()
    app = PhasePlotterApp(root)
    root.geometry("1000x800")
    root.mainloop()

if __name__ == "__main__":
    main()
