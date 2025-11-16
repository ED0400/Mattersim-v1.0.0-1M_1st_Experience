import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

# --- Element data: lattice, thermal expansion (alpha), bulk modulus (K, GPa)
ELEMENT_DATA = {
    "Si": ("Silicon", {"diamond": 5.43}, 2.6e-6, 98),
    "C":  ("Carbon",  {"diamond": 3.57}, 1.0e-6, 442),
    "Al": ("Aluminum", {"fcc": 4.05},   24e-6, 70),
    "Fe": ("Iron",    {"bcc": 2.87},    11.8e-6, 170),
    "Cu": ("Copper",   {"fcc": 3.61},   16.5e-6, 140),
    "Na": ("Sodium",   {"bcc": 4.23},   71e-6, 7),
    "Mg": ("Magnesium", {"hcp": 3.21},  25e-6, 45),
}
T_REF = 300   # Reference temperature K

def apply_temp_pressure_lattice(a0, T, P, alpha, bulkmodulus):
    # Thermal expansion
    a_temp = a0 * (1 + alpha * (T - T_REF))
    # Pressure correction
    a_corr = a_temp * (1 - P / (3 * bulkmodulus))
    return a_corr

def run_relaxation(atoms, temperature, pressure, device):
    calc = MatterSimCalculator(device=device)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    try:
        stress = atoms.get_stress(voigt=False)
    except Exception:
        stress = None
    return {"energy": energy, "forces": forces, "stress": stress}

def run_simulation():
    try:
        elt = element_var.get()
        struct = structure_var.get()
        if not elt or not struct:
            raise ValueError('Please select element and structure.')
        a0 = float(lattice_var.get()) if lattice_var.get().strip() else ELEMENT_DATA[elt][1][struct]
        T = float(temp_var.get())
        P = float(press_var.get())
        alpha = ELEMENT_DATA[elt][2]
        K = ELEMENT_DATA[elt][3]
        a_corr = apply_temp_pressure_lattice(a0, T, P, alpha, K)
        lattice_var.set(f"{a_corr:.6f}")
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, f"Lattice constant at T={T}K & P={P}GPa: {a_corr:.6f} Å\n")
        atoms = bulk(elt, struct, a=a_corr)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        res = run_relaxation(atoms, T, P, device)
        forces_str = np.array2string(res['forces'], precision=4)
        output_text.insert(tk.END, f"Energy: {res['energy']:.4f} eV\nForces:\n{forces_str}\n")
        if res['stress'] is not None:
            s = res['stress']; sg = s / GPa
            output_text.insert(tk.END, "Stress (eV/Å³):\n" + np.array2string(s, precision=4) + "\n")
            output_text.insert(tk.END, "Stress (GPa):\n" + np.array2string(sg, precision=4) + "\n")
        output_text.insert(tk.END, "\nSimulation completed successfully.\n")
    except Exception as e:
        messagebox.showerror('Error', str(e))

def on_element_select(event=None):
    elt = element_var.get()
    struct_map = ELEMENT_DATA.get(elt, (None, {}))[1]
    struct_keys = list(struct_map.keys())
    structure_cb.config(values=struct_keys)
    # Suggest first available structure and lattice constant
    if struct_keys:
        structure_var.set(struct_keys[0])
        a0 = struct_map[struct_keys[0]]
        lattice_var.set(str(a0))
    else:
        structure_var.set('')
        lattice_var.set('')

root = tk.Tk()
root.title('MatterSim Unified Simulator')

tk.Label(root, text='Element:').grid(column=0, row=0, sticky=tk.W)
element_var = tk.StringVar()
element_cb = ttk.Combobox(root, textvariable=element_var, values=list(ELEMENT_DATA.keys()), state='readonly')
element_cb.grid(column=1, row=0)
element_cb.bind('<<ComboboxSelected>>', on_element_select)

tk.Label(root, text='Structure:').grid(column=0, row=1, sticky=tk.W)
structure_var = tk.StringVar()
structure_cb = ttk.Combobox(root, textvariable=structure_var, values=[], state='readonly')
structure_cb.grid(column=1, row=1)

tk.Label(root, text='Lattice (Å):').grid(column=0, row=2, sticky=tk.W)
lattice_var = tk.StringVar()
lattice_entry = ttk.Entry(root, textvariable=lattice_var)
lattice_entry.grid(column=1, row=2)

tk.Label(root, text='Temperature (K):').grid(column=0, row=3, sticky=tk.W)
temp_var = tk.StringVar(value='300')
temp_entry = ttk.Entry(root, textvariable=temp_var)
temp_entry.grid(column=1, row=3)

tk.Label(root, text='Pressure (GPa):').grid(column=0, row=4, sticky=tk.W)
press_var = tk.StringVar(value='0')
press_entry = ttk.Entry(root, textvariable=press_var)
press_entry.grid(column=1, row=4)

run_btn = ttk.Button(root, text='Run Simulation', command=run_simulation)
run_btn.grid(column=0, row=5, columnspan=2, pady=10)

output_text = tk.Text(root, height=15, width=60)
output_text.grid(column=0, row=6, columnspan=4, padx=5, pady=5)

root.mainloop()
