import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import matplotlib.pyplot as plt
from mendeleev import element
from thermo import Chemical
from ase.build import bulk, molecule
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator
from mattersim.applications.phonon import PhononWorkflow

# --- Element Data: full names and lattice constants by structure ---
ELEMENT_DATA = {
    "Si": ("Silicon", {"diamond": 5.43}),
    "C":  ("Carbon",  {"diamond": 3.57}),
    "Al": ("Aluminum", {"fcc": 4.05}),
    "Fe": ("Iron",    {"bcc": 2.87}),
    "Cu": ("Copper",   {"fcc": 3.61}),
    "Na": ("Sodium",   {"bcc": 4.23}),
    "Mg": ("Magnesium", {"hcp": 3.21}),
}

# --- Phonon default parameters per element/molecule ---
PHONON_DEFAULTS = {
    "Si": {"work_dir": "/tmp/phonon_Si", "amplitude": 0.01, "supercell": 4, "qmesh": 12},
    "C":  {"work_dir": "/tmp/phonon_C",  "amplitude": 0.02, "supercell": 3, "qmesh": 9},
    "Al": {"work_dir": "/tmp/phonon_Al", "amplitude": 0.015, "supercell": 4, "qmesh": 10},
    "_molecule_": {"work_dir": "/tmp/phonon_mol", "amplitude": 0.02, "supercell": 2, "qmesh": 6},
}

# --- Simulation Handlers ---
def run_relaxation(atoms, temperature, pressure, device):
    calc = MatterSimCalculator(device=device, temperature=temperature, pressure=pressure)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    try:
        stress = atoms.get_stress(voigt=False)
    except:
        stress = None
    return {"energy": energy, "forces": forces, "stress": stress}

def run_phonon(atoms, work_dir, amplitude, supercell_matrix, qpoints_mesh, find_prim):
    atoms.calc = MatterSimCalculator()
    workflow = PhononWorkflow(
        atoms=atoms,
        work_dir=work_dir,
        supercell_matrix=supercell_matrix,
        qpoints_mesh=qpoints_mesh,
        amplitude=amplitude,
        find_prim=find_prim,
    )
    return workflow.run()

# --- Property Functions for Material Mode ---
def get_melting_point(symbol):
    el = element(symbol)
    return el.melting_point  # K

def get_heat_capacity(symbol, T):
    # thermo.Chemical.Cp is a property; loop over T
    return np.array([Chemical(symbol, T=t).Cp for t in T])  # J/kg/K

def get_conductivity(symbol, T):
    # thermo.Chemical.conductivity is a property; loop over T
    return np.array([Chemical(symbol, T=t).conductivity for t in T])  # W/m/K

def get_phase_diagram(symbol, T):
    # thermo.Chemical.Psat is saturation pressure property; loop over T
    return np.array([Chemical(symbol, T=t).Psat for t in T])  # Pa

# --- Field Updaters ---
def update_fields():
    mode = sim_mode.get()
    bulk_active = mode.startswith('Bulk')
    molecule_active = mode.startswith('Molecule')
    phonon_active = 'Phonon' in mode
    material_active = (mode == 'Material Properties')

    # Toggle widgets
    state = 'readonly' if bulk_active or material_active else 'disabled'
    element_cb.config(state=state)
    structure_cb.config(state=state)
    lattice_entry.config(state='normal' if bulk_active else 'disabled')
    molecule_entry.config(state='normal' if molecule_active else 'disabled')
    Tmin_entry.config(state='normal' if material_active else 'disabled')
    Tmax_entry.config(state='normal' if material_active else 'disabled')

    phonon_state = 'normal' if phonon_active else 'disabled'
    for w in (phonon_dir_entry, amp_entry, sc_entry, qp_entry, prim_check):
        w.config(state=phonon_state)

    # Populate phonon defaults
    if phonon_active:
        key = element_var.get() or '_molecule_'
        defaults = PHONON_DEFAULTS.get(key, PHONON_DEFAULTS['_molecule_'])
        phonon_dir_var.set(defaults['work_dir'])
        amp_var.set(str(defaults['amplitude']))
        sc_var.set(str(defaults['supercell']))
        qp_var.set(str(defaults['qmesh']))

# --- Callback for element selection ---
def on_element_select(event=None):
    elt = element_var.get()
    struct_map = ELEMENT_DATA.get(elt, (None, {}))[1]
    structure_cb.config(values=list(struct_map.keys()))
    structure_var.set('')
    lattice_var.set('')
    update_fields()

# --- GUI Callback ---
def run_simulation():
    mode = sim_mode.get()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_text.delete('1.0', tk.END)
    try:
        if mode == 'Material Properties':
            symbol = element_var.get()
            if not symbol:
                raise ValueError('Select element for material properties.')
            full, struct_map = ELEMENT_DATA.get(symbol, (None, {}))
            struct = structure_var.get() or next(iter(struct_map), '')
            lattice = struct_map.get(struct, '')
            mp = get_melting_point(symbol)

            output_text.insert(tk.END, f"Element: {full} ({symbol})\n")
            output_text.insert(tk.END, f"Structure: {struct}, Lattice: {lattice} Å\n")
            output_text.insert(tk.END, f"Melting Point: {mp:.1f} K\n\n")

            Tmin = float(Tmin_var.get()); Tmax = float(Tmax_var.get())
            T = np.linspace(Tmin, Tmax, 200)

            # Plot Heat Capacity
            Cp = get_heat_capacity(symbol, T)
            plt.figure(); plt.plot(T, Cp)
            plt.title('Heat Capacity vs Temperature')
            plt.xlabel('T (K)'); plt.ylabel('Cp (J/kg·K)')
            plt.show()

            # Plot Thermal Conductivity
            k = get_conductivity(symbol, T)
            plt.figure(); plt.plot(T, k)
            plt.title('Thermal Conductivity vs Temperature')
            plt.xlabel('T (K)'); plt.ylabel('k (W/m·K)')
            plt.show()

            # Plot Vapor Pressure (Phase Diagram)
            P = get_phase_diagram(symbol, T)
            plt.figure(); plt.semilogy(T, P)
            plt.title('Vapor Pressure vs Temperature')
            plt.xlabel('T (K)'); plt.ylabel('P (Pa)')
            plt.show()

        else:
            # Bulk or Molecule Relax/Phonon
            if mode in ('Bulk Relax', 'Bulk Phonon'):
                elt = element_var.get(); struct = structure_var.get()
                a_val = lattice_var.get().strip() or ELEMENT_DATA[elt][1].get(struct)
                atoms = bulk(elt, struct, a=float(a_val))
            else:
                mol = molecule_var.get().strip()
                atoms = molecule(mol)

            T = float(temp_var.get()); P = float(press_var.get())

            if mode.endswith('Relax'):
                res = run_relaxation(atoms, T, P, device)
                forces_str = np.array2string(res['forces'], precision=4)
                output_text.insert(
                    tk.END,
                    f"Energy: {res['energy']:.4f} eV\nForces:\n{forces_str}\n"
                )
                if res['stress'] is not None:
                    s = res['stress']; sg = s / GPa
                    output_text.insert(
                        tk.END,
                        "Stress (eV/Å³):\n" + np.array2string(s,4) + "\n"
                        "Stress (GPa):\n"  + np.array2string(sg,4) + "\n"
                    )
            else:
                wdir = phonon_dir_var.get()
                amp = float(amp_var.get()); sc = int(sc_var.get())
                qp = int(qp_var.get()); prim = prim_var.get()
                sc_mat = np.diag([sc]*3); qp_mesh = np.array([qp]*3)
                has_im, phonon = run_phonon(atoms, wdir, amp, sc_mat, qp_mesh, prim)
                output_text.insert(tk.END, f"Imaginary modes present: {has_im}\n")

        output_text.insert(tk.END, "\nDone.\n")

    except Exception as e:
        messagebox.showerror('Error', str(e))

# --- GUI Setup ---
root = tk.Tk()
root.title('MatterSim Unified Simulator')

# Mode selection
tk.Label(root, text='Mode:').grid(column=0, row=0, sticky=tk.W)
sim_mode = tk.StringVar(value='Bulk Relax')
modes = [
    'Bulk Relax', 'Bulk Phonon',
    'Molecule Relax', 'Molecule Phonon',
    'Material Properties'
]
for i, m in enumerate(modes):
    ttk.Radiobutton(root, text=m, variable=sim_mode, value=m, command=update_fields)\
        .grid(column=i, row=1, padx=5, pady=5)

# Element & structure
ttk.Label(root, text='Element:').grid(column=0, row=2, sticky=tk.W)
element_var = tk.StringVar()
element_cb = ttk.Combobox(
    root, textvariable=element_var,
    values=list(ELEMENT_DATA.keys()), state='readonly'
)
element_cb.grid(column=1, row=2)
element_cb.bind('<<ComboboxSelected>>', on_element_select)

ttk.Label(root, text='Structure:').grid(column=0, row=3, sticky=tk.W)
structure_var = tk.StringVar()
structure_cb = ttk.Combobox(root, textvariable=structure_var, values=[], state='disabled')
structure_cb.grid(column=1, row=3)

ttk.Label(root, text='Lattice (Å):').grid(column=0, row=4, sticky=tk.W)
lattice_var = tk.StringVar()
lattice_entry = ttk.Entry(root, textvariable=lattice_var, state='disabled')
lattice_entry.grid(column=1, row=4)

# Molecule input
ttk.Label(root, text='Molecule:').grid(column=0, row=5, sticky=tk.W)
molecule_var = tk.StringVar()
molecule_entry = ttk.Entry(root, textvariable=molecule_var, state='disabled')
molecule_entry.grid(column=1, row=5)

# Temp & Press
ttk.Label(root, text='Temp (K):').grid(column=0, row=6, sticky=tk.W)
temp_var = tk.StringVar(value='300')
temp_entry = ttk.Entry(root, textvariable=temp_var)
temp_entry.grid(column=1, row=6)

ttk.Label(root, text='Press (GPa):').grid(column=0, row=7, sticky=tk.W)
press_var = tk.StringVar(value='0')
press_entry = ttk.Entry(root, textvariable=press_var)
press_entry.grid(column=1, row=7)

# Phonon params
ttk.Label(root, text='Phonon Work Dir:').grid(column=0, row=8, sticky=tk.W)
phonon_dir_var = tk.StringVar()
phonon_dir_entry = ttk.Entry(root, textvariable=phonon_dir_var, state='disabled')
phonon_dir_entry.grid(column=1, row=8)

ttk.Label(root, text='Amplitude:').grid(column=0, row=9, sticky=tk.W)
amp_var = tk.StringVar()
amp_entry = ttk.Entry(root, textvariable=amp_var, state='disabled')
amp_entry.grid(column=1, row=9)

ttk.Label(root, text='Supercell:').grid(column=0, row=10, sticky=tk.W)
sc_var = tk.StringVar()
sc_entry = ttk.Entry(root, textvariable=sc_var, state='disabled')
sc_entry.grid(column=1, row=10)

ttk.Label(root, text='Q-mesh:').grid(column=0, row=11, sticky=tk.W)
qp_var = tk.StringVar()
qp_entry = ttk.Entry(root, textvariable=qp_var, state='disabled')
qp_entry.grid(column=1, row=11)

prim_var = tk.BooleanVar()
prim_check = ttk.Checkbutton(root, text='Find Primitive', variable=prim_var, state='disabled')
prim_check.grid(column=0, row=12, columnspan=2)

# Material Props inputs
ttk.Label(root, text='Tmin (K):').grid(column=2, row=6, sticky=tk.W)
Tmin_var = tk.StringVar(value='1000')
Tmin_entry = ttk.Entry(root, textvariable=Tmin_var, state='disabled')
Tmin_entry.grid(column=3, row=6)

ttk.Label(root, text='Tmax (K):').grid(column=2, row=7, sticky=tk.W)
Tmax_var = tk.StringVar(value='5000')
Tmax_entry = ttk.Entry(root, textvariable=Tmax_var, state='disabled')
Tmax_entry.grid(column=3, row=7)

# Run & output
tk.Button(root, text='Run Simulation', command=run_simulation)\
    .grid(column=0, row=13, columnspan=2, pady=10)

output_text = tk.Text(root, height=15, width=80)
output_text.grid(column=0, row=14, columnspan=4, padx=5, pady=5)

# Initialize and start
update_fields()
root.mainloop()
