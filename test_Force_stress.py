import tkinter as tk 
from tkinter import ttk, messagebox
import numpy as np
import torch
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
    """
    Run phonon workflow with correct keyword arguments to ensure parameters map properly.
    """
    # Ensure atoms has a calculator for force computations
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

# --- GUI Callback ---
def run_simulation():
    try:
        mode = sim_mode.get()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_text.delete('1.0', tk.END)

        # Build atoms object
        if mode in ('Bulk Relax', 'Bulk Phonon'):
            elt = element_var.get()
            struct = structure_var.get()
            if not elt or not struct:
                raise ValueError('Please select element and structure for bulk simulation.')
            a_val = lattice_var.get().strip()
            if a_val:
                a = float(a_val)
            else:
                _, struct_map = ELEMENT_DATA[elt]
                a = struct_map.get(struct)
                lattice_var.set(str(a))
                output_text.insert(tk.END, f"Using default lattice: {a} Å for {elt} ({struct})\n")
            atoms = bulk(elt, struct, a=a)
        else:
            mol = molecule_var.get().strip()
            if not mol:
                raise ValueError('Please enter molecule formula for molecule simulation.')
            atoms = molecule(mol)

        T = float(temp_var.get())
        P = float(press_var.get())

        if mode.endswith('Relax'):
            res = run_relaxation(atoms, T, P, device)
            forces_str = np.array2string(res['forces'], precision=4)
            output_text.insert(tk.END, f"Energy: {res['energy']:.4f} eV\nForces:\n{forces_str}\n")
            if res['stress'] is not None:
                s = res['stress']; sg = s / GPa
                output_text.insert(tk.END, "Stress (eV/Å³):\n" + np.array2string(s, precision=4) + "\n")
                output_text.insert(tk.END, "Stress (GPa):\n" + np.array2string(sg, precision=4) + "\n")
        else:
            # Phonon parameters
            wdir = phonon_dir_var.get()
            amp = float(amp_var.get())
            sc = int(sc_var.get())
            qp = int(qp_var.get())
            prim = prim_var.get()
            sc_mat = np.diag([sc]*3)
            qp_mesh = np.array([qp]*3)
            has_im, phonon = run_phonon(atoms, wdir, amp, sc_mat, qp_mesh, prim)
            output_text.insert(tk.END, f"Imaginary modes present: {has_im}\n")

        output_text.insert(tk.END, "\nSimulation completed successfully.\n")
    except Exception as e:
        messagebox.showerror('Error', str(e))

# --- Field Updaters ---
def update_fields():
    mode = sim_mode.get()
    bulk_active = mode.startswith('Bulk')
    phonon_active = 'Phonon' in mode
    molecule_active = mode.startswith('Molecule')

    element_cb.config(state='readonly' if bulk_active else 'disabled')
    structure_cb.config(state='readonly' if bulk_active else 'disabled')
    lattice_entry.config(state='normal' if bulk_active else 'disabled')
    molecule_entry.config(state='normal' if molecule_active else 'disabled')

    for widget in (phonon_dir_entry, amp_entry, sc_entry, qp_entry, prim_check):
        widget.config(state='normal' if phonon_active else 'disabled')

    # Populate phonon defaults when phonon active
    if phonon_active:
        key = element_var.get() if mode.endswith('Phonon') and element_var.get() else '_molecule_'
        defaults = PHONON_DEFAULTS.get(key, PHONON_DEFAULTS['_molecule_'])
        phonon_dir_var.set(defaults['work_dir'])
        amp_var.set(str(defaults['amplitude']))
        sc_var.set(str(defaults['supercell']))
        qp_var.set(str(defaults['qmesh']))

# --- Callback for element selection to populate structure choices ---
def on_element_select(event=None):
    elt = element_var.get()
    struct_map = ELEMENT_DATA.get(elt, (None, {}))[1]
    structure_cb.config(values=list(struct_map.keys()))
    structure_var.set('')
    lattice_var.set('')
    update_fields()

# --- GUI Setup ---
root = tk.Tk()
root.title('MatterSim Unified Simulator')

# Simulation mode
tk.Label(root, text='Simulation Mode:').grid(column=0, row=0, sticky=tk.W)
sim_mode = tk.StringVar(value='Bulk Relax')
modes = ['Bulk Relax', 'Bulk Phonon', 'Molecule Relax', 'Molecule Phonon']
for i, m in enumerate(modes):
    ttk.Radiobutton(root, text=m, variable=sim_mode, value=m, command=update_fields).grid(column=i, row=1, padx=5, pady=5)

# Bulk inputs
ttk.Label(root, text='Element:').grid(column=0, row=2, sticky=tk.W)
element_var = tk.StringVar()
element_cb = ttk.Combobox(root, textvariable=element_var, values=list(ELEMENT_DATA.keys()), state='readonly')
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

# Temp & Pressure
ttk.Label(root, text='Temperature (K):').grid(column=0, row=6, sticky=tk.W)
temp_var = tk.StringVar(value='300')
temp_entry = ttk.Entry(root, textvariable=temp_var)
temp_entry.grid(column=1, row=6)

ttk.Label(root, text='Pressure (GPa):').grid(column=0, row=7, sticky=tk.W)
press_var = tk.StringVar(value='0')
press_entry = ttk.Entry(root, textvariable=press_var)
press_entry.grid(column=1, row=7)

# Phonon settings
ttk.Label(root, text='Phonon Work Dir:').grid(column=0, row=8, sticky=tk.W)
phonon_dir_var = tk.StringVar()
phonon_dir_entry = ttk.Entry(root, textvariable=phonon_dir_var, state='disabled')
phonon_dir_entry.grid(column=1, row=8)

ttk.Label(root, text='Displacement Amplitude:').grid(column=0, row=9, sticky=tk.W)
amp_var = tk.StringVar()
amp_entry = ttk.Entry(root, textvariable=amp_var, state='disabled')
amp_entry.grid(column=1, row=9)

ttk.Label(root, text='Supercell Size:').grid(column=0, row=10, sticky=tk.W)
sc_var = tk.StringVar()
sc_entry = ttk.Entry(root, textvariable=sc_var, state='disabled')
sc_entry.grid(column=1, row=10)

ttk.Label(root, text='Q-point Mesh:').grid(column=0, row=11, sticky=tk.W)
qp_var = tk.StringVar()
qp_entry = ttk.Entry(root, textvariable=qp_var, state='disabled')
qp_entry.grid(column=1, row=11)

prim_var = tk.BooleanVar()
prim_check = ttk.Checkbutton(root, text='Find Primitive', variable=prim_var, state='disabled')
prim_check.grid(column=0, row=12, columnspan=2)

# Run & Output
run_btn = ttk.Button(root, text='Run Simulation', command=run_simulation)
run_btn.grid(column=0, row=13, columnspan=2, pady=10)

output_text = tk.Text(root, height=15, width=60)
output_text.grid(column=0, row=14, columnspan=4, padx=5, pady=5)

update_fields()
root.mainloop()

