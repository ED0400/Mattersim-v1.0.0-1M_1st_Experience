import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from ase.build import bulk, molecule
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator
from mattersim.applications.phonon import PhononWorkflow

# -----------------------------
# Element Data (lattice, expansion, bulk modulus)
# -----------------------------
ELEMENT_DATA = {
    "Si": ("Silicon", {"diamond": 5.43}, 2.6e-6, 98),
    "C":  ("Carbon",  {"diamond": 3.57}, 1.0e-6, 442),
    "Al": ("Aluminum", {"fcc": 4.05},   24e-6, 70),
    "Fe": ("Iron",    {"bcc": 2.87},    11.8e-6, 170),
    "Cu": ("Copper",   {"fcc": 3.61},   16.5e-6, 140),
    "Na": ("Sodium",   {"bcc": 4.23},   71e-6, 7),
    "Mg": ("Magnesium", {"hcp": 3.21},  25e-6, 45),
}

T_REF = 300  # reference temperature


# -----------------------------
# Compute lattice correction
# -----------------------------
def apply_temp_pressure_lattice(a0, T, P, alpha, bulkmodulus):
    a_temp = a0 * (1 + alpha * (T - T_REF))
    a_corr = a_temp * (1 - P / (3 * bulkmodulus))
    return a_corr


# -----------------------------
# Relaxation (energy/force/stress)
# -----------------------------
def run_relaxation(atoms, temperature, pressure, device):
    calc = MatterSimCalculator(device=device)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    try:
        stress = atoms.get_stress(voigt=False)
    except:
        stress = None

    return {"energy": energy, "forces": forces, "stress": stress}


# -----------------------------
# Phonon simulation
# -----------------------------
def run_phonons(atoms, workdir, amp, sc, qmesh):
    atoms.calc = MatterSimCalculator()

    workflow = PhononWorkflow(
        atoms=atoms,
        work_dir=workdir,
        amplitude=amp,
        supercell_matrix=np.diag([sc, sc, sc]),
        qpoints_mesh=np.array([qmesh, qmesh, qmesh]),
        find_prim=True,
    )

    has_imag, phonon = workflow.run()
    return has_imag, phonon


# -----------------------------
# Simulation Runner
# -----------------------------
def run_simulation():
    try:
        mode = sim_mode.get()
        output_text.delete("1.0", tk.END)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------
        # BULK RELAX
        # -----------------------------------------
        if mode == "Bulk Relax":
            elt = element_var.get()
            struct = structure_var.get()

            if not elt or not struct:
                raise ValueError("Select element and structure.")

            # Read temperature & pressure
            T = float(temp_var.get())
            P = float(press_var.get())

            # Lattice
            a0 = ELEMENT_DATA[elt][1][struct]
            alpha = ELEMENT_DATA[elt][2]
            K = ELEMENT_DATA[elt][3]

            a_corr = apply_temp_pressure_lattice(a0, T, P, alpha, K)
            lattice_var.set(f"{a_corr:.4f}")

            atoms = bulk(elt, struct, a=a_corr)

            result = run_relaxation(atoms, T, P, device)

            output_text.insert(tk.END, f"Using default lattice: {a_corr:.4f} Å for {elt} ({struct})\n")
            output_text.insert(tk.END, f"Energy: {result['energy']:.4f} eV\n")
            output_text.insert(tk.END, "Forces (eV/Å):\n")
            output_text.insert(tk.END, f"{np.array2string(result['forces'], precision=4)}\n")

            if result["stress"] is not None:
                output_text.insert(tk.END, "Stress (eV/Å³):\n")
                output_text.insert(tk.END, f"{np.array2string(result['stress'], precision=4)}\n")
                output_text.insert(tk.END, "Stress (GPa):\n")
                output_text.insert(tk.END, f"{np.array2string(result['stress']/GPa, precision=4)}\n")

        # -----------------------------------------
        # MOLECULE RELAX
        # -----------------------------------------
        elif mode == "Molecule Relax":
            mol = molecule_var.get().strip()
            if not mol:
                raise ValueError("Enter molecule formula.")

            atoms = molecule(mol)

            T = float(temp_var.get())
            P = float(press_var.get())

            result = run_relaxation(atoms, T, P, device)

            output_text.insert(tk.END, f"Molecule: {mol}\n")
            output_text.insert(tk.END, f"Energy: {result['energy']:.4f} eV\n")
            output_text.insert(tk.END, "Forces (eV/Å):\n")
            output_text.insert(tk.END, f"{np.array2string(result['forces'], precision=4)}\n")

            if result["stress"] is not None:
                output_text.insert(tk.END, "Stress (eV/Å³):\n")
                output_text.insert(tk.END, f"{np.array2string(result['stress'], precision=4)}\n")

        # -----------------------------------------
        # BULK PHONON
        # -----------------------------------------
        elif mode == "Bulk Phonon":
            elt = element_var.get()
            struct = structure_var.get()

            if not elt or not struct:
                raise ValueError("Select element & structure.")

            a0 = ELEMENT_DATA[elt][1][struct]
            atoms = bulk(elt, struct, a=a0)

            work = workdir_var.get()
            amp = float(amp_var.get())
            sc = int(sc_var.get())
            qmesh = int(qmesh_var.get())

            has_imag, phonon = run_phonons(atoms, work, amp, sc, qmesh)

            output_text.insert(tk.END, "Phonon simulation finished.\n")
            output_text.insert(tk.END, f"Imaginary modes: {has_imag}\n")

        # -----------------------------------------
        # MOLECULE PHONON
        # -----------------------------------------
        elif mode == "Molecule Phonon":
            mol = molecule_var.get().strip()
            if not mol:
                raise ValueError("Enter molecule formula.")

            atoms = molecule(mol)

            work = workdir_var.get()
            amp = float(amp_var.get())
            sc = int(sc_var.get())
            qmesh = int(qmesh_var.get())

            has_imag, phonon = run_phonons(atoms, work, amp, sc, qmesh)

            output_text.insert(tk.END, "Phonon simulation finished.\n")
            output_text.insert(tk.END, f"Imaginary modes: {has_imag}\n")

        output_text.insert(tk.END, "\nSimulation completed successfully.\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -----------------------------
# Dynamic field control
# -----------------------------
def update_fields():
    mode = sim_mode.get()

    bulk = mode.startswith("Bulk")
    mol = mode.startswith("Molecule")
    phon = "Phonon" in mode

    # Bulk fields
    element_cb.config(state="readonly" if bulk else "disabled")
    structure_cb.config(state="readonly" if bulk else "disabled")
    lattice_entry.config(state="normal" if bulk else "disabled")

    # Molecule field
    molecule_entry.config(state="normal" if mol else "disabled")

    # Phonon-related
    for widget in (workdir_entry, amp_entry, sc_entry, qmesh_entry):
        widget.config(state="normal" if phon else "disabled")


def on_element_select(event=None):
    elt = element_var.get()
    struct_map = ELEMENT_DATA.get(elt, (None, {}))[1]
    structure_cb.config(values=list(struct_map.keys()))

    if struct_map:
        key = list(struct_map.keys())[0]
        structure_var.set(key)
        lattice_var.set(str(struct_map[key]))


# -----------------------------
# GUI Layout
# -----------------------------
root = tk.Tk()
root.title("MatterSim Unified Simulator")


# Simulation Mode
tk.Label(root, text="Simulation Mode:").grid(row=0, column=0, sticky="w")
sim_mode = tk.StringVar(value="Bulk Relax")

modes = ["Bulk Relax", "Bulk Phonon", "Molecule Relax", "Molecule Phonon"]
for i, m in enumerate(modes):
    ttk.Radiobutton(root, text=m, variable=sim_mode, value=m, command=update_fields)\
        .grid(row=1, column=i, padx=4, pady=3)


# Element / Structure
tk.Label(root, text="Element:").grid(row=2, column=0, sticky="w")
element_var = tk.StringVar()
element_cb = ttk.Combobox(root, textvariable=element_var, state="readonly",
                          values=list(ELEMENT_DATA.keys()))
element_cb.grid(row=2, column=1)
element_cb.bind("<<ComboboxSelected>>", on_element_select)

tk.Label(root, text="Structure:").grid(row=3, column=0, sticky="w")
structure_var = tk.StringVar()
structure_cb = ttk.Combobox(root, textvariable=structure_var, state="disabled")
structure_cb.grid(row=3, column=1)

tk.Label(root, text="Lattice (Å):").grid(row=4, column=0, sticky="w")
lattice_var = tk.StringVar()
lattice_entry = ttk.Entry(root, textvariable=lattice_var, state="disabled")
lattice_entry.grid(row=4, column=1)


# Molecule input
tk.Label(root, text="Molecule:").grid(row=5, column=0, sticky="w")
molecule_var = tk.StringVar()
molecule_entry = ttk.Entry(root, textvariable=molecule_var, state="disabled")
molecule_entry.grid(row=5, column=1)


# Temp and Pressure
tk.Label(root, text="Temperature (K):").grid(row=6, column=0, sticky="w")
temp_var = tk.StringVar(value="300")
tk.Entry(root, textvariable=temp_var).grid(row=6, column=1)

tk.Label(root, text="Pressure (GPa):").grid(row=7, column=0, sticky="w")
press_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=press_var).grid(row=7, column=1)


# Phonon fields
tk.Label(root, text="Phonon Work Dir:").grid(row=8, column=0, sticky="w")
workdir_var = tk.StringVar(value="./phonons")
workdir_entry = ttk.Entry(root, textvariable=workdir_var, state="disabled")
workdir_entry.grid(row=8, column=1)

tk.Label(root, text="Displacement Amplitude:").grid(row=9, column=0, sticky="w")
amp_var = tk.StringVar(value="0.01")
amp_entry = ttk.Entry(root, textvariable=amp_var, state="disabled")
amp_entry.grid(row=9, column=1)

tk.Label(root, text="Supercell Size:").grid(row=10, column=0, sticky="w")
sc_var = tk.StringVar(value="3")
sc_entry = ttk.Entry(root, textvariable=sc_var, state="disabled")
sc_entry.grid(row=10, column=1)

tk.Label(root, text="Q-point Mesh:").grid(row=11, column=0, sticky="w")
qmesh_var = tk.StringVar(value="8")
qmesh_entry = ttk.Entry(root, textvariable=qmesh_var, state="disabled")
qmesh_entry.grid(row=11, column=1)


# Run Button
run_btn = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_btn.grid(row=12, column=0, columnspan=2, pady=10)


# Output box
output_text = tk.Text(root, height=20, width=80)
output_text.grid(row=13, column=0, columnspan=4, padx=5, pady=5)

update_fields()  # initial UI state

root.mainloop()
