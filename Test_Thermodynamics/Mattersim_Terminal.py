#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

from mattersim.forcefield import MatterSimCalculator

def simulate_melting_point(element, structure, lattice_constant,
                           temperature_range, steps_per_temp=500, timestep_fs=1.0):
    """Estimate melting by MD: avg. potential energy vs. temperature."""
    data = []
    for T in temperature_range:
        atoms = bulk(element, structure, a=lattice_constant)
        atoms.calc = MatterSimCalculator(temperature=T)
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)
        dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
        energies = []
        for _ in range(steps_per_temp):
            dyn.run(1)
            energies.append(atoms.get_potential_energy())
        data.append((T, np.mean(energies)))
    return np.array(data)

def compute_heat_capacity(temperatures, gibbs_energies):
    """Cp = –T * d²G/dT² (numerical second derivative)."""
    T = np.asarray(temperatures)
    G = np.asarray(gibbs_energies)
    d2G = np.gradient(np.gradient(G, T), T)
    return -T * d2G

def predict_thermal_conductivity(atoms_factory, temperature_range):
    """
    Placeholder κ(T) = κ0 * (T0/T). Replace with real BTE solver.
    """
    κ0, T0 = 100.0, 300.0
    return np.array([κ0 * (T0 / T) for T in temperature_range])

def construct_phase_diagram(phases, temperature_range):
    """
    Identify lowest-energy phase at each T.
    `phases` is dict[name] = (factory_function, kwargs_dict),
    where kwargs use 'name', 'crystalstructure', 'a', and optionally 'c'.
    """
    energies = {name: [] for name in phases}
    for T in temperature_range:
        for name, (factory, kw) in phases.items():
            atoms = factory(**kw)
            atoms.calc = MatterSimCalculator(temperature=T)
            energies[name].append(atoms.get_potential_energy())
    stable = []
    for i in range(len(temperature_range)):
        # pick phase with minimal energy at T[i]
        min_phase = min(phases, key=lambda n: energies[n][i])
        stable.append(min_phase)
    return stable, energies

def plot_xy(x, y, xlabel, ylabel, title, logy=False):
    plt.figure()
    if logy:
        plt.semilogy(x, y)
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    print("MatterSim: Characteristics of Matter")
    print("1) Melting Point (MD)")
    print("2) Heat Capacity (Cp from G(T))")
    print("3) Thermal Conductivity (κ(T))")
    print("4) Phase Diagram (Potential Energy)")
    choice = input("Choose [1-4]: ").strip()

    if choice == '1':
        element = input("Element symbol (e.g. Si): ").strip()
        struct  = input("Crystal structure (e.g. diamond): ").strip()
        a       = float(input("Lattice constant (Å): ").strip())
        T1, T2  = map(float, input("Tmin,Tmax (e.g. 1000,2000): ").split(','))
        temps   = np.linspace(T1, T2, 10)
        data    = simulate_melting_point(element, struct, a, temps)
        plot_xy(data[:,0], data[:,1],
                "Temperature (K)", "Avg Potential Energy (eV)",
                f"Melting Estimation: {element}")

    elif choice == '2':
        T1, T2 = map(float, input("Tmin,Tmax (e.g. 300,1000): ").split(','))
        temps   = np.linspace(T1, T2, 50)
        # Replace with real G(T) from MatterSim
        a, b, c = 0.0, -0.01, 1e-5
        G       = a + b*temps + c*temps**2
        Cp      = compute_heat_capacity(temps, G)
        plot_xy(temps, Cp,
                "Temperature (K)", "Cp (J/K)",
                "Heat Capacity vs T")

    elif choice == '3':
        from ase.build import bulk as bulk_factory
        # bulk_factory expects name, crystalstructure, a
        kw = dict(name="Si", crystalstructure="diamond", a=5.43)
        T1, T2 = map(float, input("Tmin,Tmax (e.g. 300,1000): ").split(','))
        temps   = np.linspace(T1, T2, 50)
        κ       = predict_thermal_conductivity(bulk_factory, temps)
        plot_xy(temps, κ,
                "Temperature (K)", "κ (W/m·K)",
                "Thermal Conductivity vs T")

    elif choice == '4':
        # Corrected phase definitions for ase.build.bulk
        phases = {
            "diamond":     (bulk, {"name": "C",
                                   "crystalstructure": "diamond",
                                   "a": 3.57}),
            "lonsdaleite": (bulk, {"name": "C",
                                   "crystalstructure": "hcp",    # use 'hcp'
                                   "a": 2.52,
                                   "c": 4.12}),                 # include c for hcp cell
        }
        T1, T2 = map(float, input("Tmin,Tmax (e.g. 300,2000): ").split(','))
        temps   = np.linspace(T1, T2, 50)
        stable, energies = construct_phase_diagram(phases, temps)
        plt.figure()
        plt.scatter(temps, [list(phases).index(s) for s in stable], c='k')
        plt.yticks(range(len(phases)), list(phases))
        plt.xlabel("Temperature (K)")
        plt.title("Phase Stability Diagram")
        plt.grid(True)
        plt.show()

    else:
        print("Invalid choice.")
        sys.exit(1)

if __name__ == "__main__":
    main()
