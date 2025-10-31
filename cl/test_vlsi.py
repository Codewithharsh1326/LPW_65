# vlsi_analyzer.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------
# Exceptions
# ---------------------
class RulParseError(Exception):
    pass

class GateError(Exception):
    pass

# ---------------------
# .rul Parser
# ---------------------
def parse_rul_file(filepath):
    data = {"GLOBAL": {}, "NMOS": {}, "PMOS": {}}
    current = "GLOBAL"
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            # change section if 'NMOS' / 'PMOS' line occurs alone
            if line == "NMOS":
                current = "NMOS"
                continue
            if line == "PMOS":
                current = "PMOS"
                continue
            if "=" in line:
                left, right = line.split("=", 1)
                key = left.strip()
                # value may have trailing comment or parentheses
                value_str = re.split(r"[ \t(]", right.strip())[0]
                try:
                    value = float(value_str)
                except:
                    value = value_str
                data[current][key] = value
    return data

# ---------------------
# MOS device model (Shichman-Hodges / square law)
# ---------------------
class MOSFET:
    def __init__(self, mu, Vth, Cox, W=1e-6, L=0.12e-6, name="MOS"):
        # mu [m^2/Vs], Cox [F/m^2], W/L in meters
        self.mu = mu
        self.Vth = Vth
        self.Cox = Cox
        self.W = W
        self.L = L
        self.name = name

    def beta(self):
        return self.mu * self.Cox * (self.W / self.L)  # A/V^2

    def Id(self, Vgs, Vds):
        """Return drain current (A). For PMOS, feed positive Vsg,Vsd and use same model."""
        # small threshold handling
        if Vgs <= self.Vth:
            return 0.0
        beta = self.beta()
        Vov = Vgs - self.Vth
        if Vds < Vov:
            # linear region
            return beta * (Vov * Vds - 0.5 * Vds * Vds)
        else:
            # saturation
            return 0.5 * beta * Vov * Vov

# ---------------------
# Gate base and specific gates (NOT, NAND2, NOR2)
# ---------------------
class GateBase:
    def __init__(self, name, VDD, nmos: MOSFET, pmos: MOSFET, C_load=1e-12, f=1e6, alpha=0.5, Ileak=1e-12):
        self.name = name
        self.VDD = VDD
        self.nmos = nmos
        self.pmos = pmos
        self.C_load = C_load
        self.f = f
        self.alpha = alpha
        self.Ileak = Ileak

    def vtc(self, points=200):
        Vin_vals = np.linspace(0, self.VDD, points)
        Vout_vals = np.array([self._solve_vout_for_vin(vin) for vin in Vin_vals])
        return Vin_vals, Vout_vals

    def _solve_vout_for_vin(self, vin, tol=1e-9, maxiter=60):
        """
        Solve for Vout by finding root of f(vout) = Id_nmos - Id_pmos
        Uses bisection on [0, VDD]. If sign doesn't change, returns endpoint approximation.
        """
        a, b = 0.0, self.VDD
        def f(vout):
            Idn = self._Idn_for_gate(vin, vout)
            Idp = self._Idp_for_gate(vin, vout)
            return Idn - Idp

        fa, fb = f(a), f(b)
        if abs(fa) < tol:
            return a
        if abs(fb) < tol:
            return b
        # If sign change, bisection
        if fa * fb <= 0.0:
            low, high = a, b
            flow, fhigh = fa, fb
            for _ in range(maxiter):
                mid = 0.5 * (low + high)
                fmid = f(mid)
                if abs(fmid) < tol:
                    return mid
                if flow * fmid <= 0:
                    high = mid
                    fhigh = fmid
                else:
                    low = mid
                    flow = fmid
            return 0.5 * (low + high)
        # otherwise fallback: choose point where magnitude minimal (scan)
        v_grid = np.linspace(a, b, 201)
        fvals = np.array([abs(f(v)) for v in v_grid])
        vbest = v_grid[np.argmin(fvals)]
        return float(vbest)

    # Methods below should be overridden for multi-input gates, but default treats as inverter
    def _Idn_for_gate(self, vin, vout):
        # NMOS: Vgs = vin, Vds = vout
        return self.nmos.Id(Vgs=vin, Vds=vout)

    def _Idp_for_gate(self, vin, vout):
        # PMOS: use Vsg = VDD - vin, Vsd = VDD - vout
        return self.pmos.Id(Vgs=(self.VDD - vin), Vds=(self.VDD - vout))

    def delay(self, Cload=None):
        """Estimate propagation delay: t_pd = 0.69 * R_eq * Cload.
        R_eq approximated as VDD / Ion_on where Ion_on is current when input = VDD (pull-down) or 0 (pull-up).
        We'll compute propagation using pull-down (worst-case)."""
        C = self.C_load if Cload is None else Cload
        Ion = self._estimate_Ion()
        if Ion <= 0:
            return np.inf
        Req = self.VDD / Ion
        return 0.69 * Req * C

    def _estimate_Ion(self):
        """Estimate on current for pull-down (NMOS network when input(s) = VDD)"""
        # Default: inverter Ion = Id_nmos at Vgs=VDD, Vds=VDD
        return self._Idn_for_gate(self.VDD, 0.0) + 1e-18  # avoid zero

    def dynamic_power(self, Vdd=None, Cload=None, f=None):
        Vdd = self.VDD if Vdd is None else Vdd
        C = self.C_load if Cload is None else Cload
        fr = self.f if f is None else f
        return self.alpha * C * (Vdd**2) * fr

    def static_power(self, Vdd=None):
        Vdd = self.VDD if Vdd is None else Vdd
        return self.Ileak * Vdd

# ---------------------
# Inverter (NOT)
# ---------------------
class Inverter(GateBase):
    pass  # default GateBase behavior is inverter

# ---------------------
# NAND2 (2-input) and NOR2 (2-input)
# Simplified composition model:
# - NMOS series: W_eff = W / N_series
# - PMOS parallel: W_eff = W * N_parallel
# (For NOR, reverse: NMOS parallel, PMOS series)
# ---------------------
class NAND2(GateBase):
    def __init__(self, name, VDD, nmos_template: MOSFET, pmos_template: MOSFET,
                 C_load=1e-12, f=1e6, alpha=0.5, Ileak=1e-12, W_unit=None):
        # choose W_unit to be template W (if not specified)
        W_unit = nmos_template.W if W_unit is None else W_unit
        # For NAND2 assume both inputs drive similar sized transistors
        # Series NMOS count = 2 -> W_eff_n = W_unit / 2
        nmos_eff = MOSFET(mu=nmos_template.mu, Vth=nmos_template.Vth, Cox=nmos_template.Cox,
                          W=W_unit/2.0, L=nmos_template.L, name="NAND2_NMOS_eff")
        # PMOS parallel count = 2 -> W_eff_p = W_unit * 2
        pmos_eff = MOSFET(mu=pmos_template.mu, Vth=pmos_template.Vth, Cox=pmos_template.Cox,
                          W= W_unit*2.0, L=pmos_template.L, name="NAND2_PMOS_eff")
        super().__init__(name, VDD, nmos_eff, pmos_eff, C_load, f, alpha, Ileak)

    def _Idn_for_gate(self, vin, vout):
        # For VTC we must consider both inputs A and B. We'll approximate by using worst-case:
        # worst-case pull-down when both inputs = vin (i.e., parallel? no series), but here we approximate
        # For VTC curve we will consider one input switching, the other tied high (logical '1') or low depending slice.
        # To produce standard VTC slice, we assume other input = vin (worst-case).
        # So NMOS in series: W reduced already: use Id formula with Vgs=vin, Vds=vout
        return self.nmos.Id(Vgs=vin, Vds=vout)

    def _Idp_for_gate(self, vin, vout):
        # PMOS parallel -> stronger pull-up; using effective pmos object
        return self.pmos.Id(Vgs=(self.VDD - vin), Vds=(self.VDD - vout))

class NOR2(GateBase):
    def __init__(self, name, VDD, nmos_template: MOSFET, pmos_template: MOSFET,
                 C_load=1e-12, f=1e6, alpha=0.5, Ileak=1e-12, W_unit=None):
        W_unit = nmos_template.W if W_unit is None else W_unit
        # NMOS parallel: W_eff_n = W_unit * 2
        nmos_eff = MOSFET(mu=nmos_template.mu, Vth=nmos_template.Vth, Cox=nmos_template.Cox,
                          W= W_unit*2.0, L=nmos_template.L, name="NOR2_NMOS_eff")
        # PMOS series: W_eff_p = W_unit / 2
        pmos_eff = MOSFET(mu=pmos_template.mu, Vth=pmos_template.Vth, Cox=pmos_template.Cox,
                          W=W_unit/2.0, L=pmos_template.L, name="NOR2_PMOS_eff")
        super().__init__(name, VDD, nmos_eff, pmos_eff, C_load, f, alpha, Ileak)

    def _Idn_for_gate(self, vin, vout):
        # NMOS parallel approximation: stronger pull-down
        return self.nmos.Id(Vgs=vin, Vds=vout)

    def _Idp_for_gate(self, vin, vout):
        # PMOS series approximation: weaker pull-up
        return self.pmos.Id(Vgs=(self.VDD - vin), Vds=(self.VDD - vout))

# ---------------------
# Utility: plotting and sweeps
# ---------------------
def sweep_and_plot_gate(gate: GateBase, outfolder="results", save_plots=True):
    Path(outfolder).mkdir(parents=True, exist_ok=True)
    name = gate.name

    # 1) VTC
    Vin, Vout = gate.vtc(points=201)
    plt.figure()
    plt.plot(Vin, Vout, label=f"{name} VTC")
    plt.plot(Vin, Vin, 'r--', label="y=x")
    plt.xlabel("Vin (V)")
    plt.ylabel("Vout (V)")
    plt.title(f"{name} - VTC")
    plt.grid(True)
    plt.legend()
    if save_plots:
        plt.savefig(Path(outfolder) / f"{name}_VTC.png", dpi=200)
    plt.close()

    # 2) Delay vs Cload
    C_sweep = np.logspace(-15, -11, 30)  # 1 fF to 10 pF
    delays_ns = []
    for C in C_sweep:
        d = gate.delay(Cload=C)
        delays_ns.append(d * 1e9 if np.isfinite(d) else np.nan)
    plt.figure()
    plt.semilogx(C_sweep * 1e12, delays_ns, marker='o')
    plt.xlabel("C_load (pF)")
    plt.ylabel("Delay (ns)")
    plt.title(f"{name} - Delay vs C_load")
    plt.grid(True)
    if save_plots:
        plt.savefig(Path(outfolder) / f"{name}_Delay_vs_C.png", dpi=200)
    plt.close()

    # 3) Power vs VDD and PDP
    VDD_sweep = np.linspace(max(0.4, gate.VDD*0.5), gate.VDD*1.5, 30)
    dyn_power_uW = []
    stat_power_nW = []
    pdp_pJ = []
    for V in VDD_sweep:
        pdyn = gate.dynamic_power(Vdd=V) * 1e6  # uW
        pstat = gate.static_power(Vdd=V) * 1e9  # nW
        # estimate delay using approximate Ion at this V (we keep same device objects but adjust VDD context)
        # We'll temporarily set gate.VDD to V to compute Ion estimate used by delay()
        oldV = gate.VDD
        gate.VDD = V
        tpd = gate.delay()  # seconds
        gate.VDD = oldV
        dyn_power_uW.append(pdyn)
        stat_power_nW.append(pstat)
        pdp_pJ.append(pdyn * 1e-6 * tpd * 1e12 if (tpd is not None and np.isfinite(tpd)) else np.nan)
        # pdp: dyn_power (W) * delay (s) => Joules -> pJ multiply 1e12
    plt.figure()
    plt.plot(VDD_sweep, dyn_power_uW, label="Dynamic Power (µW)")
    plt.plot(VDD_sweep, stat_power_nW, label="Static Power (nW)")
    plt.xlabel("VDD (V)")
    plt.ylabel("Power")
    plt.title(f"{name} - Power vs VDD")
    plt.grid(True)
    plt.legend()
    if save_plots:
        plt.savefig(Path(outfolder) / f"{name}_Power_vs_VDD.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(VDD_sweep, pdp_pJ)
    plt.xlabel("VDD (V)")
    plt.ylabel("PDP (pJ)")
    plt.title(f"{name} - PDP vs VDD")
    plt.grid(True)
    if save_plots:
        plt.savefig(Path(outfolder) / f"{name}_PDP_vs_VDD.png", dpi=200)
    plt.close()

    # summary numbers for default C_load and VDD
    summary = {
        "Gate": name,
        "VDD (V)": gate.VDD,
        "C_load (F)": gate.C_load,
        "Delay_ns": gate.delay() * 1e9,
        "Pdyn_uW": gate.dynamic_power() * 1e6,
        "Pstat_nW": gate.static_power() * 1e9,
    }
    return summary

# ---------------------
# Main driver: parse .rul, create devices & gates, sweep and save results
# ---------------------
def run_analysis(rul_path="cmos012.rul", outfolder="results", plot=True):
    params = parse_rul_file(rul_path)
    G = params.get("GLOBAL", {})
    NM = params.get("NMOS", {})
    PM = params.get("PMOS", {})

    # Basic derived constants
    eps0 = 8.8541878128e-12
    gateK = G.get("gateK", 3.9)
    tox = NM.get("l3tox", NM.get("b4toxe", 2e-9))
    Cox = (eps0 * gateK) / tox  # F/m^2

    # mobilities from rul: l3u0 or b4u0; convert units if needed (rul usually gives values in ?).
    # We'll assume values are in m^2/Vs if small; if they look like 0.05 we keep them as m^2/Vs.
    mu_n = NM.get("l3u0", NM.get("b4u0", 0.05))
    mu_p = abs(PM.get("l3u0", PM.get("b4u0", 0.02)))

    # thresholds (abs for pmos)
    Vth_n = NM.get("l3vto", NM.get("b4vtho", 0.4))
    Vth_p = abs(PM.get("l3vto", PM.get("b4vtho", -0.45)))

    # default geometry
    lam = G.get("lambda", 0.06)
    # gate length default: 2*lambda (from earlier conversation)
    L_default = 2 * lam * 1e-6  # in meters (lambda given in um basis? rul used lambda=0.06 meaning 0.06 um)
    # If lambda seems like microns, convert properly: we assume lambda in microns -> multiply by 1e-6
    # The above line converts lambda (assumed um) to meters.
    W_default = 1e-6  # 1 um width by default

    # device templates
    nmos_template = MOSFET(mu=mu_n, Vth=Vth_n, Cox=Cox, W=W_default, L=L_default, name="NMOS_template")
    pmos_template = MOSFET(mu=mu_p, Vth=Vth_p, Cox=Cox, W=W_default, L=L_default, name="PMOS_template")

    VDD_proc = G.get("vdd", 1.2)
    Ileak_default = 1e-12

    # create gates
    inv = Inverter("INV", VDD_proc, nmos_template, pmos_template, C_load=1e-12, f=1e6, alpha=0.5, Ileak=Ileak_default)
    nand2 = NAND2("NAND2", VDD_proc, nmos_template, pmos_template, C_load=2e-12, f=1e6, alpha=0.5, Ileak=Ileak_default)
    nor2 = NOR2("NOR2", VDD_proc, nmos_template, pmos_template, C_load=2e-12, f=1e6, alpha=0.5, Ileak=Ileak_default)

    gates = [inv, nand2, nor2]
    summaries = []
    for g in gates:
        print(f"Processing gate: {g.name}")
        s = sweep_and_plot_gate(g, outfolder=outfolder, save_plots=plot)
        summaries.append(s)

    # Save summary CSV
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(Path(outfolder) / "gate_summary.csv", index=False)
    print(f"Saved summary to {Path(outfolder) / 'gate_summary.csv'} and plots in {outfolder}")

# ---------------------
# If run as script
# ---------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLSI Timing & Power Analyzer using .rul foundry")
    parser.add_argument("--rul", default="cmos012.rul", help="Path to .rul file")
    parser.add_argument("--out", default="results", help="Output folder for plots and CSV")
    args = parser.parse_args()
    run_analysis(rul_path=args.rul, outfolder=args.out, plot=True)
