# -*- coding: utf-8 -*-

from internal_helpers import *

"""
surface_code.py

Circuit generation for a rotated surface code memory experiment (Z or X basis)
on IQM Emerald. By default distance 3: 9 data + 8 ancilla = 17 qubits.

Pipeline via Qrisp OR Qiskit
------------------------------------
  Qrisp  : circuit → Qrisp DetectorExperiment → IQM Emerald
           → detection events → decoder
  Qiskit : circuit → Qiskit circuit → IQM Emerald → raw measurements
           → extract_syndromes() → decoder


Generators (this file): pick ONE starting point
-----------------------------------------------
  make_stim_circuit()      For both Qrisp and Qiskit. Use one of the converters (VERIFY their funcionality FIRST!) to run on hardware, as Stim circuits dont work.
                           (stim_to_qiskit.py / stim_to_qrisp.py)
  make_qiskit_circuit()    Qiskit-native alternative
  make_qrisp_experiment()  Qrisp-native alternative

Where the other stages live
---------------------------
  1. Circuit generation              this file
  2. Stim → Qiskit / Qrisp           stim_to_qiskit.py, stim_to_qrisp.py
  3. OPTIONAL Emerald qubit mapping  build_emerald_qubit_rotated.py  (only needed for Qiskit)
  4. Simulation                      TBD (section 4 below)
  5. Hardware execution              run_on_hardware.py: run_qiskit() / run_qrisp()
  6. Syndromes → Decoder             extract_syndromes.py (only needed for Qiskit),
                                     run_on_hardware.py: decode_hardware_results()
  Helpers                            internal_helpers.py

Usage
-----
  from surface_code import make_stim_circuit, DEFAULT_NOISE
  from run_on_hardware import run_qiskit, run_qrisp, decode_hardware_results


"""

import numpy as np
import stim
import pymatching
from qiskit import QuantumCircuit, transpile
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1.  STIM CIRCUIT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

#: Default noise model matching IQM Emerald typical gate error rates (~0.1-0.5%).
#: Adjust these based on your latest calibration data from Resonance.
DEFAULT_NOISE = dict(
    after_clifford_depolarization  = 0.003,
    after_reset_flip_probability   = 0.003,
    before_measure_flip_probability= 0.003,
    before_round_data_depolarization = 0.003,
)


def make_stim_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    noise: Optional[dict] = None,
) -> stim.Circuit:
    """
    Generates a Stim circuit for the rotated surface code.

    Shared starting point for BOTH tracks. A Stim circuit does not run on hardware, you need to convert it first
    first:
        Qiskit track : stim_to_qiskit(stim_circ)               (stim_to_qiskit.py)
        Qrisp track  : stim_to_detector_experiment(stim_circ)  (stim_to_qrisp.py)
    Alternatively, you can start from make_qiskit_circuit() / make_qrisp_experiment().

    Parameters
    ----------
    distance : code distance
    rounds   : number of stabilizer rounds
    basis    : "Z" or "X" memory experiment NOTE: Stim's X-memory circuits use RX / MX,
             which the provided converters do not handle. Possible avenue for extension.
    no_reset : True -> ancillas are measured without reset (required on Emerald)
    noise    : None for the hardware circuit. For for simulation / DEM it can be  e.g. DEFAULT_NOISE or sophisticated choices.

    NOTE: Emerald does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant.
    With reset: ancilla starts fresh each round. Raw measurement m_k = syndrome s_k of round k. Detection event = s_k XOR s_(k-1) = m_k XOR m_(k-1).
    Without reset: ancilla accumulates. Raw measurement m_k = s_k XOR m_(k-1), i.e. the XOR of all syndromes up to round k.
    The detection event is the same quantity (s_k XOR s_(k-1)), but in raw measurements it becomes m_k XOR m_(k-2)  (with m_0 = 0).
    So the DETECTOR definitions must be rewritten for the no-reset circuit. stim_to_qrisp.py does this rewrite for you (PLEASE VERIFY!); on the Qiskit track it is on you.
    Trade-off: no-reset avoids reset errors (X_ERROR after R), but the ancilla stays "live" (potentially decohering) throughout, which causes amplitude damping.


    Returns
    -------
    stim.Circuit
    """

    # ── Argument validation ────────────────────────────────────────────────
    if isinstance(distance, bool) or not isinstance(distance, (int, np.integer)):
        raise TypeError(f"distance must be an int, got {type(distance).__name__}")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")

    if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)):
        raise TypeError(f"rounds must be an int, got {type(rounds).__name__}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    if not isinstance(basis, str) or basis.upper() not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X', got {basis!r}")
    basis = basis.upper()

    noise = _validate_noise(noise)

    # ── Standard (with-reset) circuit from Stim ────────────────────────────
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=int(distance),
        rounds=int(rounds),
        **noise,
    )

    if not no_reset:
        return circ

    # ── No-reset variant (required on Emerald) ─────────────────────────────
    return _remove_ancilla_resets(circ)


#: Noise parameters accepted by stim.Circuit.generated().
_ALLOWED_NOISE_KEYS = (
    "after_clifford_depolarization",
    "after_reset_flip_probability",
    "before_measure_flip_probability",
    "before_round_data_depolarization",
)

#: Measure-and-reset gate -> plain measurement in the same basis.
_MR_TO_M = {"MR": "M", "MRX": "MX", "MRY": "MY"}
#: Error channel Stim places right after a reset in that basis (the reset error).
_RESET_ERROR = {"MR": "X_ERROR", "MRX": "Z_ERROR", "MRY": "X_ERROR"}


def _validate_noise(noise: Optional[dict]) -> dict:
    """Returns a validated *copy* of the noise dict ({} for None). Never mutates the input."""
    if noise is None:
        return {}
    if not isinstance(noise, dict):
        raise TypeError(f"noise must be a dict or None, got {type(noise).__name__}")

    unknown = set(noise) - set(_ALLOWED_NOISE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown noise parameter(s) {sorted(unknown)}. "
            f"Allowed: {list(_ALLOWED_NOISE_KEYS)}"
        )

    clean = {}
    for key, p in noise.items():
        if isinstance(p, bool) or not isinstance(p, (int, float, np.floating, np.integer)):
            raise TypeError(f"noise[{key!r}] must be a number, got {type(p).__name__}")
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"noise[{key!r}] must be a probability in [0, 1], got {p}")
        clean[key] = p
    return clean


def _remove_ancilla_resets(circ: stim.Circuit) -> stim.Circuit:
    """
    Turns a with-reset Stim circuit into its no-reset equivalent.

    - Every mid-circuit MR / MRX / MRY becomes M / MX / MY (no reset after measurement).
    - The reset error channel Stim inserts right after these MR gates is dropped
      (there is no reset anymore, so no reset error).
    - DETECTOR and OBSERVABLE_INCLUDE are rewritten for accumulating ancillas:
      without reset, raw outcome m_j = s_j XOR m_prev, so every syndrome s_j referenced
      by the original circuit is replaced by {m_j, m_prev} (symmetric difference).
      For a detector s_k XOR s_(k-1) this yields m_k XOR m_(k-2), as expected.

    The result has correct detectors itself, so it can be used directly for
    compile_m2d_converter() / extract_syndromes() and detector_error_model().
    Passing it to stim_to_detector_experiment() is also safe: it contains no MR,
    so that converter performs no additional rewrite.
    """
    out = stim.Circuit()
    num_meas = 0
    last_meas_on_qubit: dict[int, int] = {}
    reset_removed: set[int] = set()          # measurement indices whose reset was removed
    prev_accum: dict[int, int] = {}          # j -> earlier measurement that m_j accumulates on
    pending_reset_error: dict[int, str] = {} # qubit -> reset error gate to drop right after MR

    def rewrite(indices):
        acc = set()
        for j in indices:
            acc ^= {j}
            if j in prev_accum:
                acc ^= {prev_accum[j]}
        return [stim.target_rec(j - num_meas) for j in sorted(acc)]

    for inst in circ.flattened():
        name = inst.name
        targets = inst.targets_copy()
        args = inst.gate_args_copy()

        # Drop the reset error that directly follows a removed reset.
        if pending_reset_error:
            if name in ("X_ERROR", "Z_ERROR", "Y_ERROR"):
                kept = [t for t in targets
                        if not (t.is_qubit_target and pending_reset_error.get(t.value) == name)]
                for t in targets:
                    if t.is_qubit_target and pending_reset_error.get(t.value) == name:
                        del pending_reset_error[t.value]
                if kept:
                    out.append(name, kept, args)
                continue
            pending_reset_error.clear()

        if name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            for t in targets:
                q = t.value
                prev = last_meas_on_qubit.get(q)
                if prev is not None and prev in reset_removed:
                    prev_accum[num_meas] = prev
                if name in _MR_TO_M:
                    reset_removed.add(num_meas)
                    pending_reset_error[q] = _RESET_ERROR[name]
                last_meas_on_qubit[q] = num_meas
                num_meas += 1
            out.append(_MR_TO_M.get(name, name), targets, args)

        elif name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            recs = [num_meas + t.value for t in targets if t.is_measurement_record_target]
            others = [t for t in targets if not t.is_measurement_record_target]
            out.append(name, rewrite(recs) + others, args)

        elif name in ("R", "RX", "RY") and any(
            t.value in last_meas_on_qubit for t in targets
        ):
            # A reset on an already-measured qubit would be a mid-circuit reset.
            raise NotImplementedError(
                f"Mid-circuit {name} on a measured qubit is not supported in no-reset mode."
            )

        else:
            out.append(inst)

    return out


def get_circuit_info(circuit: stim.Circuit) -> dict:
    """
    Returns a summary dict of qubit counts and measurement structure.
    Useful for sanity-checking before hardware runs.
    """
    data_q, anc_q = get_qubit_lists(circuit)
    return {
        "num_qubits"      : circuit.num_qubits,
        "num_data_qubits" : len(data_q),
        "num_anc_qubits"  : len(anc_q),
        "num_measurements": circuit.num_measurements,
        "num_detectors"   : circuit.num_detectors,
        "num_observables" : circuit.num_observables,
        "data_stim_indices": sorted(data_q),
        "anc_stim_indices" : sorted(anc_q),
        "meas_order"      : get_meas_order(circuit),
    }


def make_qiskit_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
) -> QuantumCircuit:
    """
    QISKIT TRACK, native alternative.
    Generates a Qiskit circuit for the rotated surface code.

    Pros (vs. Stim + converter)
    ---------------------------
    + Full control over gate basis (e.g. IQM native) and order (e.g. qubit placement). 
    + No converter before it is passed to the transpiler.

    Cons
    ----
    - No detectors or logical observables: Tracking needs to be done by user, across rounds and
      for the no-reset case.
    - No detector error model: for decoding you still need a Stim version of your circuit
      (or build the decoding graph by hand). extract_syndromes() only works with a
      Stim circuit that matches yours.

    NOTE: Emerald does not support mid-circuit reset, so the circuit that runs on hardware must be the no-reset variant.
    With reset: ancilla starts fresh each round. Raw measurement m_k = syndrome s_k of round k. Detection event = s_k XOR s_(k-1) = m_k XOR m_(k-1).
    Without reset: ancilla accumulates. Raw measurement m_k = s_k XOR m_(k-1), i.e. the XOR of all syndromes up to round k.
    The detection event is the same quantity (s_k XOR s_(k-1)), but in raw measurements it changes to m_k XOR m_(k-2)  (with m_0 = 0).


    Returns
    -------
    qiskit.QuantumCircuit
    """

    return

def make_qrisp_experiment(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    p: Optional[float] = None,
):
    """
    QRISP TRACK, native alternative to make_stim_circuit() + stim_to_detector_experiment().
    Generates a Qrisp DetectorExperiment for the rotated surface code.

    Pros (vs. Stim + converter)
    ---------------------------
    + Detectors and observables are part of the experiment (qrisp.parity). Qrisp derives the Stim circuit and DEM from it.
    + Noise model per gate via stim_noise annotations: easy to adjust according to calibration data.
    + Full control over gate choice, CX/CZ schedule and layout.

    Cons
    ----
    - Geometry, CX schedule and detector bookkeeping: Qrisp has no
      surface-code generator.
    - Expectation values must be correct. Wrong choice is Invisible in a
      noiseless Stim simulation, but breaks the hardware results.
    - The no-reset detector algebra (which raw measurements form a detector) needs to be implemented.

    Parameters
    ----------
    distance, rounds, basis, no_reset : as in make_stim_circuit()
    p        : noise strength for stim_noise annotations (decoder weights only;
               ignored on hardware). None -> noiseless.

    Returns
    -------
    iqm.qrisp_iqm.DetectorExperiment
    """

    return


##########################
# Below the necessary steps. No need to put everything in this file. 
##########################

# ─────────────────────────────────────────────────────────────────────────────
#  2.  STIM → Qrisp/QISKIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  3.  EMERALD QUBIT MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  4.  STIM/AER SIMULATION  (no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  5.  Hardware execution
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  6.  Offline Decoding
# ─────────────────────────────────────────────────────────────────────────────

# TBD