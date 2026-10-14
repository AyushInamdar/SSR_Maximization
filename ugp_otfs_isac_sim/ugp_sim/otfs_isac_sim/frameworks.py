"""
Transmission frameworks (OTFS and non-OTFS) compared in the simulator.

  otfs_isac      proposed: OTFS, beams from ISAC echo-based prediction (Draft)
  ofdm_isac      same sensing pipeline, OFDM waveform: Doppler ICI + per-symbol CP
  otfs_feedback  Yuan21 benchmark: angle from one downlink pilot + codebook index
                 feedback, no kinematic prediction, pilot/feedback overhead; Eves
                 sensed by a separate isotropic probe every frame
  otfs_perfect   genie: true user and Eve states known (upper bound, zero-size E_j)

DRAFT notes:
  * OFDM-ISAC sensing uses the same crude CRB as OTFS. A proper comparison needs
    the OFDM ambiguity function / Doppler-resolution penalty — TODO.
  * The feedback benchmark is a stylised version of Yuan21's (one pilot, fixed
    overhead fraction), not the exact protocol of their ref. [27].
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import SimParams
from .waveform import rate_prefactor


@dataclass(frozen=True)
class Framework:
    name: str
    waveform: str            # "otfs" | "ofdm"
    user_sensing: str        # "isac" | "feedback" | "perfect"
    eve_sensing: str         # "isac" | "probe" | "perfect"
    overhead: float = 0.0    # extra fraction of the frame lost (pilots/feedback)

    def prefactor(self, p: SimParams) -> float:
        return rate_prefactor(self.waveform, p) * (1 - self.overhead)


def get_frameworks(p: SimParams) -> dict[str, Framework]:
    return {
        "otfs_isac": Framework("otfs_isac", "otfs", "isac", "isac"),
        "ofdm_isac": Framework("ofdm_isac", "ofdm", "isac", "isac"),
        "otfs_feedback": Framework("otfs_feedback", "otfs", "feedback", "probe",
                                   overhead=p.feedback_overhead),
        "otfs_perfect": Framework("otfs_perfect", "otfs", "perfect", "perfect"),
    }
