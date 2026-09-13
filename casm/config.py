"""Locked Phase 1 CASM experiment constants.

These values are deliberately small and explicit so later experiments cannot
silently change the hypothesis while retaining the same experiment name.
"""

FAN_IN_CAP = 2
BOOLEAN_OPS = ("AND", "OR", "XOR", "NOT")
ROUTER_TEMPERATURE_INIT = 2.0
ROUTER_LOGIT_STD_TARGET = 0.2
ALPHA_PARAMETERIZATION = "c * softplus(eta_r)"
ALPHA_INDEXING = "syntactic_relation_only"
ROUTER_INPUT = "structural_only"
RUNTIME_INPUT = "execution_only"
HARD_CONTROL_VALUE_SEPARATION = True

# Phase 1 is feed-forward. Recurrent stability and temporal routing are Phase 2.
RECURRENT_SETTLING = False
SPECTRAL_RADIUS_CALIBRATION = False

# The first control must be able to falsify the benchmark before router training.
REQUIRE_STRICT_CANDIDATE_SUPERSET = True
REQUIRE_EXHAUSTIVE_MINIMALITY_FOR_SMALL_GRAPHS = True
