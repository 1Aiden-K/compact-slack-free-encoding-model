from pulser import AnalogDevice

from ising import build_ising, make_pulse
from qubo import build_q_matrix

params = build_ising(
    build_q_matrix(0.1),
    energy_scale=0.2,
    interaction_coeff=AnalogDevice.interaction_coeff,
)
pulse = make_pulse(params)
