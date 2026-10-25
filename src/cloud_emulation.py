# https://docs.pasqal.com/pulser/tutorials/backends/
import os
import time

import numpy as np
import pulser
import pulser_simulation
from dotenv import load_dotenv
from pasqal_cloud import PasqalCloudConnection

# Load the .env file
load_dotenv()

USERNAME = os.getenv("USERNAME")
PROJECT_ID = os.getenv("PROJECT_ID")
PASSWORD = os.getenv("PASSWORD")

connection = PasqalCloudConnection(
    username=USERNAME,  # Your username or email address for the Pasqal Cloud Platform
    project_id=PROJECT_ID,  # The ID of the project associated to your account
    password=PASSWORD,  # The password for your Pasqal Cloud Platform account
)


def wait_for_results(results, name, timeout=300):
    start = time.monotonic()

    while True:
        status = results.get_batch_status()
        print(f"{name}: {status}")

        if status.name == "DONE":
            return results[-1]

        if status.name in {"ERROR", "CANCELED", "TIMED_OUT"}:
            raise RuntimeError(f"{name} failed: {status}")

        if time.monotonic() - start > timeout:
            raise TimeoutError(f"{name} timed out: {status}")

        time.sleep(2)


# 1. Picking a Device
device = pulser.AnalogDevice

# 2. Creating the Register
R_interatomic = 5  # um
register = pulser.Register.hexagon(1, R_interatomic, prefix="q")

seq = pulser.Sequence(register, device)

# 3. Picking the channels
seq.declare_channel("rydberg_global", "rydberg_global")

# 4. Adding the pulses

# Parameters in rad/µs
U = device.interaction_coeff / R_interatomic**6

# Time parameters
total_duration = 4000  # in ns
interp_pts = np.linspace(0, 1, 4)  # between 0 and 1

seq.add(
    pulser.Pulse(
        pulser.InterpolatedWaveform(
            total_duration,
            U * np.array([1e-9, 0.22, 0.2181, 1e-9]),
            times=interp_pts,
        ),
        pulser.InterpolatedWaveform(
            total_duration,
            U * np.array([-1, 0.0556, 0.332, 1]),
            times=interp_pts,
        ),
        0,
    ),
    "rydberg_global",
)

seq.draw(draw_register=True)

qutip_bknd = pulser.backends.QutipBackendV2(seq)
qutip_results = qutip_bknd.run()

print(pulser.backends.QutipBackendV2.default_config)

print(qutip_results)

print("Stored results' tags:", qutip_results.get_result_tags())
print(
    "Evaluation times for 'bitstrings':",
    qutip_results.get_result_times("bitstrings"),
)
# These are equivalent
qutip_results.get_result("bitstrings", 1.0)
qutip_results.bitstrings[-1]

# Pick a backend, here we chose QutipBackendV2
emu_backend_class = pulser_simulation.QutipBackendV2

config_class = emu_backend_class.config_type  # In this case, `QutipConfig`
state_class = config_class.state_type  # In this case, `QutipState`

fidelity_state = state_class.from_state_amplitudes(
    eigenstates=("r", "g"),
    amplitudes={"grgrgrg": 1 / np.sqrt(2), "ggrgrgr": 1 / np.sqrt(2)},
)
fidelity = pulser.backend.Fidelity(
    state=fidelity_state, tag_suffix="AFM_7", evaluation_times=[0.5, 1.0]
)

# Start from the default_config and add the Fidelity observable
default_config = emu_backend_class.default_config
new_observables = list(default_config.observables) + [fidelity]
config = default_config.with_changes(observables=new_observables)
print(config)

qutip_bknd_custom = emu_backend_class(seq, config=config)
custom_results = qutip_bknd_custom.run()
print(custom_results)

remote_fidelity_state = pulser.backend.StateRepr.from_state_amplitudes(
    eigenstates=("r", "g"),
    amplitudes={"grgrgrg": 1 / np.sqrt(2), "ggrgrgr": 1 / np.sqrt(2)},
)
remote_fidelity = pulser.backend.Fidelity(
    state=remote_fidelity_state,
    tag_suffix="AFM_7",
    evaluation_times=[0.5, 1.0],
)

remote_config = pulser.backend.EmulationConfig(
    observables=[remote_fidelity],
)

free_bknd = pulser.backends.RemoteEmuFreeBackend(
    seq, config=remote_config, connection=connection
)
mps_bknd = pulser.backends.RemoteMPSBackend(
    seq, config=remote_config, connection=connection
)

# Submit jobs
free_results = free_bknd.run()
print(f"Submitted batch {free_results.batch_id} to RemoteEmuFreeBackend")

mps_results = mps_bknd.run()
print(f"Submitted batch {mps_results.batch_id} to RemoteMPSBackend")

# Wait for completion
free_result = wait_for_results(free_results, "RemoteEmuFreeBackend")
mps_result = wait_for_results(mps_results, "RemoteMPSBackend")

# Display results
print("\nFree Backend:")
print(free_result)
print("Fidelities:", free_result.fidelity_AFM_7)

print("\nMPS Backend:")
print(mps_result)
print("Fidelities:", mps_result.fidelity_AFM_7)
