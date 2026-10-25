"""Emulation-only site-dependent detuning for an analog Rydberg register.

The DMM can only supply negative detuning with site weights in [0, 1].
To match target detunings delta_i, take the global endpoint at max(delta_i)
and subtract the individual differences using one weighted DMM waveform.
"""

from dataclasses import replace

import numpy as np


def dmm_plan(target_detunings, duration_ns):
    """Return a global endpoint and synchronous negative DMM ramp.

    The starting DMM detuning is zero, so the existing global negative initial
    detuning still initializes all atoms in |g>. At the final time every site
    sees its desired diagonal field, *before* geometry approximation.
    """
    target = np.asarray(target_detunings, dtype=float)
    if target.ndim != 1 or not target.size or not np.isfinite(target).all():
        raise ValueError('DMM target detunings must be a finite vector.')
    if duration_ns < 2:
        raise ValueError('DMM duration must be at least 2 ns.')
    global_final = float(np.max(target))
    deficits = np.maximum(global_final - target, 0.0)
    amplitude = float(np.max(deficits))
    weights = deficits / amplitude if amplitude > 0 else np.zeros_like(deficits)
    # An exactly zero profile requires no DMM channel at all.
    samples = -amplitude * np.linspace(0.0, 1.0, int(duration_ns))
    return dict(global_final=global_final, weights=weights,
                amplitude=amplitude, samples=samples,
                total_site_detuning=float(deficits.sum()))


def make_dmm_device(base_device):
    """Copy AnalogDevice's geometry/interaction properties into a virtual DMM device.

    The DMM specifications follow Pulser's WeightedAnalogDevice. This is for
    emulator comparison only, NOT a valid program for a global-only QPU.
    """
    from pulser.devices import WeightedAnalogDevice

    if getattr(base_device, 'dmm_channels', {}):
        return base_device
    virtual = base_device.to_virtual() if hasattr(base_device, 'to_virtual') else base_device
    return replace(virtual, name='AnalogDeviceWithDMM',
                   dmm_objects=WeightedAnalogDevice.dmm_objects)


def validate_dmm_plan(plan, dmm_channel, safety_factor=1.0):
    """Fail clearly if the desired detunings exceed DMM constraints."""
    required = plan['amplitude']
    total = plan['total_site_detuning']
    floor = dmm_channel.bottom_detuning
    total_floor = dmm_channel.total_bottom_detuning
    if floor is not None and required > -float(floor) * safety_factor + 1e-9:
        raise ValueError(f'DMM requires {required:.4g} rad/us per atom, '
                         f'but the channel allows {-float(floor):.4g}; reduce --scale.')
    if total_floor is not None and total > -float(total_floor) * safety_factor + 1e-9:
        raise ValueError(f'DMM requires {total:.4g} rad/us total, '
                         f'but the channel allows {-float(total_floor):.4g}; reduce --scale.')
