#!/usr/bin/env python3
"""Draw a Pulser register from a run-result JSON file.

Install:
    python -m pip install pulser-core matplotlib

Examples:
    uv run draw_pulser_register.py tier_s_seed_0000.json
    uv run draw_pulser_register.py tier_s_seed_0000.json --no-show
    uv run draw_pulser_register.py tier_s_seed_0000.json --blockade-radius 7.0
    uv run draw_pulser_register.py tier_s_seed_0000.json --no-estimate

The atom coordinates come from program.positions_um, not instance.coordinates.

Radius selection, in priority order:
  1. --blockade-radius (explicit command-line override)
  2. interaction_radius_um / blockade_radius_um (and unsuffixed variants)
     in program, program.encoding, settings, or the top level of the JSON
  3. An estimated blockade radius from program.omega_peak, using Pulser's
     AnalogDevice, if no radius is present. This assumes the AnalogDevice
     interaction coefficient and is NOT an explicit radius from the JSON.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pulser import Register

RADIUS_KEYS = (
    "interaction_radius_um",
    "blockade_radius_um",
    "rydberg_blockade_radius_um",
    "interaction_radius",
    "blockade_radius",
    "rydberg_blockade_radius",
)


def positive_radius(value: Any, source: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive number in micrometers.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive number in micrometers.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{source} must be finite and greater than zero.")
    return result


def load_data(filename: Path) -> tuple[dict, list[list[float]]]:
    with filename.open(encoding="utf-8") as f:
        data = json.load(f)

    try:
        positions = data["program"]["positions_um"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected 'program.positions_um' in the JSON.") from exc

    if not isinstance(positions, list) or not positions:
        raise ValueError("'program.positions_um' must be a nonempty list.")

    try:
        coords = [[float(x), float(y)] for x, y in positions]
    except (ValueError, TypeError) as exc:
        raise ValueError("Each position must be an (x, y) numeric pair.") from exc

    if not all(math.isfinite(x) and math.isfinite(y) for x, y in coords):
        raise ValueError("Atom coordinates must be finite numbers.")

    declared = data["program"].get("encoding", {}).get("n_atoms")
    if declared is not None and declared != len(coords):
        raise ValueError(
            f"Found {len(coords)} positions, but encoding.n_atoms={declared}."
        )

    return data, coords


def radius_from_json(data: dict) -> tuple[float, str] | None:
    """Read an explicit radius; do not mistake interaction error for radius."""
    program = data.get("program") or {}
    for parent, section in (
        ("program", program),
        ("program.encoding", program.get("encoding") or {}),
        ("settings", data.get("settings") or {}),
        ("root", data),
    ):
        if not isinstance(section, dict):
            continue
        for key in RADIUS_KEYS:
            if key in section and section[key] is not None:
                return positive_radius(
                    section[key], f"{parent}.{key}"
                ), f"{parent}.{key}"
    return None


def choose_radius(
    data: dict, override: float | None, estimate: bool
) -> tuple[float | None, str]:
    if override is not None:
        return positive_radius(override, "--blockade-radius"), "command-line override"

    existing = radius_from_json(data)
    if existing is not None:
        return existing

    if not estimate:
        return None, "no radius found in JSON (estimation disabled)"

    omega = (data.get("program") or {}).get("omega_peak")
    if omega is None:
        return None, "no radius or program.omega_peak found in JSON"

    omega = positive_radius(omega, "program.omega_peak")
    from pulser import AnalogDevice

    rb = AnalogDevice.rydberg_blockade_radius(omega)
    return float(rb), "ESTIMATE from program.omega_peak using pulser.AnalogDevice"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="Input JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("register.png"),
        help="Output image (default: register.png)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save without opening a plot window"
    )
    parser.add_argument(
        "--blockade-radius",
        type=float,
        default=None,
        help="Override radius in micrometers (takes precedence over JSON)",
    )
    parser.add_argument(
        "--no-estimate",
        action="store_true",
        help="If JSON has no explicit radius, draw without a blockade radius",
    )
    args = parser.parse_args()

    data, positions = load_data(args.instance)
    radius, source = choose_radius(data, args.blockade_radius, not args.no_estimate)

    # Retain the exact positions supplied by the file (no recentering).
    register = Register.from_coordinates(positions, center=False, prefix="q")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    radius = 2 * radius

    register.draw(
        with_labels=True,
        blockade_radius=radius,
        draw_half_radius=radius is not None,
        draw_graph=radius is not None,
        fig_name=str(args.output),
        kwargs_savefig={"dpi": 200, "bbox_inches": "tight"},
        show=not args.no_show,
    )

    print(f"Drew {len(positions)} atoms: {', '.join(map(str, register.qubit_ids))}")
    if radius is None:
        print(f"Blockade radius: unavailable ({source})")
    else:
        print(f"Blockade radius: {radius:.4f} um ({source})")
    print(f"Saved register image to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
