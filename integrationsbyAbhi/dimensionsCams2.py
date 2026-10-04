"""
dimensionsCams2.py — Dimensional Analysis & Length/Width Inspection Module.
Alias/Re-export of integrationsbyAbhi/dimensionsCam2.py for seamless compatibility.
"""
from integrationsbyAbhi.dimensionsCam2 import (
    KNOWN_DISTANCE_MM,
    KNOWN_WIDTH_MM,
    find_largest_object,
    calculate_focal_length,
    calculate_real_width,
    calculate_real_length,
    calculate_real_dimension,
    check_length_and_width,
    draw_dimension_annotations,
    dimensional_analysis,
    run_live_pinhole_measurement,
)

__all__ = [
    "KNOWN_DISTANCE_MM",
    "KNOWN_WIDTH_MM",
    "find_largest_object",
    "calculate_focal_length",
    "calculate_real_width",
    "calculate_real_length",
    "calculate_real_dimension",
    "check_length_and_width",
    "draw_dimension_annotations",
    "dimensional_analysis",
    "run_live_pinhole_measurement",
]

if __name__ == "__main__":
    run_live_pinhole_measurement()
