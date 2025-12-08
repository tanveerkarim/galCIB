

_COLOR_CORRECTIONS = {
    100: 1.076,
    143: 1.017,
    217: 1.119,
    353: 1.097,
    545: 1.068,
    857: 0.995,
}


def get_color_correction(nu: int) -> float:
    if nu not in _COLOR_CORRECTIONS:
        raise ValueError(f"Do not have a color correction for nu = {nu}")
    return _COLOR_CORRECTIONS[nu]
