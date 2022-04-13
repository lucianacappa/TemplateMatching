import cv2
import numpy
from timer import Timer


METHODS = {
    'TM_CCOEFF': (cv2.TM_CCOEFF, -1),
    'TM_CCOEFF_NORMED': (cv2.TM_CCOEFF_NORMED, -1),
    'TM_CCORR': (cv2.TM_CCORR, -1),
    'TM_CCORR_NORMED': (cv2.TM_CCORR_NORMED, -1),
    'TM_SQDIFF': (cv2.TM_SQDIFF, 1),
    'TM_SQDIFF_NORMED': (cv2.TM_SQDIFF_NORMED, 1),
}

PARAMETER_SPECS = {
    'method': {
        'type': str,
        'options': [
            'TM_CCOEFF',
            'TM_CCOEFF_NORMED',
            'TM_CCORR',
            'TM_CCORR_NORMED',
            'TM_SQDIFF',
            'TM_SQDIFF_NORMED',
        ]
    },
    'n_matches': {'type': int, 'min': 1, 'max': 100, 'step': 1},
    'min_strength': {'type': float, 'min': 0.01, 'max': 1.0, 'step': 0.01},
}


def generate_plot(base, query, matches):
    bh, bw = base.shape[:2]
    qh, qw = query.shape[:2]
    plot_image = numpy.zeros((max(bh, qh), bw + qw, 3), numpy.uint8)
    plot_image[0:bh, 0:bw] = base
    plot_image[0:qh, bw:bw + qw] = query
    for match in matches:
        location = match[0]
        value = match[1]
        bottom_right = (location[0] + qw, location[1] + qh)
        cv2.rectangle(plot_image, location, bottom_right, (0, int(value), 255 - int(value)), 5)
    return plot_image


def plot(base, query, method: str, n_matches: int = None, min_strength: float = None):
    # print('template_matcher.plot: base=', base, ', query=', query, ', method=', method, ', n_matches:', n_matches, ', min_strength:', min_strength)
    sorted_matches = []
    top_results = []
    timer = Timer()
    timer.start()
    match_result = cv2.matchTemplate(base, query, METHODS[method][0])
    timer.mark('Convolution')
    if n_matches == 1:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        sorted_matches = [[min_loc, min_val], [max_loc, max_val]]
    else:
        bw = base.shape[1]
        zipped_matches = [[(i // bw, i % bw), value] for i, value in enumerate(numpy.array(match_result).flatten())]
        sorted_matches = sorted(zipped_matches, key=lambda match: match[1])
    timer.mark('Match sorting')
    if METHODS[method][1] > 0:
        sorted_matches = [[match[0], 255 - match[1]] for match in sorted_matches]
    else:
        sorted_matches = sorted_matches[::-1]
    timer.mark('Match adjustment')
    if n_matches > 1 and min_strength is not None:
        lbound = 255 * min_strength
        top_results = [match for match in sorted_matches if match[1] >= lbound]
    timer.mark('Match rating')
    if n_matches is not None and len(top_results) < n_matches:
        top_results = sorted_matches[:n_matches]
    timer.mark('Match selection')
    return {'duration': timer.stop(), 'image': generate_plot(base, query, top_results)}
