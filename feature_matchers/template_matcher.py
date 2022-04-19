import cv2
import numpy
from timer import Timer


METHODS = {
    'TM_CCOEFF': {'enum': cv2.TM_CCOEFF, 'mult': -1},
    'TM_CCOEFF_NORMED': {'enum': cv2.TM_CCOEFF_NORMED, 'mult': -1},
    'TM_CCORR': {'enum': cv2.TM_CCORR, 'mult': -1},
    'TM_CCORR_NORMED': {'enum': cv2.TM_CCORR_NORMED, 'mult': -1},
    'TM_SQDIFF': {'enum': cv2.TM_SQDIFF, 'mult': 1},
    'TM_SQDIFF_NORMED': {'enum': cv2.TM_SQDIFF_NORMED, 'mult': 1},
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
        ],
        'default': 'TM_SQDIFF_NORMED',
    },
    'filter_by': {'type': str, 'options': ['number', 'ratio'], 'default': 'number'},
    'n_matches': {'type': int, 'min': 1, 'max': 100, 'step': 1, 'default': 1},
    'match_ratio_threshold': {'type': float, 'nullable': True, 'min': 0.01, 'max': 1.0, 'step': 0.01, 'default': None},
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


def plot(base, query, method: str, filter_by: str = None, n_matches: int = None, match_ratio_threshold: float = None):
    sorted_matches = []
    top_results = []
    timer = Timer()
    timer.start()
    match_result = cv2.matchTemplate(base, query, METHODS[method]['enum'])
    timer.mark('Convolution')
    if filter_by == 'number' and n_matches == 1:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        sorted_matches = [[min_loc, min_val], [max_loc, max_val]]
    else:
        bw = base.shape[1]
        zipped_matches = [[(i // bw, i % bw), value] for i, value in enumerate(numpy.array(match_result).flatten())]
        sorted_matches = sorted(zipped_matches, key=lambda match: match[1])
    timer.mark('Match sorting')
    if METHODS[method]['mult'] > 0:
        sorted_matches = [[match[0], 255 - match[1]] for match in sorted_matches]
    else:
        sorted_matches = sorted_matches[::-1]
    timer.mark('Match adjustment')
    if filter_by == 'ratio':
        lbound = 255 * match_ratio_threshold
        top_results = [match for match in sorted_matches if match[1] >= lbound]
    timer.mark('Match rating')
    if n_matches is not None and len(top_results) < n_matches:
        top_results = sorted_matches[:n_matches]
    timer.mark('Match selection')
    return {'duration': timer.stop(), 'image': generate_plot(base, query, top_results)}
