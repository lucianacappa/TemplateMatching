import cv2
import numpy
import sys
import traceback
from timer import Timer


# TODO: Parameterise
FLANN_KD = 2

PARAMETER_SPECS = {
    'detector': {'type': str, 'options': ['ORB', 'SIFT']},
    'match_ratio_threshold': {'type': float, 'min': 0.01, 'max': 1.00, 'step': 0.01},
    'min_matches': {'type': int, 'min': 1, 'max': 100, 'step': 1},
}


# TODO: Parameterize type
def create_detector(detector: str):
    if detector == 'ORB':
        return cv2.ORB_create()  # TODO: play around with arguments to ORB
    elif detector == 'SIFT':
        return cv2.SIFT_create()  # TODO: play around with arguments to SIFT
    # elif detector == 'SURF': return cv2.SURF_create()  # TODO: play around with arguments to SURF


def create_flann_matcher():
    index_params = dict(algorithm=cv2.DESCRIPTOR_MATCHER_FLANNBASED, trees=100)
    search_params = dict(checks=500)
    return cv2.FlannBasedMatcher(index_params, search_params)


def filter_matches(all_matches, match_ratio_threshold: float, min_matches: int):
    lowe = [pair for pair in all_matches if pair[0].distance < match_ratio_threshold * pair[1].distance]
    return sorted(all_matches, key=lambda m: m[0].distance)[:min_matches] \
        if len(lowe) < min_matches else lowe


def plot(base, query, detector: str, match_ratio_threshold: float, min_matches: int):  # TODO: Add setting arguments
    timer = Timer()
    timer.start()
    detector_object = create_detector(detector)
    timer.mark(f'Detector {detector} creation')
    # TODO: second argument to detectAndCompute is the mask... might be worth trying later.
    base_kp, base_desc = detector_object.detectAndCompute(base, None)
    timer.mark('Base image detection')
    query_kp, query_desc = detector_object.detectAndCompute(query, None)
    timer.mark('Query image detection')
    flann = create_flann_matcher()
    timer.mark('FLANN matcher creation')
    matches_image = None
    try:
        matches = flann.knnMatch(base_desc, query_desc, k=FLANN_KD) if len(query_kp) > 1 else []
        timer.mark('FLANN KNN matching')
        filtered_matches = filter_matches(matches, match_ratio_threshold, min_matches)
        timer.mark('Match filtering')
        matches_image = plot_matches_and_outline(base, base_kp, query, query_kp, filtered_matches)
        timer.mark('Plotting')
    except:
        print_traceback(sys.exc_info())
        timer.mark('Exception handling')
        bh, bw = base.shape[:2]
        qh, qw = query.shape[:2]
        matches_image = numpy.zeros((max(bh, qh), bw + qw, 3), numpy.uint8)
        timer.mark('Blank image creation')
        matches_image[0:bh, 0:bw] = base[0:bh, 0:bw]
        timer.mark('Base image superimposing')
        matches_image[0:qh, bw:bw + qw] = query[0:qh, 0:qw]
        timer.mark('Query image superimposing')
    return {'duration': timer.stop(), 'image': matches_image}


def plot_matches(base, base_kp, query, query_kp, matches: list[list[cv2.DMatch]]):
    return cv2.drawMatchesKnn(base, base_kp, query, query_kp, matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


def plot_matches_and_outline(base, base_kp, query, query_kp, matches: list[list[cv2.DMatch]]):
    result_image = plot_matches(base, base_kp, query, query_kp, matches)
    try:
        src_pts = numpy.float32([query_kp[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([base_kp[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        t_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        qh, qw = query.shape[:2]
        pts = numpy.float32([[0, 0], [0, qh - 1], [qw - 1, qh - 1], [qw - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, t_matrix)
        result_image = cv2.polylines(result_image, [numpy.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    except:
        print_traceback(sys.exc_info())
    return result_image


def print_traceback(exc_info):
    exc_type, exc_value, exc_traceback = exc_info
    traceback.print_tb(exc_traceback, limit=1, file=sys.stderr)
