import argparse
import cv2
import glob
import numpy
import sys
import traceback

FLANN_KD = 2
KEY_BOTTOM_ARROW = 0x10000 * 0x28
KEY_ESC = 0x1B
KEY_LEFT_ARROW = 0x10000 * 0x25
KEY_PG_DOWN = 0x10000 * 0x22
KEY_PG_UP = 0x10000 * 0x21
KEY_RIGHT_ARROW = 0x10000 * 0x27
KEY_SPACE = 0x20
KEY_UP_ARROW = 0x10000 * 0x26
MATCH_RATIO_THRESHOLD = 0.5
MIN_IMAGE_DIMENSION = 10
N_MATCHES = 20
SCALE_MAX = 5.0
SCALE_MIN = 0.2
WIN_HEIGHT = 600
WIN_NAME_CONTROLS = 'controls'
WIN_NAME_MATCHES = 'matches'
WIN_WIDTH = 800


def check_image(name: str, image: numpy.ndarray):
    h, w = image.shape[:2]
    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        raise Exception(f'{name} image is empty')


def check_images(images: dict):
    for name, data in images.items():
        check_image(name, data['image'])


def check_key_points_and_descriptors(images: dict):
    pass
    # for name, data in images.items():
    #     if len(data['key_points']) < 1:
    #         raise Exception(f'{data["filename"]} image only has {len(data["key_points"])} key-points')


def compute_and_store_matches(args: dict, images: dict):
    flann = cv2.FlannBasedMatcher_create()
    for index in range(len(args['query'])):
        query_name = f'query{index}'
        images[query_name]['matches'] = []
        images[query_name]['filtered_matches'] = []
        try:
            images[query_name]['matches'] = flann.knnMatch(images['base']['descriptors'],
                                                           images[query_name]['descriptors'], k=FLANN_KD) \
                if len(images[query_name]['key_points']) > 1 else []
            images[query_name]['filtered_matches'] = filter_matches(images[query_name]['matches'])
            images[query_name]['matches_image'] = plot_matches_and_outline(images, query_name,
                                                                           sorted(images[query_name]['matches'],
                                                                                  key=lambda m: m[0].distance)[
                                                                           :N_MATCHES]
                                                                           if len(images[query_name][
                                                                                      'filtered_matches']) < N_MATCHES
                                                                           else images[query_name]['filtered_matches'])
        except:
            print_traceback(sys.exc_info())
            base_image = images['base']['image']
            query_image = images[query_name]['image']
            bh, bw = base_image.shape[:2]
            qh, qw = query_image.shape[:2]
            images[query_name]['matches_image'] = numpy.zeros((max(bh, qh), bw + qw, 3), numpy.uint8)
            # images[query_name]['matches_image'][0:bh, 0:bw] = base_image[0:bh, 0:bw]
            # images[query_name]['matches_image'][0:qh, bw:bw + qw] = query_image[0:qh, 0:qw]


def create_flann_matcher():
    index_params = dict(algorithm=cv2.DESCRIPTOR_MATCHER_FLANNBASED, trees=100)
    search_params = dict(checks=500)
    return cv2.FlannBasedMatcher(index_params, search_params)


def create_detector():
    # return cv2.ORB_create()  # TODO: play around with arguments to ORB
    return cv2.SIFT_create()  # TODO: play around with arguments to SIFT
    # return cv2.SURF_create()  # TODO: play around with arguments to SURF


def detect_and_store_features(images):
    detector = create_detector()
    for img_name in images:
        # TODO: second argument to detectAndCompute is the mask... might be worth trying later.
        images[img_name]['key_points'], images[img_name]['descriptors'] = \
            detector.detectAndCompute(images[img_name]['image'], None)


def filter_matches(matches: list[list[cv2.DMatch]]):
    """Lowe's ratio test"""
    return [pair for pair in matches if pair[0].distance < MATCH_RATIO_THRESHOLD * pair[1].distance]


def flat_map(xs):
    ys = []
    for x in xs:
        ys.extend(x)
    return ys


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--base", required=True, help="Base (canvas) image filename")
    ap.add_argument("-q", "--query", required=True, nargs='+', help="Query image filename")
    args = vars(ap.parse_args())
    args['query'] = flat_map([glob.glob(query) for query in args['query']])
    return args


def load_images(args):
    images = {
        'base': {
            'filename': args['base'],
            'image': cv2.imread(args['base'], cv2.IMREAD_COLOR),
            'full_image': cv2.imread(args['base'], cv2.IMREAD_COLOR),
        },
    }
    for i, query in enumerate(args['query']):
        images[f'query{i}'] = {
            'filename': str(query),
            'image': cv2.imread(query, cv2.IMREAD_COLOR),
            'full_image': cv2.imread(query, cv2.IMREAD_COLOR)
        }
    return images


def main():
    args = get_args()
    images = load_images(args)
    check_images(images)
    detect_and_store_features(images)
    check_key_points_and_descriptors(images)
    compute_and_store_matches(args, images)
    show_flann_matches_and_outlines(images, list(range(len(args['query']))), initial_scale=0.5)


def match_to_string(match: cv2.DMatch):
    return f'{{i={match.imgIdx}, q={match.queryIdx}, t={match.trainIdx}, d={match.distance}}}'


def on_mouse(event: int, x: int, y: int, flag: int, userdata):
    if flag & cv2.EVENT_FLAG_LBUTTON:
        if 'drag_from' in userdata:
            userdata['center'] = {
                'x': userdata['center']['x'] + userdata['drag_from']['x'] - x,
                'y': userdata['center']['y'] + userdata['drag_from']['y'] - y
            }
        userdata['drag_from'] = {'x': x, 'y': y}
        print(f'>>>>> on mouse: event={event}, x={x}, y={y}, flag={flag}, userdata={userdata}')
    elif ('drag_from' in userdata) and (not flag & cv2.EVENT_FLAG_LBUTTON):
        del userdata['drag_from']
        print(f'>>>>> on mouse: event={event}, x={x}, y={y}, flag={flag}, userdata={userdata}')
    elif event == cv2.EVENT_MOUSEWHEEL:
        _, _, w, h = cv2.getWindowImageRect(WIN_NAME_MATCHES)
        scale = userdata['scale']
        new_scale = scale
        if flag > 0 and scale < SCALE_MAX:
            new_scale = scale + 0.1
        elif userdata['scale'] > SCALE_MIN:
            new_scale = scale - 0.1
        userdata['scale'] = new_scale
        cx, cy = userdata['center']['x'], userdata['center']['y']
        tr_cx, tr_cy = int(cx * new_scale / scale), int(cy * new_scale / scale)
        if new_scale > scale:
            delta_x = (x - w // 2) * (1 - new_scale / scale)
            delta_y = (y - h // 2) * (1 - new_scale / scale)
            # userdata['center'] = {'x': cx + x - w // 2, 'y': cy + y - h // 2}
            userdata['center'] = {'x': tr_cx + delta_x, 'y': tr_cy + delta_y}  # XXX: FIX
        print(f'>>>>> on mouse: event={event}, x={x}, y={y}, flag={flag}, userdata={userdata}')


def plot_matches(images: dict, query_name: str, matches: list[list[cv2.DMatch]]):
    return cv2.drawMatchesKnn(
        images['base']['image'],
        images['base']['key_points'],
        images[query_name]['image'],
        images[query_name]['key_points'],
        matches,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )


def plot_matches_and_outline(images: dict, query_name: str, matches: list[list[cv2.DMatch]]):
    try:
        src_pts = numpy.float32([images[query_name]['key_points'][m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([images['base']['key_points'][m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        t_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = images[query_name]['image'].shape[:2]
        pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, t_matrix)
        matches_image = plot_matches(images, query_name, matches)
        result_image = cv2.polylines(matches_image, [numpy.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    except:
        result_image = plot_matches(images, query_name, matches)
    return result_image


def print_matches(matches: list[list[cv2.DMatch]]):
    print('ALL MATCHES:')
    for match_pair in matches:
        print('    pair:', ' <==> '.join([match_to_string(match) for match in match_pair]))


def print_traceback(exc_info):
    exc_type, exc_value, exc_traceback = exc_info
    print("*** print_tb:")
    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
    print("*** print_exception:")
    # exc_type below is ignored on 3.5 and later
    traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
    print("*** print_exc:")
    traceback.print_exc(limit=2, file=sys.stdout)
    print("*** format_exc, first and last line:")
    formatted_lines = traceback.format_exc().splitlines()
    print(formatted_lines[0])
    print(formatted_lines[-1])
    print("*** format_exception:")
    # exc_type below is ignored on 3.5 and later
    print(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    print("*** extract_tb:")
    print(repr(traceback.extract_tb(exc_traceback)))
    print("*** format_tb:")
    print(repr(traceback.format_tb(exc_traceback)))
    print("*** tb_lineno:", exc_traceback.tb_lineno)


def show_flann_matches_and_outlines(images: dict, indexes: list[int], initial_scale: float = 1.0):
    quitting = False
    i = 0
    view_params = {'scale': initial_scale}
    cv2.namedWindow(WIN_NAME_MATCHES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME_MATCHES, WIN_WIDTH, WIN_HEIGHT)
    # cv2.namedWindow(WIN_NAME_MATCHES, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME_MATCHES, on_mouse, view_params)
    print('Available controls:\n'
          '    cycling queries: Arrows, PgUp/Down, Space\n'
          '    quitting: Q/q, Esc.\n'
          '    zoom-in/out: Mouse Wheel\n'
          '    repositioning: Mouse Drag\n')
    while not quitting:
        index = indexes[i]
        query_name = f'query{index}'
        image = images[query_name]['matches_image']
        if view_params is not None and view_params != 1.0:
            image = cv2.resize(image, (0, 0), fx=view_params['scale'], fy=view_params['scale'])
        h, w = image.shape[:2]
        view_params['img_dimensions'] = {'height': h, 'width': w}
        if 'center' not in view_params:
            view_params['center'] = {'x': w // 2, 'y': h // 2}
        # Fix center:
        _, _, win_width, win_height = cv2.getWindowImageRect(WIN_NAME_MATCHES)[:4]
        cx, cy = view_params['center']['x'], view_params['center']['y']
        if w - cx < win_width // 2:
            view_params['center']['x'] = w - win_width // 2
        elif cx < win_width // 2:
            view_params['center']['x'] = win_width // 2
        elif h - cy < win_height // 2:
            view_params['center']['y'] = h - win_height // 2
        elif cy < win_height // 2:
            view_params['center']['y'] = win_height // 2
        cx, cy = view_params['center']['x'], view_params['center']['y']
        top = int(max(cy - win_height // 2, 0))
        bottom = int(min(cy + win_height // 2, h - 1))
        left = int(max(cx - win_width // 2, 0))
        right = int(min(cx + win_width // 2, w - 1))
        # print(f'cropping: center=({cx}, {cy}), borders={{({left}, {top}), ({right}, {bottom})}}, window={win_width}x{win_height}')
        image = image[top:bottom, left:right]
        win_image = numpy.zeros((win_height, win_width, 3), numpy.uint8)
        win_image[0:image.shape[0], 0:image.shape[1]] = image
        cv2.imshow(WIN_NAME_MATCHES, win_image)
        cv2.setWindowTitle(WIN_NAME_MATCHES, f'{images["base"]["filename"]} <-- {images[query_name]["filename"]}')
        key = cv2.waitKeyEx(1)  # TODO: There has to be a better and more performant way
        if key in [KEY_LEFT_ARROW, KEY_PG_UP, KEY_UP_ARROW]:
            i = (i - 1) % len(indexes)
            del view_params['center']
            view_params['scale'] = initial_scale
        elif key in [KEY_BOTTOM_ARROW, KEY_PG_DOWN, KEY_RIGHT_ARROW, KEY_SPACE]:
            i = (i + 1) % len(indexes)
            del view_params['center']
            view_params['scale'] = initial_scale
        elif key in [KEY_ESC, ord('Q'), ord('q')]:
            quitting = True
        elif key >= 0:
            print('invalid key pressed:', key)
    cv2.destroyWindow(WIN_NAME_MATCHES)


def show_image(name: str, image: numpy.ndarray, scale: float = 1.0):
    if scale != 1.0:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(name, image)
    cv2.waitKeyEx(0)
    cv2.destroyWindow(name)


if __name__ == '__main__':
    main()
