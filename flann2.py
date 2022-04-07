import argparse
import cv2
import glob
import numpy

FLANN_KD = 2
MATCH_RATIO_THRESHOLD = 0.8
MIN_IMAGE_DIMENSION = 10


def check_image(name: str, image: numpy.ndarray):
    h, w = image.shape
    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        raise Exception(f'{name} image is empty')


def check_images(images: dict):
    for name, data in images.items():
        check_image(name, data['image'])


def check_key_points_and_descriptors(images: dict):
    for name, data in images.items():
        if len(data['key_points']) < 2:
            raise Exception(f'{name} image only has {len(data["key_points"])} key-points')


def create_flann_matcher():
    index_params = dict(algorithm=cv2.DESCRIPTOR_MATCHER_FLANNBASED, trees=100)
    search_params = dict(checks=500)
    return cv2.FlannBasedMatcher(index_params, search_params)


def create_detector():
    return cv2.SIFT_create()  # TODO: play around with arguments to SIFT


def detect_and_store_features(images):
    detector = create_detector()
    for img_name in images:
        # TODO: second argument to detectAndCompute is the mask... might be worth trying later.
        images[img_name]['key_points'], images[img_name]['descriptors'] =\
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
        'base': {'filename': args['base'], 'image': cv2.imread(args['base'], cv2.IMREAD_GRAYSCALE)},
    }
    for i, query in enumerate(args['query']):
        images[f'query{i}'] = {
            'filename': str(query),
            'image': cv2.imread(query, cv2.IMREAD_GRAYSCALE),
            'full_image': cv2.imread(query, cv2.IMREAD_COLOR)
        }
    return images


def main():
    args = get_args()
    images = load_images(args)
    check_images(images)
    detect_and_store_features(images)
    check_key_points_and_descriptors(images)
    flann = cv2.FlannBasedMatcher_create()
    for i in range(len(args['query'])):
        query_name = f'query{i}'
        matches = flann.knnMatch(images['base']['descriptors'], images[query_name]['descriptors'], k=FLANN_KD)
        filtered_matches = filter_matches(matches)
        show_matches_and_outline(images, query_name, filtered_matches)


def match_to_string(match: cv2.DMatch):
    return f'{{i={match.imgIdx}, q={match.queryIdx}, t={match.trainIdx}, d={match.distance}}}'


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
        # matches_mask = mask.ravel().tolist()  # ???
        h, w = images[query_name]['image'].shape[:2]
        pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, t_matrix)
        matches_image = plot_matches(images, query_name, matches)
        return cv2.polylines(matches_image, [numpy.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)  # Red outline
    except:
        return plot_matches(images, query_name, matches)


def print_matches(matches: list[list[cv2.DMatch]]):
    print('ALL MATCHES:')
    for match_pair in matches:
        print('    pair:', ' <==> '.join([match_to_string(match) for match in match_pair]))


def show_image(name: str, image: numpy.ndarray, scale: float = 1.0):
    if scale != 1.0:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def show_matches(images: dict, query_name: str, matches: list[list[cv2.DMatch]]):
    show_image(
        f'Matches for {images[query_name]["filename"]} on {images["base"]["filename"]}',
        plot_matches(images, query_name, matches),
        scale=0.5
    )


def show_matches_and_outline(images: dict, query_name: str, matches: list[list[cv2.DMatch]]):
    show_image(
        f'Matches for {images[query_name]["filename"]} on {images["base"]["filename"]}',
        plot_matches_and_outline(images, query_name, matches),
        scale=0.5
    )


if __name__ == '__main__':
    main()
