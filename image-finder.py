import argparse
import cv2
import glob
import os
from feature_matchers import flann_matcher, template_matcher
import yaml
from controls_window import ControlsWindow
from plot_window import PlotWindow
from utils import flat_map


KEY_BOTTOM_ARROW = 0x10000 * 0x28
KEY_ESC = 0x1B
KEY_F1 = 0x700000
KEY_F5 = 0x740000  # 0x700000 + 0x10000 * (5 - 1)
KEY_LEFT_ARROW = 0x10000 * 0x25
KEY_PG_DOWN = 0x10000 * 0x22
KEY_PG_UP = 0x10000 * 0x21
KEY_RIGHT_ARROW = 0x10000 * 0x27
KEY_SPACE = 0x20
KEY_UP_ARROW = 0x10000 * 0x26
PLOT_FUNCTIONS = {
    'FLANN': flann_matcher.plot,
    'MatchTemplate': template_matcher.plot,
}


def create_controls_window(plot_window, base, query, query_index_ref, algorithm_specs, view_settings_ref):
    controls_window = ControlsWindow({
        'FLANN': flann_matcher.PARAMETER_SPECS,
        'MatchTemplate': template_matcher.PARAMETER_SPECS,
    }, view_settings_ref)
    return controls_window


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--base', help='Base (canvas) image filename')
    ap.add_argument('-c', '--config', help='Configuration file', default='config.yaml')
    ap.add_argument('-q', '--query', nargs='+', help='Query image filename')
    ap.add_argument('-w', '--window-dimensions', help='Window dimensions ("{width}x{height}", e.g.: "800x600")',
                    default='800x600')
    args = vars(ap.parse_args())
    with open(args['config'], 'r', encoding='utf-8') as f:
        data = yaml.load(f.read(), yaml.Loader)
        if not args['base']:
            assert 'base' in data, 'Need base filename either on command line or in config file.'
            args['base'] = data['base']
        base = args['base']
        assert base, 'Need base filename either on command line or in config file.'
        assert os.path.isfile(base) and os.access(base, os.R_OK), f'File {base} does not exist or is not readable.'
        if not args['query']:
            args['query'] = data['query']
        assert args['query'], 'Need query path list either on command line or in config file.'
        args['query'] = flat_map([glob.glob(query) for query in args['query']])
        assert 'algorithms' in data, 'Must specify algorithms.'
        args['algorithms'] = data['algorithms']
    return args


def load_images(base_filename: str, query_filenames: list[str]):
    return [
        {'filename': base_filename, 'image': cv2.imread(base_filename, -1)},
        [{'filename': query_filename, 'image': cv2.imread(query_filename, -1)} for query_filename in query_filenames]
    ]


def main():
    args = get_args()
    base, query = load_images(args['base'], args['query'])
    query_index = {'query_index': 0}
    plot_window = PlotWindow()
    view_settings = {
        'all': {
            'FLANN': {'algorithm': 'FLANN', 'detector': 'SIFT', 'match_ratio_threshold': 0.5, 'min_matches': 10},
            'MatchTemplate': {'algorithm': 'MatchTemplate', 'method': 'TM_SQDIFF_NORMED', 'n_matches': 1,
                              'min_strength': 0.5},
        },
    }
    view_settings['current'] = view_settings['all']['MatchTemplate']
    controls_window = create_controls_window(plot_window, base, query, query_index, args['algorithms'], view_settings)
    run_plots(plot_window, controls_window, base, query, query_index, view_settings)
    controls_window.destroy()


def plot(base, query, algorithm: str, **kwargs):
    # print('plot: algorithm=', algorithm, ', kwargs:', kwargs)
    return PLOT_FUNCTIONS[algorithm](base, query, **kwargs)


def run_plots(plot_window, controls_window, base, query, query_index_ref, settings_ref):
    last_index = None
    last_params = None
    last_window_dimensions = None
    quitting = False
    while not quitting:
        if last_params != settings_ref['current'] or last_index != query_index_ref['query_index']:
            last_index = query_index_ref['query_index']
            last_params = settings_ref['current'].copy()
            last_window_dimensions = plot_window.get_window_dimensions()
            a_query = query[last_index]
            plotted = plot(base['image'], a_query['image'], **last_params)
            controls_window.record_last_duration(plotted['duration'])
            plot_window.set_plot_image(plotted['image'], f'{base["filename"]} <-- {a_query["filename"]}')
        elif last_window_dimensions != plot_window.get_window_dimensions():
            last_window_dimensions = plot_window.get_window_dimensions()
            plot_window.redraw()
        key = cv2.waitKeyEx(1)  # TODO: There has to be a better and more performant way to detect resizes
        if key in [KEY_LEFT_ARROW, KEY_PG_UP, KEY_UP_ARROW]:
            query_index_ref['query_index'] = (query_index_ref['query_index'] - 1) % len(query)
        elif key in [KEY_BOTTOM_ARROW, KEY_PG_DOWN, KEY_RIGHT_ARROW, KEY_SPACE]:
            query_index_ref['query_index'] = (query_index_ref['query_index'] + 1) % len(query)
        elif key in [KEY_ESC, ord('Q'), ord('q')]:
            quitting = True
        elif key in [KEY_F5, ord('R'), ord('r')]:
            last_index = None
            last_params = None
        elif key in [KEY_F1]:
            print('====================================================')
            print('HELP')
            print('----------------------------------------------------')
            print('previous/next image: Arrows, PgUp/Down, Space (next)')
            print('quit:                Esc, q/Q')
            print('re-process:          F5, r/R')
            print('this help message:   F1')
            print('====================================================')
        elif key >= 0:
            print('invalid key pressed:', key, '. Press F1 for help.')
    plot_window.destroy()


if __name__ == '__main__':
    main()
