import time


class TimerError(Exception):
    pass


class Timer:
    RESERVED = ['partial', 'total']

    def __init__(self):
        self._first = None
        self._last = None
        self._start_time = None
        self._times = {}

    def current_total(self):
        return time.perf_counter() - self._start_time

    def mark(self, name: str):
        if name in Timer.RESERVED:
            raise TimerError(f'Mark name "{name}" not allowed.')

        self._times[name] = time.perf_counter() - self._start_time
        if self._last:
            self._times[name] -= self._times[self._last]
        else:
            self._first = name
        self._last = name

    def partial(self):
        partial_time = self.current_total()
        partial_times = self._times.copy()
        if 'total' in partial_times:
            del partial_times['total']
        partial_times['partial'] = partial_time
        return partial_times

    def start(self):
        self._first = None
        self._last = None
        self._times = {}
        self._start_time = time.perf_counter()

    def stop(self, add_tail: bool = False):
        if add_tail:
            self._times['total'] = self.current_total()
        else:
            self._times['total'] = 0.0
            self._times['total'] = sum(self._times.values())
        return self._times.copy()
