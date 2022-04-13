import threading
from tkinter import BooleanVar, Checkbutton, DoubleVar, Entry, Frame, HORIZONTAL, IntVar, Label, LEFT, OptionMenu, \
    RIGHT, Scale, StringVar, Tk


class ControlsWindow:
    SETTING_WIDGET_CONSTRUCTORS = {
        bool: lambda self: self._build_bool_setting,
        float: lambda self: self._build_float_setting,
        int: lambda self: self._build_int_setting,
        str: lambda self: self._build_str_setting,
    }

    def __init__(self, algorithm_specs, initial_settings):
        self._algorithm_specs = algorithm_specs
        self._all_settings = initial_settings
        self._current_settings = {}
        self._last_process_duration = {}
        self._listeners = []
        self._variable = None
        self._window = None
        self._thread = threading.Thread(target=self._create)
        self._thread.start()

    def _build_bool_setting(self, name, spec, current_value: bool):
        setting = BooleanVar(self._window, current_value, name)
        control = Checkbutton(self._window, text=name, variable=setting)
        cb_name = setting.trace('w', self._on_change)
        control.pack(fill='x', expand=True)
        return {'cb_name': cb_name, 'container': control, 'control': control, 'var': setting}

    def _build_float_setting(self, name, spec, current_value: float):
        frame = Frame(self._window)
        label = Label(frame, text=name + ':')
        setting = DoubleVar(self._window, current_value, name)
        if 'options' in spec:
            control = OptionMenu(frame, setting, *spec['options'])
        else:
            control = Scale(frame, orient=HORIZONTAL, from_=spec['min'], resolution=spec['step'], to=spec['max'],
                            variable=setting)
        cb_name = setting.trace('w', self._on_change)
        label.pack(side=LEFT)
        control.pack(side=RIGHT, fill='x', expand=True)
        frame.pack(fill='x', expand=True)
        return {'cb_name': cb_name, 'container': frame, 'control': control, 'var': setting}

    def _build_int_setting(self, name, spec, current_value: int):
        frame = Frame(self._window)
        label = Label(frame, text=name + ':')
        setting = IntVar(self._window, current_value, name)
        if 'options' in spec:
            control = OptionMenu(frame, setting, *spec['options'])
        else:
            control = Scale(frame, orient=HORIZONTAL, from_=spec['min'], resolution=spec['step'], to=spec['max'],
                            variable=setting)
        cb_name = setting.trace('w', self._on_change)
        label.pack(side=LEFT)
        control.pack(side=RIGHT, fill='x', expand=True)
        frame.pack(fill='x', expand=True)
        return {'cb_name': cb_name, 'container': frame, 'control': control, 'var': setting}

    def _build_str_setting(self, name, spec, current_value: str, callback=None):
        callback = callback if callback else self._on_change
        frame = Frame(self._window)
        label = Label(frame, text=name + ':')
        setting = StringVar(self._window, current_value, name)
        if 'options' in spec:
            control = OptionMenu(frame, setting, *spec['options'])
        else:
            control = Entry(frame, setting)
        cb_name = setting.trace('w', callback)
        label.pack(side=LEFT)
        control.pack(side=RIGHT, fill='x', expand=True)
        frame.pack(fill='x', expand=True)
        return {'cb_name': cb_name, 'container': frame, 'control': control, 'var': setting}

    def _create(self):
        self._window = Tk()
        self._window.geometry('300x300')
        self._last_process_duration['var'] = StringVar(self._window, 'Last process duration: 0.0',
                                                       'last_process_duration')
        self._last_process_duration['control'] = Label(self._window, anchor='w', justify=LEFT,
                                                       textvariable=self._last_process_duration['var'])
        self._last_process_duration['control'].pack(fill='x')
        algorithms = list(self._algorithm_specs.keys())
        algorithm = self._all_settings['current']['algorithm']
        self._algorithm = self._build_str_setting('algorithm', {'options': algorithms}, algorithm,
                                                  self._on_change_algorithm)
        self._reread_settings()
        self._window.mainloop()

    # noinspection PyMethodMayBeStatic
    def _on_change(self, *args):
        # print('changed', args, 'to:', self._settings[args[0]]['var'].get())
        # print('current_settings:', self.current_settings())
        name = args[0]
        value = self._current_settings[name]['var'].get()
        self._all_settings['current'][name] = value
        for listener in self._listeners:
            listener(**self.current_settings())

    def _on_change_algorithm(self, *args):
        # print('changed the algorithm:', args)
        algorithm = self._algorithm['var'].get()
        # print('value:', algorithm)
        self._reread_settings()

    def _reread_settings(self):
        algorithm = self._algorithm['var'].get()
        for name in self._current_settings:
            setting = self._current_settings[name]
            setting['container'].pack_forget()
            setting['var'].trace_remove('write', setting['cb_name'])
        specs = self._algorithm_specs[algorithm]
        current_settings_raw = self._all_settings['all'][algorithm]
        self._all_settings['current'] = current_settings_raw
        # print('current_settings_raw:', current_settings_raw)
        self._current_settings = {}
        for spec_name in specs:
            current_value = current_settings_raw[spec_name] if spec_name in current_settings_raw else None
            self._current_settings[spec_name] = \
                ControlsWindow.SETTING_WIDGET_CONSTRUCTORS[specs[spec_name]['type']](self)(
                    spec_name, specs[spec_name], current_value)

    def add_change_listener(self, listener):
        self._listeners.append(listener)

    def current_settings(self):
        current = {'algorithm': self._algorithm['var'].get()}
        for name in self._current_settings:
            setting = self._current_settings[name]
            current[name] = setting['var'].get()
        return current

    def destroy(self):
        self._window.quit()

    def record_last_duration(self, duration):
        if 'var' in self._last_process_duration:
            content = '\n'.join([f'Last {mark}: {duration[mark]} sec.' for mark in duration])
            self._last_process_duration['var'].set(content)
