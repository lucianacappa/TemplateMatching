import threading
from tkinter import BooleanVar, Checkbutton, DoubleVar, Entry, Frame, HORIZONTAL, IntVar, Label, LEFT, OptionMenu, \
    RIGHT, Scale, StringVar, Tk


ON_OFF_VARIABLE_NAME_SUFFIX = '_on/off'


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
        setting = BooleanVar(self._window, spec['default'] if current_value is None else current_value, name)
        return self._pack_controls(name, spec, setting, current_value,
                                   lambda frame: Checkbutton(frame, text=name, variable=setting))

    def _build_float_setting(self, name, spec, current_value: float):
        setting = DoubleVar(self._window, spec['default'] if current_value is None else current_value, name)
        return self._pack_controls(name, spec, setting, current_value,
                                   lambda frame: OptionMenu(frame, setting, *spec['options']) if 'options' in spec else
                                   Scale(frame, orient=HORIZONTAL, from_=spec['min'], resolution=spec['step'],
                                         to=spec['max'], variable=setting))

    def _build_int_setting(self, name, spec, current_value: int):
        setting = IntVar(self._window, spec['default'] if current_value is None else current_value, name)
        return self._pack_controls(name, spec, setting, current_value,
                                   lambda frame: OptionMenu(frame, setting, *spec['options']) if 'options' in spec else
                                   Scale(frame, orient=HORIZONTAL, from_=spec['min'], resolution=spec['step'],
                                         to=spec['max'], variable=setting))

    # noinspection PyMethodMayBeStatic
    def _build_nullable_check(self, name, spec, frame, current_value):
        is_nullable = 'nullable' in spec and spec['nullable']
        onoff_var = BooleanVar(self._window, current_value is not None, f'{name}{ON_OFF_VARIABLE_NAME_SUFFIX}')
        if is_nullable:
            nullable_checkbox = Checkbutton(frame, text='(on/off)', variable=onoff_var)
            nullable_checkbox.pack(side=LEFT)
        return onoff_var

    def _build_str_setting(self, name, spec, current_value: str, callback=None):
        setting = StringVar(self._window, spec['default'] if current_value is None else current_value, name)
        return self._pack_controls(name, spec, setting, current_value,
                                   lambda frame: OptionMenu(frame, setting, *spec['options']) if 'options' in spec else
                                   Entry(frame, setting), callback)

    def _create(self):
        self._window = Tk()
        # self._window.geometry('300x300')
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
        name = args[0]
        if name.endswith(ON_OFF_VARIABLE_NAME_SUFFIX):
            name = name.removesuffix(ON_OFF_VARIABLE_NAME_SUFFIX)
        value = self._current_settings[name]['var'].get()
        on_off = self._current_settings[name]['onoff_var'].get()
        if on_off:
            self._all_settings['current'][name] = value
        elif name in self._all_settings['current']:
            del self._all_settings['current'][name]
        if len(self._listeners) > 0:
            current = self.current_settings()
            for listener in self._listeners:
                listener(**current)

    def _on_change_algorithm(self, *args):
        self._reread_settings()

    def _pack_controls(self, name, spec, setting, current_value, control_builder, callback=None):
        frame = Frame(self._window)
        onoff_var = self._build_nullable_check(name, spec, frame, current_value)
        onoff_cb_name = onoff_var.trace('w', self._on_change)
        label = Label(frame, text=name + ':')
        control = control_builder(frame)
        var_cb_name = setting.trace('w', self._on_change if callback is None else callback)
        label.pack(side=LEFT)
        control.pack(side=RIGHT, fill='x', expand=True)
        frame.pack(fill='both', expand=True)
        return {'container': frame, 'control': control, 'var_cb_name': var_cb_name, 'var': setting,
                'onoff_cb_name': onoff_cb_name, 'onoff_var': onoff_var}

    def _reread_settings(self):
        algorithm = self._algorithm['var'].get()
        for name in self._current_settings:
            setting = self._current_settings[name]
            setting['container'].pack_forget()
            setting['var'].trace_remove('write', setting['var_cb_name'])
            setting['onoff_var'].trace_remove('write', setting['onoff_cb_name'])
        specs = self._algorithm_specs[algorithm]
        current_settings_raw = self._all_settings['all'][algorithm]
        self._all_settings['current'] = current_settings_raw
        self._current_settings = {}
        for spec_name in specs:
            current_value = current_settings_raw[spec_name] if spec_name in current_settings_raw else None
            self._current_settings[spec_name] = \
                ControlsWindow.SETTING_WIDGET_CONSTRUCTORS[specs[spec_name]['type']](self)(
                    spec_name, specs[spec_name], current_value)
        self._window.geometry()

    def add_change_listener(self, listener):
        self._listeners.append(listener)

    def current_settings(self):
        current = {'algorithm': self._algorithm['var'].get()}
        for name in self._current_settings:
            setting = self._current_settings[name]
            value = setting['var'].get()
            on_off = setting['onoff_var'].get()
            if on_off and value is not None:
                current[name] = value
        print('CURRENT SETTINGS:', current)
        return current

    def destroy(self):
        self._window.quit()

    def record_last_duration(self, duration):
        if 'var' in self._last_process_duration:
            content = '\n'.join([f'Last {mark}: {duration[mark]} sec.' for mark in duration])
            self._last_process_duration['var'].set(content)
