import importlib.util

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super(dotdict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = dotdict(value)

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_and_convert_config()

    def _load_config(self):
        spec = importlib.util.spec_from_file_location("config", self.config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config

    def _convert_to_dotdict(self, module):
        config_dict = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}
        return dotdict(config_dict)

    def _load_and_convert_config(self):
        config_module = self._load_config()
        return self._convert_to_dotdict(config_module)

    def __getattr__(self, item):
        return getattr(self.config, item)
