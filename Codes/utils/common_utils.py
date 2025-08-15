import os


class DictArgs(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def get_files(dir_path, filter_str=None):
    file_list = []
    for filepath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filter_str is None or (filter_str is not None and filter_str in filename):
                file_list.append(os.path.join(filepath, filename))
    return file_list
