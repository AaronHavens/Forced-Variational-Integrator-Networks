from os import path


def learned_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../learned_models'))
