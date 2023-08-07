def str2bool(v) -> bool:
    '''
        Convert string to boolean, basically used by the cmd parser.
    '''
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")