def load_api_keys(file_path):
    keys = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                keys[key] = value
    return keys


# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.