def human_readable(v, step=1024, unit=['B', 'KB', 'MB', 'GB', 'TB']):
    assert step > 0
    
    idx = 0
    while v > step:
        v /= step
        idx += 1
    return f'{v:.4f}{unit[min(idx, len(unit)-1)]}'