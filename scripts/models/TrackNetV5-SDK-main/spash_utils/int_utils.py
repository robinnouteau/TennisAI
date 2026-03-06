

def is_int(nb):
    try:
        _ = int(nb)
        return True
    except (TypeError, ValueError):
        return False
