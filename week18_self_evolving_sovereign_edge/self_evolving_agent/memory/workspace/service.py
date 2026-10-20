def parse(line):
    try:
        line = line.strip()
        if not line:
            return None
        a, b = line.split(',')
        return int(a) / int(b)
    except (AttributeError, ZeroDivisionError, ValueError):
        return None

def run(lines):
    return [result for result in [parse(l) for l in lines] if result is not None]
