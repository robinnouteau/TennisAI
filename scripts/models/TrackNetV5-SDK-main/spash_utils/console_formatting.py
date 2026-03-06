GREEN = "32"
YELLOW = "33"
ORANGE = "38;5;202"
RED = "31"
GRAY = "90"
BLUE = "34"
PURPLE = "36"

PASS = "✅"
FAIL = "❌"
ERROR = "💥"
SKIP = "⏭️"
TARGET = "🌀"


DISABLE_COLOR = False


def color(text, color_code):
    if DISABLE_COLOR:
        return text
    return f"\033[{color_code}m{text}\033[0m"


def bold(text):
    if DISABLE_COLOR:
        return text
    return f"\033[1m{text}\033[0m"


def color_pass(text):
    return bold(color(text, GREEN))


def color_fail(text):
    return bold(color(text, ORANGE))


def color_error(text):
    return bold(color(text, RED))


def color_skip(text):
    return bold(color(text, GRAY))


def format_pass(text):
    return f"{PASS} {color_pass(text)}"


def format_fail(text):
    return f"{FAIL} {color_fail(text)}"


def format_skip(text):
    return f"{SKIP} {color_skip(text)}"


def format_error(text):
    return f"{ERROR} {color_error(text)}"

