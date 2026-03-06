import argparse
from pathlib import Path
from typing import Optional, Union
from .paths import get_dataset_dir

DATASET_DIR = get_dataset_dir()

PADEL_DIR = DATASET_DIR / "padel"
FOOT_DIR = DATASET_DIR / "foot"

NEW_CAPTATION_DIR = DATASET_DIR / "New_captation"  # TODO move under padel at some point
TESTESPRIT_DIR = PADEL_DIR / "TestEsprit"
SESSION_DIR = PADEL_DIR / "Sessions"

aliases = {
    SESSION_DIR / "sess3/video1.mkv": ["sess3"],
    SESSION_DIR / "sess10/video1.mkv": ["sess10"],
    SESSION_DIR / "sess23/video1.mkv": ["sess23"],
    SESSION_DIR / "sess26_1/video1.mkv": ["sess26"],
    SESSION_DIR / "sess27_1/video1.mkv": ["sess27"],
    SESSION_DIR / "sess39_1/video1.mkv": ["sess39"],
    TESTESPRIT_DIR / "677fb3881420d796784853/video0.mkv": [],
    TESTESPRIT_DIR / "677fb3881420d796784853/video1.mkv": [],
    TESTESPRIT_DIR / "677fb61994e5b894820555/video1.mkv": [],
    TESTESPRIT_DIR / "677fb9460dd87948518490/video1.mkv": [],
    TESTESPRIT_DIR / "677fba8a03821495180828/video1.mkv": [],
    TESTESPRIT_DIR / "677fbbd7a3420043914501/video1.mkv": [],
    TESTESPRIT_DIR / "677fbcc463dce082368534/video1.mkv": [],
    TESTESPRIT_DIR / "677fbf2bd46d0452444895/video1.mkv": [],
    TESTESPRIT_DIR / "677fc3963ed26026640187/video1.mkv": [],
    TESTESPRIT_DIR / "677fc69aba0a3598137386/video1.mkv": [],
    TESTESPRIT_DIR / "677fc70a3d37d145808074/video1.mkv": [],
    TESTESPRIT_DIR / "67a0d096b8fe1662840345/video1-cut.mkv": ["malcolm"],
    TESTESPRIT_DIR / "67a4c260b00e5123828867/video0.mkv": [],
    TESTESPRIT_DIR / "67a4d7b92aeeb091334523/video0.mkv": [],
    TESTESPRIT_DIR / "67a9b47b95de0356558208/video0.mkv": [],
    TESTESPRIT_DIR / "67adfdb89d976570971882/video0.mkv": ["paul"],
    TESTESPRIT_DIR / "67b8790e4f3cd829917368/video0.mkv": ["team-ia"],
    TESTESPRIT_DIR / "67bc52572fb68414218247/video0.mkv": [],
    TESTESPRIT_DIR / "67bc527f9f19c898942112/video0.mkv": [],
    TESTESPRIT_DIR / "67bc5296869d2098402951/video0.mkv": [],
    TESTESPRIT_DIR / "67d17a4f525b6917770031/video0.mkv": ["tristan-paul"],
    TESTESPRIT_DIR / "67d42b8bbe259792483089/video0.mkv": ["tournoi-tristan-paul"],
    TESTESPRIT_DIR / "67e272e746afd127831297/video0.mkv": ["adrien-pablo"],
    TESTESPRIT_DIR / "67e3e0fb125f4991716456/video0.mkv": ["tristan-malcolm"],
    TESTESPRIT_DIR / "685beb9a1aaa8596666254/video1.m3u8": ["team-ia2"],
}


alias_to_path = {}
for path, alias_list in aliases.items():
    for alias in alias_list:
        if alias in alias_to_path:
            raise ValueError(f"Alias {alias} already exists for {alias_to_path[alias]} and {path}")
        alias_to_path[alias] = path


class VideoAliasPathAction(argparse.Action):
    """Process list of video path and apply aliases (if match)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            alias = maybe_get_alias(values)
            setattr(namespace, self.dest, alias)

        elif isinstance(values, (list, tuple)):
            values = [maybe_get_alias(v) for v in values]
            setattr(namespace, self.dest, values)
        else:
            raise TypeError(f"Expected type in [string, list, tuple], got {type(values)}")


def maybe_get_alias(alias_or_path: str) -> str:
    return str(alias_to_path.get(alias_or_path, alias_or_path))


def get_alias(alias: str) -> str:
    if alias in alias_to_path:
        return alias_to_path[alias]
    raise ValueError(f"Alias {alias} not found, choices are {list(alias_to_path.keys())}")


def maybe_get_session_alias(alias_or_path: str) -> str:
    if alias_or_path in alias_to_path:
        path = alias_to_path[alias_or_path]
        session = Path(path).parent.name
        return session
    return alias_or_path


def path_to_aliases(path: str) -> list[str]:
    return aliases.get(Path(path), [])


def path_to_alias(path: str) -> Optional[str]:
    alias_list = path_to_aliases(path)
    if len(alias_list) == 0:
        return None
    return alias_list[0]


def session_name(path: Path) -> str:
    return Path(path).parent.name


def path_session_or_alias(path_or_alias: Union[list, Path], short: bool = False) -> str:
    if isinstance(path_or_alias, str):
        raise TypeError(f"path_or_alias must be a Path, got {type(path_or_alias)}")
    if isinstance(path_or_alias, list) and len(path_or_alias) == 1:
        path_or_alias = path_or_alias[0]
    if isinstance(path_or_alias, Path):
        if path_or_alias in aliases:
            alias_list = aliases[path_or_alias]
            if len(alias_list) != 0:
                return alias_list[0]

    name = session_name(path_or_alias)
    if short:
        name = short_name(name)
    return name


def short_name(name: str) -> str:
    if len(name) > 7:
        return name[:3] + '…' + name[-3:]
    return name
