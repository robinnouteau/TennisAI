import platform
from typing import Optional
from loguru import logger
import sys
import os
import sentry_sdk
from inference.version import __version__
from spash_utils.env_utils import env_get_sport


def get_log_filename(video_path):
    base_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(base_name)[0]
    log_filename = f"{name_without_ext}.log"
    return log_filename


# Définir une fonction de filtre personnalisée
def filter(record):
    # Permettre les messages DEBUG pour 'accuracy_prediction.py'
    if record["file"].name == "accuracy_prediction.py":
        return True
    # Permettre les messages de niveau INFO ou supérieur pour les autres fichiers
    elif record["level"].no >= logger.level("INFO").no:
        return True
    else:
        return False


FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{extra}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
FORMAT_SHORT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
FORMAT_XS = "<level>{message}</level>"


def configure_stdout_logging():
    logger.remove()

    # Ajouter un sink pour la console avec le filtre personnalisé
    logger.add(
        sys.stdout,
        level="DEBUG",  # Niveau minimum pour capturer tous les logs
        filter=filter,  # Appliquer le filtre personnalisé
        format=FORMAT
    )


def add_sentry_logger(video_path=''):
    sentry_sdk.init(
        dsn="https://005a322ac1c99d2a481f50298a1c030a@o416787.ingest.us.sentry.io/4508053663907840",
        traces_sample_rate=0.1,
        environment='prod',
        server_name=os.getenv('CENTER', None)
    )

    # Définir le handler Sentry
    def sentry_handler(message):
        if message.record["level"].no >= logger.level("ERROR").no:
            mdl = message.record.get('module', None)
            if mdl is not None and str(mdl).startswith('pika'):
                # Ignore pika error message
                return
            exception = message.record.get("exception", None)
            if exception:
                sentry_sdk.capture_exception(exception)
            else:
                sentry_sdk.capture_message(message)

    # Ajouter le sink Sentry
    logger.add(sentry_handler, level="ERROR")

    sentry_sdk.set_tags({'ia.platform': platform.platform(),
                         'ia.center': os.getenv('CENTER', ''),
                         'ia.version': __version__,
                         'ia.sport': env_get_sport()})

    with sentry_sdk.configure_scope() as scope:
        if video_path:  # not none or empty
            scope.set_tag("video_filename", video_path)


def configure_logging(
        video_path,
        environment,
        debug: Optional[list[str]] = None,
        trace: Optional[list[str]] = None,
        format: str = FORMAT):
    log_filename = get_log_filename(video_path)

    logger.remove()

    # Ajouter un sink pour la console avec le filtre personnalisé
    logger.add(
        sys.stdout,
        level="DEBUG",  # Niveau minimum pour capturer tous les logs
        filter=filter,  # Appliquer le filtre personnalisé
        format=format,
    )

    # Ajouter un sink pour le fichier de log avec rotation, rétention et le filtre personnalisé
    logger.add(
        log_filename,
        rotation="100 MB",
        retention="1 day",
        level="INFO",
        filter=filter,
        format="{time} | {level} | {message}"
    )

    debug_patterns = debug or []
    for pattern in debug_patterns:
        logger.add(sys.stdout, level="DEBUG", filter=lambda record, p=pattern: p in record["file"].name, format=format)

    trace_patterns = trace or []
    for pattern in trace_patterns:
        logger.add(sys.stdout, level="TRACE", filter=lambda record, p=pattern: p in record["file"].name, format=format)

    if environment == 'prod':
        add_sentry_logger(video_path)
