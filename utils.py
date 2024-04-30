import os, zipfile, shutil, subprocess, shlex, sys # noqa
from urllib.parse import urlparse
import re
import logging


def load_file_from_url(
    url: str,
    model_dir: str,
    file_name: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """Download a file from `url` into `model_dir`,
    using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    # Overwrite
    if os.path.exists(cached_file):
        if overwrite or os.path.getsize(cached_file) == 0:
            remove_files(cached_file)

    # Download
    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    else:
        logger.debug(cached_file)

    return cached_file


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name, extension


def download_manager(
    url: str,
    path: str,
    extension: str = "",
    overwrite: bool = False,
    progress: bool = True,
):
    url = url.strip()

    name, ext = friendly_name(url)
    name += ext if not extension else f".{extension}"

    if url.startswith("http"):
        filename = load_file_from_url(
            url=url,
            model_dir=path,
            file_name=name,
            overwrite=overwrite,
            progress=progress,
        )
    else:
        filename = path

    return filename


def remove_files(file_list):
    if isinstance(file_list, str):
        file_list = [file_list]

    for file in file_list:
        if os.path.exists(file):
            os.remove(file)


def remove_directory_contents(directory_path):
    """
    Removes all files and subdirectories within a directory.

    Parameters:
    directory_path (str): Path to the directory whose
    contents need to be removed.
    """
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
        logger.info(f"Content in '{directory_path}' removed.")
    else:
        logger.error(f"Directory '{directory_path}' does not exist.")


# Create directory if not exists
def create_directories(directory_path):
    if isinstance(directory_path, str):
        directory_path = [directory_path]
    for one_dir_path in directory_path:
        if not os.path.exists(one_dir_path):
            os.makedirs(one_dir_path)
            logger.debug(f"Directory '{one_dir_path}' created.")


def setup_logger(name_log):
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.INFO)

    _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
    _default_handler.flush = sys.stderr.flush
    logger.addHandler(_default_handler)

    logger.propagate = False

    handlers = logger.handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s] >> %(message)s")
        handler.setFormatter(formatter)

    # logger.handlers

    return logger


logger = setup_logger("ss")
logger.setLevel(logging.INFO)

