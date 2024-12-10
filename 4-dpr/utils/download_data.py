import argparse
import gzip
import logging
import os
import pathlib
import wget

from typing import Tuple

logger = logging.getLogger(__name__)

RESOURCES_MAP = {
    "data.retriever.squad1-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 train subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.qas.curatedtrec-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "CuratedTrec test subset for Retriever validation and IR results generation",
    },
    # Retriever project data
    "data.wikipedia_split.psgs_w100": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
}

def unpack(gzip_file: str, out_file: str):
    logger.info("Uncompressing %s", gzip_file)
    input = gzip.GzipFile(gzip_file, "rb")
    s = input.read()
    input.close()
    output = open(out_file, "wb")
    output.write(s)
    output.close()
    logger.info(" Saved to %s", out_file)

def download_resource(
    s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str
) -> Tuple[str, str]:
    logger.info("Requested resource from %s", s3_url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    logger.info("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] + (".tmp" if compressed else original_ext)))

    wget.download(s3_url, out=local_file)

    logger.info("Downloaded to %s", local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file)
        os.remove(local_file)
        local_file = uncompressed_file
    return save_root, local_file


def download_file(s3_url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(s3_url, out=local_file)
    logger.info("Downloaded to %s", local_file)
    

def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        logger.info("matched by prefix resources: %s", resources)
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            logger.info("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    s3_url = download_info["s3_url"]

    save_root_dir = None
    data_files = []
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir, local_file = download_resource(
                url,
                download_info["original_ext"],
                download_info["compressed"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
            data_files.append(local_file)
    else:
        save_root_dir, local_file = download_resource(
            s3_url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
        data_files.append(local_file)

    license_files = download_info.get("license_files", None)
    if license_files:
        download_file(license_files[0], save_root_dir, "LICENSE")
        download_file(license_files[1], save_root_dir, "README")
    return data_files

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        logger.warning("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            logger.warning("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()