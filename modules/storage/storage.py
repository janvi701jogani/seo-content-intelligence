import json
from pathlib import Path


class Storage:

    @staticmethod
    def ensure_directory(path: Path):

        path.mkdir(
            parents=True,
            exist_ok=True
        )

    @staticmethod
    def save_json(path: Path, data: dict):

        Storage.ensure_directory(path.parent)

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

    @staticmethod
    def load_json(path: Path):

        if not path.exists():

            return None

        with open(
            path,
            encoding="utf-8"
        ) as f:

            return json.load(f)

    @staticmethod
    def exists(path: Path):

        return path.exists()

    @staticmethod
    def delete(path: Path):

        if path.exists():

            path.unlink()