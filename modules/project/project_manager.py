import os
import json

PROJECTS_FOLDER = "projects"


def initialize():

    os.makedirs(PROJECTS_FOLDER, exist_ok=True)


def create_project(name, keyword, country):

    slug = (
        name.lower()
        .replace(" ", "-")
        .replace("/", "-")
    )

    path = os.path.join(
        PROJECTS_FOLDER,
        slug
    )

    os.makedirs(path, exist_ok=True)

    project = {
        "name": name,
        "keyword": keyword,
        "country": country,
        "status": {
            "serp": False,
            "competitors": False,
            "entities": False,
            "topics": False,
            "research": False,
            "gsc": False,
            "brief": False
        }
    }

    with open(
        os.path.join(path, "project.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            project,
            f,
            indent=4
        )

    return slug


def list_projects():

    initialize()

    folders = []

    for folder in os.listdir(PROJECTS_FOLDER):

        if os.path.isdir(
            os.path.join(PROJECTS_FOLDER, folder)
        ):
            folders.append(folder)

    return sorted(folders)


def load_project(slug):

    with open(
        os.path.join(
            PROJECTS_FOLDER,
            slug,
            "project.json"
        ),
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_project(slug, data):

    with open(
        os.path.join(
            PROJECTS_FOLDER,
            slug,
            "project.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )