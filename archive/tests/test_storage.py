from modules.storage.storage import Storage
from modules.storage.paths import WORKSPACE

test = {
    "hello": "world"
}

path = WORKSPACE / "test.json"

Storage.save_json(path, test)

print(Storage.load_json(path))