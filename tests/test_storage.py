from pathlib import Path

from modules.storage.storage import Storage

test = {

    "hello": "world"

}

Storage.save_json(

    Path("workspace/test.json"),

    test

)

print(

    Storage.load_json(

        Path("workspace/test.json")

    )

)