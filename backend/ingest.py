import json
from backend.vector.qdrant_store import TerrainVectorStore

LOG_PATH = "outputs/hard_examples.json"


def load_data():
    with open(LOG_PATH, "r") as f:
        return json.load(f)


def ingest():
    store = TerrainVectorStore()

    data = load_data()

    print(f"Ingesting {len(data)} samples into Qdrant...")

    store.index_batch(data)

    print("Done ✔")


if __name__ == "__main__":
    ingest()