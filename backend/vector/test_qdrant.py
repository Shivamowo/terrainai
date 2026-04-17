from qdrant_store import TerrainVectorStore
import json
from pathlib import Path

LOG_PATH = Path("outputs/training_logs.json")


def check_logs():
    if not LOG_PATH.exists():
        print("❌ training_logs.json missing")
        return

    data = json.loads(LOG_PATH.read_text())

    print("\n📊 TRAINING LOGS")
    print("Total epochs:", len(data))

    if len(data) > 0:
        last = data[-1]
        print("Last epoch:", last["epoch"])
        print("mIoU:", last["metrics"]["miou"])
        print("loss:", last["training"]["loss"])


def check_qdrant():
    store = TerrainVectorStore()

    print("\n🧠 QDRANT DB")

    count = store.client.count(
        collection_name="terrain_patches",
        exact=True
    )

    print("Total vectors stored:", count.count)

    # sample retrieval
    results = store.client.search(
        collection_name="terrain_patches",
        query_vector=[1.0] * 512,
        limit=5,
        with_payload=True
    )

    print("\nSample records:")
    for r in results:
        p = r.payload
        print({
            "class_id": p["class_id"],
            "iou": p["iou"],
            "epoch": p["epoch"],
            "image_path": p["image_path"]
        })


def check_consistency():
    store = TerrainVectorStore()

    results = store.client.scroll(
        collection_name="terrain_patches",
        limit=100,
        with_payload=True
    )[0]

    epochs = set()
    class_counts = {}

    for r in results:
        p = r.payload
        epochs.add(p["epoch"])

        cid = p["class_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1

    print("\n📌 CONSISTENCY CHECK")
    print("Unique epochs stored:", len(epochs))
    print("Class distribution:", class_counts)


def main():
    print("\n================ PIPELINE VALIDATION ================\n")

    check_logs()
    check_qdrant()
    check_consistency()

    print("\n====================================================\n")


if __name__ == "__main__":
    main()