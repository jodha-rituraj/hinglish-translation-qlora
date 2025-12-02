from datasets import load_dataset
import json
from pathlib import Path
import ast


def saveJSON(split, outPath, englishCol, hinglishCol):
    outPath.parent.mkdir(parents = True, exist_ok = True)

    items = []
    for row in split:
        input = row[englishCol].strip()
        try:
            outputs = ast.literal_eval(row[hinglishCol].strip())
        except:
            outputs = row[hinglishCol].strip()
        
        if isinstance(outputs, (list, tuple)):
            outputs = [str(i) for i in outputs]
        else:
            outputs = [str(outputs)]

        for output in outputs:
            item = {
                "input": input,
                "output": output.strip()
            }
            items.append(item)

    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(items)} Items At : {outPath}")

def downloadHingeDataset():
    dataset = load_dataset("LingoIITGN/HinGE")
    print(dataset["train"][:5])
    dataDirectory = Path("data")
    dataDirectory.mkdir(exist_ok = True)

    # Save splits into data/
    if "train" in dataset:
        print("Saving train split...")
        saveJSON(dataset["train"], dataDirectory / "hinge_train.json", "English", "Human-generated Hinglish")

    if "validation" in dataset:
        print("Saving validation split...")
        saveJSON(dataset["validation"], dataDirectory / "hinge_validation.json", "English", "Human-generated Hinglish")

    if "test" in dataset:
        print("Saving test split...")
        saveJSON(dataset["test"], dataDirectory / "hinge_test.json", "English", "Human-generated Hinglish")

    print("All files stored in the data/ folder.")

def downloadEngToHinglishDataset():
    dataset = load_dataset("findnitai/english-to-hinglish")

    dataDirectory = Path("data")
    dataDirectory.mkdir(exist_ok = True)

    # Save splits into data/
    if "train" in dataset:
        print("Saving train split...")
        saveJSON(dataset["train"]['translation'], dataDirectory / "eng_hinglish_train.json", "en", "hi_ng")

    if "validation" in dataset:
        print("Saving validation split...")
        saveJSON(dataset["validation"], dataDirectory / "eng_hinglish_validation.json", "en", "hi_ng")

    if "test" in dataset:
        print("Saving test split...")
        saveJSON(dataset["test"], dataDirectory / "eng_hinglish_test.json", "en", "hi_ng")

    print("All files stored in the data/ folder.")


if __name__ == "__main__":
    downloadHingeDataset()
    downloadEngToHinglishDataset()