import json

from datasets import load_dataset


def load_texts(args):
    """Load model inputs from chat JSON or HuggingFace dataset split."""
    # Accept either chat JSON turns or an HF dataset split.
    if args.chat_json:
        print("Loading chat turns...")
        with open(args.chat_json, "r", encoding="utf-8") as f:
            chat_items = json.load(f)
        if not isinstance(chat_items, list):
            raise ValueError("--chat-json must contain a JSON list of {role, content} objects")
        print(f"  {len(chat_items)} turns loaded")
        return chat_items

    print("Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)
    texts = [row["text"] for row in ds]
    print(f"  {len(texts)} chunks loaded")
    return texts
