import json

def extract_unique_categories(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_cats = set()
    for article in data:
        all_cats.update(article.get("categories", []))

    return sorted(all_cats)

if __name__ == "__main__":
    categories = extract_unique_categories("onepiece_structured.json")
    print(f"✅ Found {len(categories)} unique categories")
    
    # Optionally save to a file
    with open("unique_categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
