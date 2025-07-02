import xml.etree.ElementTree as ET
import mwparserfromhell
import json
from tqdm import tqdm

def extract_categories(wikicode):
    # Collect categories from wikilinks
    categories = [
        str(link.title).split(":", 1)[1].strip()
        for link in wikicode.ifilter_wikilinks()
        if str(link.title).lower().startswith("category:")
    ]
    return list(set(categories))  # remove duplicates

def extract_headings(wikicode):
    return [
        str(h.title).strip()
        for h in wikicode.filter_headings()
        if h.level in {2, 3}
    ]

def parse_structured_articles(xml_path, output_path=None):
    context = ET.iterparse(xml_path, events=("end",))
    articles = []

    for event, elem in tqdm(context, desc="Parsing articles"):
        if elem.tag.endswith("page"):
            ns = elem.find("./{*}ns").text
            if ns != "0":
                elem.clear()
                continue  # Skip non-main articles

            title = elem.find("./{*}title").text
            text_elem = elem.find(".//{*}text")
            raw_text = text_elem.text or ""

            wikicode = mwparserfromhell.parse(raw_text)
            content = wikicode.strip_code().strip()
            if len(content) < 300:
                elem.clear()
                continue  # Skip stubs

            headings = extract_headings(wikicode)
            categories = extract_categories(wikicode)

            articles.append({
                "title": title,
                "categories": categories,
                "headings": headings,
                "content": content
            })

            elem.clear()

    print(f"\n✅ Parsed {len(articles)} structured articles with full preservation.")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

    return articles

if __name__ == "__main__":
    parse_structured_articles("onepiece_dump.xml", output_path="onepiece_structured.json")
