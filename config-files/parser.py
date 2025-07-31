import xml.etree.ElementTree as ET
import mwparserfromhell
from tqdm import tqdm

def parse_wiki_dump(xml_path, output_path=None):
    context = ET.iterparse(xml_path, events=("end",))
    pages = []

    for event, elem in tqdm(context, desc="Parsing pages"):
        if elem.tag.endswith("page"):
            title = elem.find("./{*}title").text
            ns = elem.find("./{*}ns").text
            text_elem = elem.find(".//{*}text")
            text = text_elem.text if text_elem is not None else ""

            # Only main article content (ns == "0")
            if ns == "0" and text:
                wikicode = mwparserfromhell.parse(text)
                plain_text = wikicode.strip_code()

                if len(plain_text.strip()) > 200:  # Ignore stubs
                    pages.append({
                        "title": title,
                        "content": plain_text.strip()
                    })

            # Clear from memory
            elem.clear()

    print(f"\nParsed {len(pages)} main articles.")
    
    if output_path:
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

    return pages

if __name__ == "__main__":
    articles = parse_wiki_dump("onepiece_dump.xml", output_path="onepiece_clean.json")
