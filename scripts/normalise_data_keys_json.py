import json
from typing import Dict, Any
from urllib.parse import urlparse, urlunparse


def normalise_canonical_url(url: str | None) -> str | None:
    """
    Fix National Archives caselaw URLs by removing the erroneous `/id/` segment.
    Example:
    https://caselaw.nationalarchives.gov.uk/id/ewca/crim/2023/1242
    -> https://caselaw.nationalarchives.gov.uk/ewca/crim/2023/1242
    """
    if not url:
        return None

    try:
        parsed = urlparse(url)
        if parsed.netloc == "caselaw.nationalarchives.gov.uk":
            path = parsed.path
            if path.startswith("/id/"):
                path = path.replace("/id/", "/", 1)
                return urlunparse(parsed._replace(path=path))
    except Exception:
        pass

    return url


def map_record_to_normalised(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a single source record into the normalised schema.
    """

    canonical_url = normalise_canonical_url(
        record.get("canonical_url") or record.get("uri")
    )

    return {
        "document_id": record.get("document_id") or record.get("_id"),
        "canonical_url": canonical_url,
        "published_date": record.get("published_date") or record.get("publicationDate"),
        "doc_type": record.get("type") or record.get("court"),
        "title": record.get("citation"),
        "excerpt": record.get("excerpt") or record.get("summary"),
        "content_text": record.get("content_text") or record.get("content"),
        "source": record.get("source"),

        "metadata": {
            "citation": record.get("citation"),
            "signature": record.get("signature"),
            "xml_uri": record.get("xml_uri"),
            "file_name": record.get("file_name"),
            "judges": record.get("judges"),
            "caseNumbers": record.get("caseNumbers"),
            "citation_references": record.get("citation_references"),
            "legislation": record.get("legislation"),
            "appeal_type": record.get("appeal_type"),
            "appeal_outcome": record.get("appeal_outcome"),
        }
    }


root_path = "/home/stirunag/work/github/JuDDGES/nbs/Data/england-wales/"
INPUT_PATH = root_path + "Instruction Data/england_wales_data_refined_7.jsonl"
OUTPUT_PATH = "data/normalised_data.jsonl"


def normalise_jsonl(input_path: str, output_path: str) -> None:
    total_docs = 0
    non_null_content = 0
    non_null_excerpt = 0
    non_null_title = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            try:
                source_record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_number}: {e}")
                continue

            normalised_record = map_record_to_normalised(source_record)
            outfile.write(json.dumps(normalised_record, ensure_ascii=False) + "\n")

            total_docs += 1

            if normalised_record.get("content_text"):
                non_null_content += 1
            if normalised_record.get("excerpt"):
                non_null_excerpt += 1
            if normalised_record.get("title"):
                non_null_title += 1

    print("Normalisation complete")
    print(f"Total documents: {total_docs}")
    print(
        f"content_text present: {non_null_content} "
        f"({(non_null_content / total_docs * 100):.2f}%)"
        if total_docs else "content_text present: 0"
    )
    print(
        f"excerpt present: {non_null_excerpt} "
        f"({(non_null_excerpt / total_docs * 100):.2f}%)"
        if total_docs else "excerpt present: 0"
    )
    print(
        f"title present: {non_null_title} "
        f"({(non_null_title / total_docs * 100):.2f}%)"
        if total_docs else "title present: 0"
    )


if __name__ == "__main__":
    normalise_jsonl(INPUT_PATH, OUTPUT_PATH)

