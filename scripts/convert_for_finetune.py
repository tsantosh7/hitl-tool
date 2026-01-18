import json
import argparse

SYSTEM_PROMPT = "You are a legal information extraction system."

USER_PROMPT_TEMPLATE = """Extract the following 43 legal fields from the judgment text.

Return the result as valid JSON.
Each field must be a list of strings.
If the information is not present, return ["data not available"].

Do not invent facts.
Do not explain your reasoning.
Return JSON only.

JUDGMENT:
{context}
"""


def convert_file(input_path: str, output_path: str):
    n = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}")

            context = row.get("context")
            output_1 = row.get("output_1")

            if not context or not output_1:
                raise RuntimeError(
                    f"Missing context or output_1 on line {line_no}"
                )

            # output_1 is stored as a JSON string in your data
            if isinstance(output_1, str):
                try:
                    output_1 = json.loads(output_1)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Invalid output_1 JSON on line {line_no}: {e}"
                    )
            else:
                output_1 = output_1
                output_1 = output_1

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(context=context),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(output_1, ensure_ascii=False),
                    },
                ]
            }

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n += 1

    print(f"Converted {n} examples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL (train or test)")
    parser.add_argument("--output", required=True, help="Output JSONL for fine-tuning")
    args = parser.parse_args()

    convert_file(args.input, args.output)
