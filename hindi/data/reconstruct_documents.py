"""Regroup line-per-sentence raw exports (e.g. IndicCorpV2) into real documents.

IndicCorpV2's HF export streams one sentence/paragraph-fragment per record,
with a blank-text record marking the boundary between original documents --
our per-document length/ratio filters need real multi-sentence documents to
work on, not isolated one-line fragments, so this reconstructs them before
the generic cleaning pipeline runs.
"""
import json
import os


def regroup_paragraph_documents(records, source_name):
    """Join consecutive non-blank records into one document per blank-line gap.

    Without this, a source like IndicCorpV2 -- exported as one sentence per
    line -- gets treated as thousands of individual one-sentence "documents",
    most of which fail a reasonable min-word-count filter even though the
    underlying source text is perfectly good prose once reassembled.

    Example: regroup_paragraph_documents([{"text": "पहला वाक्य।"},
                                           {"text": "दूसरा वाक्य।"},
                                           {"text": ""},
                                           {"text": "तीसरा वाक्य।"}], "src")
             -> [{"text": "पहला वाक्य।\\nदूसरा वाक्य।", "source": "src", "doc_id": "src-0"},
                 {"text": "तीसरा वाक्य।", "source": "src", "doc_id": "src-1"}]
    """
    docs = []
    current_lines = []

    def _flush():
        if current_lines:
            docs.append(
                {
                    "text": "\n".join(current_lines),
                    "source": source_name,
                    "doc_id": f"{source_name}-{len(docs)}",
                }
            )
            current_lines.clear()

    for record in records:
        text = record["text"].strip()
        if text == "":
            _flush()
            continue
        current_lines.append(text)
    _flush()

    return docs


def regroup_file(input_path, output_path, source_name):
    """Read a line-per-sentence raw jsonl file and write regrouped documents.

    Streams input record-by-record but buffers the current in-progress
    document, so memory use stays bounded by one document's worth of lines
    rather than the whole file.
    """
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        current_lines = []
        doc_count = 0

        def _flush():
            nonlocal doc_count
            if current_lines:
                fout.write(
                    json.dumps(
                        {
                            "text": "\n".join(current_lines),
                            "source": source_name,
                            "doc_id": f"{source_name}-{doc_count}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                doc_count += 1
                current_lines.clear()

        for line in fin:
            record = json.loads(line)
            text = record["text"].strip()
            if text == "":
                _flush()
                continue
            current_lines.append(text)
        _flush()

    return doc_count


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    input_path = os.path.join(raw_dir, "indiccorp_v2.jsonl")
    output_path = os.path.join(raw_dir, "indiccorp_v2.jsonl.regrouped")
    count = regroup_file(input_path, output_path, "indiccorp_v2")
    os.replace(output_path, input_path)
    print(f"regrouped indiccorp_v2 into {count} documents")
