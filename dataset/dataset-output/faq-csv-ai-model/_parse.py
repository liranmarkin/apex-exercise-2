#!/usr/bin/env python3

import sys

def parse_line(line: str):
    line = line.strip()

    if not line:
        return None

    first_comma = line.find(',')
    last_comma = line.rfind(',')

    if first_comma == -1 or last_comma == -1 or first_comma == last_comma:
        raise ValueError(f"Invalid line format (needs at least 2 commas): {line}")

    question = line[:first_comma].strip()
    answer = line[first_comma + 1:last_comma].strip()
    url = line[last_comma + 1:].strip()

    return {
        "question": question,
        "answer": answer,
        "url": url
    }


def main(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                parsed = parse_line(line)
                if parsed:
                    print(f"Line {i}:")
                    print(f"  Question: {parsed['question']}")
                    print(f"  Answer:   {parsed['answer']}")
                    print(f"  URL:      {parsed['url']}")
                    print()
            except Exception as e:
                print(f"Error parsing line {i}: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_file>")
        sys.exit(1)

    main(sys.argv[1])
