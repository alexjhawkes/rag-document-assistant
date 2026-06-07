import re


def clean_text(text: str) -> str:
    # Remove page markers
    text = re.sub(r'---\s*Page\s*\d+\s*---', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Strip each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove lines that are only whitespace/empty caused by PDF spacing
    # Join consecutive non-empty lines together into paragraphs
    # Only treat a sequence of 2+ blank lines as a real paragraph break
    joined_lines = []
    buffer = []
    blank_count = 0

    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count >= 2 and buffer:
                joined_lines.append(' '.join(buffer))
                buffer = []
                joined_lines.append('')
        else:
            blank_count = 0
            buffer.append(line)

    if buffer:
        joined_lines.append(' '.join(buffer))

    # Remove repeated blank lines
    cleaned_lines = []
    prev_blank = False
    for line in joined_lines:
        if line == '':
            if not prev_blank:
                cleaned_lines.append(line)
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    return '\n'.join(cleaned_lines).strip()


def main():
    from pathlib import Path
    from extract_text import extract_text_from_pdf

    pdf_path = Path("data/sample_pdfs/my_document.pdf")
    raw_text = extract_text_from_pdf(str(pdf_path))

    print("BEFORE CLEANING (first 500 chars):")
    print(raw_text[:500])

    print("\n" + "=" * 50 + "\n")

    cleaned = clean_text(raw_text)

    print("AFTER CLEANING (first 500 chars):")
    print(cleaned[:500])


if __name__ == "__main__":
    main()