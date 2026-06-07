from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        A single string containing text from all pages
    """
    reader = PdfReader(pdf_path)
    full_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Sometimes extract_text() returns None for a page
        if text:
            full_text.append(f"\n--- Page {page_number} ---\n")
            full_text.append(text)
        else:
            full_text.append(f"\n--- Page {page_number} ---\n")
            full_text.append("[No extractable text found on this page]")

    return "\n".join(full_text)


def main():
    pdf_path = Path("data/sample_pdfs/my_document.pdf")

    if not pdf_path.exists():
        print(f"Error: File not found -> {pdf_path}")
        return

    extracted_text = extract_text_from_pdf(str(pdf_path))

    print("\nExtracted text preview:\n")
    print(extracted_text[:3000])  # show first 3000 characters

    print("\n" + "=" * 50)
    print(f"Total characters extracted: {len(extracted_text)}")


if __name__ == "__main__":
    main()