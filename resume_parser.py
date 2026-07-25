"""
resume_parser.py

Extracts text from uploaded PDF resumes using PyMuPDF (fitz).
Supports .pdf files only.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text content from an uploaded PDF file.

    Args:
        pdf_file: A file-like object (BytesIO or UploadedFile) containing PDF data.

    Returns:
        str: Extracted plain text from all pages of the PDF.
    """
    text = ""

    try:
        # Read the PDF file bytes
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n\n"

        doc.close()

        if not text.strip():
            return "[No extractable text found in the PDF. It may be a scanned document or image-based PDF.]"

        return text.strip()

    except Exception as e:
        return f"[Error extracting text from PDF: {str(e)}]"


def count_pages(pdf_file) -> int:
    """
    Count the number of pages in an uploaded PDF file.

    Args:
        pdf_file: A file-like object containing PDF data.

    Returns:
        int: Number of pages.
    """
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
        # Reset file pointer for subsequent reads
        pdf_file.seek(0)
        return page_count
    except Exception:
        return 0

