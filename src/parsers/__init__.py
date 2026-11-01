"""
Document Parsers

Utilities for extracting text from PDF and DOCX files.
"""

from src.parsers.pdf_parser import extract_text_from_pdf, PDFExtractionResult
from src.parsers.docx_parser import extract_text_from_docx, DOCXExtractionResult

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "PDFExtractionResult",
    "DOCXExtractionResult"
]
