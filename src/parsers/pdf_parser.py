"""
PDF Parser

Extracts text from PDF files with multiple fallback strategies.
Uses PyMuPDF as primary parser with pypdf as fallback.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction"""
    text: str
    confidence: float
    method: str
    page_count: int
    warnings: List[str]
    success: bool
    error: Optional[str] = None


def extract_text_pymupdf(file_path: str) -> Tuple[str, float, List[str], int]:
    """
    Primary PDF extraction using PyMuPDF (fitz).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (text, confidence, warnings, page_count)
    """
    try:
        import pymupdf
    except ImportError:
        raise ImportError("pymupdf not installed. Run: pip install pymupdf")
    
    warnings = []
    text_parts = []
    
    try:
        doc = pymupdf.open(file_path)
        page_count = len(doc)
        
        if doc.is_repaired:
            warnings.append("PDF required repair during opening")
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            
            # Check for scanned pages (images but no text)
            if not page_text.strip():
                images = page.get_images()
                if images:
                    warnings.append(f"Page {page_num + 1}: May be scanned (has images but no text)")
                    # OCR would go here if enabled
            
            text_parts.append(page_text)
        
        doc.close()
        full_text = "\n\n".join(text_parts).strip()
        
        # Calculate confidence based on extraction quality
        if not full_text:
            confidence = 0.0
        elif warnings:
            confidence = 0.75
        else:
            confidence = 0.95
        
        return full_text, confidence, warnings, page_count
        
    except Exception as e:
        raise RuntimeError(f"PyMuPDF extraction failed: {e}")


def extract_text_pypdf(file_path: str) -> Tuple[str, float, List[str], int]:
    """
    Fallback PDF extraction using pypdf.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (text, confidence, warnings, page_count)
    """
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")
    
    warnings = []
    
    try:
        reader = PdfReader(file_path)
        page_count = len(reader.pages)
        
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            except Exception as e:
                warnings.append(f"Page {i + 1}: Extraction failed - {e}")
                text_parts.append("")
        
        full_text = "\n\n".join(text_parts).strip()
        
        if not full_text:
            confidence = 0.0
        elif warnings:
            confidence = 0.6
        else:
            confidence = 0.7  # pypdf generally lower quality than PyMuPDF
        
        return full_text, confidence, warnings, page_count
        
    except Exception as e:
        raise RuntimeError(f"pypdf extraction failed: {e}")


def extract_text_pdfplumber(file_path: str) -> Tuple[str, float, List[str], int]:
    """
    Alternative extraction using pdfplumber (good for tables).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (text, confidence, warnings, page_count)
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
    
    warnings = []
    text_parts = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    
                    # Also extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = "\n".join(
                                " | ".join(str(cell) if cell else "" for cell in row)
                                for row in table if row
                            )
                            page_text += "\n" + table_text
                    
                    text_parts.append(page_text)
                except Exception as e:
                    warnings.append(f"Page {i + 1}: Extraction issue - {e}")
                    text_parts.append("")
        
        full_text = "\n\n".join(text_parts).strip()
        
        if not full_text:
            confidence = 0.0
        elif warnings:
            confidence = 0.65
        else:
            confidence = 0.8
        
        return full_text, confidence, warnings, page_count
        
    except Exception as e:
        raise RuntimeError(f"pdfplumber extraction failed: {e}")


def extract_text_from_pdf(file_path: str) -> PDFExtractionResult:
    """
    Extract text from PDF with automatic fallback chain.
    
    Tries extractors in order:
    1. PyMuPDF (fastest, best quality)
    2. pdfplumber (good for tables)
    3. pypdf (pure Python fallback)
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        PDFExtractionResult with extracted text and metadata
    """
    path = Path(file_path)
    
    if not path.exists():
        return PDFExtractionResult(
            text="",
            confidence=0.0,
            method="none",
            page_count=0,
            warnings=[],
            success=False,
            error=f"File not found: {file_path}"
        )
    
    if path.suffix.lower() != '.pdf':
        return PDFExtractionResult(
            text="",
            confidence=0.0,
            method="none",
            page_count=0,
            warnings=[],
            success=False,
            error=f"Not a PDF file: {file_path}"
        )
    
    all_warnings = []
    
    # Try PyMuPDF first
    try:
        text, confidence, warnings, page_count = extract_text_pymupdf(file_path)
        if text.strip():
            return PDFExtractionResult(
                text=text,
                confidence=confidence,
                method="pymupdf",
                page_count=page_count,
                warnings=warnings,
                success=True
            )
        all_warnings.extend([f"[pymupdf] {w}" for w in warnings])
        all_warnings.append("[pymupdf] No text extracted")
    except ImportError:
        all_warnings.append("[pymupdf] Not installed")
    except Exception as e:
        all_warnings.append(f"[pymupdf] Failed: {e}")
        logger.warning(f"PyMuPDF failed: {e}")
    
    # Try pdfplumber
    try:
        text, confidence, warnings, page_count = extract_text_pdfplumber(file_path)
        if text.strip():
            return PDFExtractionResult(
                text=text,
                confidence=confidence,
                method="pdfplumber",
                page_count=page_count,
                warnings=all_warnings + warnings,
                success=True
            )
        all_warnings.extend([f"[pdfplumber] {w}" for w in warnings])
        all_warnings.append("[pdfplumber] No text extracted")
    except ImportError:
        all_warnings.append("[pdfplumber] Not installed")
    except Exception as e:
        all_warnings.append(f"[pdfplumber] Failed: {e}")
        logger.warning(f"pdfplumber failed: {e}")
    
    # Try pypdf as last resort
    try:
        text, confidence, warnings, page_count = extract_text_pypdf(file_path)
        if text.strip():
            return PDFExtractionResult(
                text=text,
                confidence=confidence,
                method="pypdf",
                page_count=page_count,
                warnings=all_warnings + warnings,
                success=True
            )
        all_warnings.extend([f"[pypdf] {w}" for w in warnings])
        all_warnings.append("[pypdf] No text extracted")
    except ImportError:
        all_warnings.append("[pypdf] Not installed")
    except Exception as e:
        all_warnings.append(f"[pypdf] Failed: {e}")
        logger.warning(f"pypdf failed: {e}")
    
    # All extractors failed
    return PDFExtractionResult(
        text="",
        confidence=0.0,
        method="none",
        page_count=0,
        warnings=all_warnings,
        success=False,
        error="All PDF extraction methods failed"
    )
