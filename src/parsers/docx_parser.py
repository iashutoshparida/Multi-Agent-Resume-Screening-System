"""
DOCX Parser

Extracts text from Word documents including paragraphs and tables.
"""

import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DOCXExtractionResult:
    """Result of DOCX text extraction"""
    text: str
    confidence: float
    paragraph_count: int
    table_count: int
    warnings: List[str]
    success: bool
    error: Optional[str] = None


def extract_text_from_docx(file_path: str) -> DOCXExtractionResult:
    """
    Extract text from DOCX file including tables.
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        DOCXExtractionResult with extracted text and metadata
    """
    path = Path(file_path)
    
    if not path.exists():
        return DOCXExtractionResult(
            text="",
            confidence=0.0,
            paragraph_count=0,
            table_count=0,
            warnings=[],
            success=False,
            error=f"File not found: {file_path}"
        )
    
    if path.suffix.lower() != '.docx':
        return DOCXExtractionResult(
            text="",
            confidence=0.0,
            paragraph_count=0,
            table_count=0,
            warnings=[],
            success=False,
            error=f"Not a DOCX file: {file_path}"
        )
    
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError:
        return DOCXExtractionResult(
            text="",
            confidence=0.0,
            paragraph_count=0,
            table_count=0,
            warnings=[],
            success=False,
            error="python-docx not installed. Run: pip install python-docx"
        )
    
    warnings = []
    text_parts = []
    paragraph_count = 0
    table_count = 0
    
    try:
        doc = Document(file_path)
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())
                paragraph_count += 1
        
        # Extract tables
        for table in doc.tables:
            table_count += 1
            table_rows = []
            for row in table.rows:
                row_cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_cells.append(cell_text)
                if row_cells:
                    table_rows.append(" | ".join(row_cells))
            if table_rows:
                text_parts.append("\n".join(table_rows))
        
        # Try to extract from headers and footers
        try:
            for section in doc.sections:
                header = section.header
                footer = section.footer
                
                for para in header.paragraphs:
                    if para.text.strip():
                        text_parts.insert(0, para.text.strip())
                
                for para in footer.paragraphs:
                    if para.text.strip():
                        text_parts.append(para.text.strip())
        except Exception as e:
            warnings.append(f"Could not extract headers/footers: {e}")
        
        full_text = "\n\n".join(text_parts).strip()
        
        # Calculate confidence
        if not full_text:
            confidence = 0.0
        elif warnings:
            confidence = 0.85
        else:
            confidence = 0.95
        
        return DOCXExtractionResult(
            text=full_text,
            confidence=confidence,
            paragraph_count=paragraph_count,
            table_count=table_count,
            warnings=warnings,
            success=True
        )
        
    except PackageNotFoundError:
        return DOCXExtractionResult(
            text="",
            confidence=0.0,
            paragraph_count=0,
            table_count=0,
            warnings=[],
            success=False,
            error="Invalid DOCX file: Package not found"
        )
    except Exception as e:
        import zipfile
        if isinstance(e, zipfile.BadZipFile):
            error_msg = "Corrupted DOCX file: Invalid ZIP format"
        else:
            error_msg = f"DOCX extraction failed: {e}"
        
        logger.error(error_msg)
        return DOCXExtractionResult(
            text="",
            confidence=0.0,
            paragraph_count=0,
            table_count=0,
            warnings=[],
            success=False,
            error=error_msg
        )


def extract_text_auto(file_path: str) -> tuple:
    """
    Automatically detect file type and extract text.
    
    Args:
        file_path: Path to resume file (PDF or DOCX)
        
    Returns:
        Tuple of (text, confidence, warnings, success, error)
    """
    from src.parsers.pdf_parser import extract_text_from_pdf
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        result = extract_text_from_pdf(file_path)
        return (
            result.text,
            result.confidence,
            result.warnings,
            result.success,
            result.error
        )
    elif suffix == '.docx':
        result = extract_text_from_docx(file_path)
        return (
            result.text,
            result.confidence,
            result.warnings,
            result.success,
            result.error
        )
    elif suffix == '.txt':
        # Plain text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return (text, 1.0, [], True, None)
        except Exception as e:
            return ("", 0.0, [], False, f"Could not read text file: {e}")
    else:
        return (
            "",
            0.0,
            [],
            False,
            f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt"
        )
