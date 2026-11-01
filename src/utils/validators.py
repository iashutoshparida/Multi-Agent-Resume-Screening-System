"""
Input Validation Utilities

Functions for validating and loading input files.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from src.parsers.pdf_parser import extract_text_from_pdf
from src.parsers.docx_parser import extract_text_from_docx

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation"""
    valid: bool
    text: str
    confidence: float
    warnings: list
    error: Optional[str] = None


def validate_resume_file(file_path: str) -> ValidationResult:
    """
    Validate and extract text from a resume file.
    
    Supports: PDF, DOCX, TXT
    
    Args:
        file_path: Path to resume file
        
    Returns:
        ValidationResult with extracted text and metadata
    """
    path = Path(file_path)
    
    # Check file exists
    if not path.exists():
        return ValidationResult(
            valid=False,
            text="",
            confidence=0.0,
            warnings=[],
            error=f"File not found: {file_path}"
        )
    
    # Check file size (reject files > 10MB)
    file_size = path.stat().st_size
    if file_size > 10 * 1024 * 1024:
        return ValidationResult(
            valid=False,
            text="",
            confidence=0.0,
            warnings=[],
            error=f"File too large: {file_size / 1024 / 1024:.1f}MB (max 10MB)"
        )
    
    # Check file type and extract
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        result = extract_text_from_pdf(file_path)
        return ValidationResult(
            valid=result.success,
            text=result.text,
            confidence=result.confidence,
            warnings=result.warnings,
            error=result.error
        )
    
    elif suffix == '.docx':
        result = extract_text_from_docx(file_path)
        return ValidationResult(
            valid=result.success,
            text=result.text,
            confidence=result.confidence,
            warnings=result.warnings,
            error=result.error
        )
    
    elif suffix == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return ValidationResult(
                valid=True,
                text=text,
                confidence=1.0,
                warnings=[]
            )
        except UnicodeDecodeError:
            # Try different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return ValidationResult(
                    valid=True,
                    text=text,
                    confidence=0.9,
                    warnings=["File encoding was not UTF-8"]
                )
            except Exception as e:
                return ValidationResult(
                    valid=False,
                    text="",
                    confidence=0.0,
                    warnings=[],
                    error=f"Could not read text file: {e}"
                )
    
    else:
        return ValidationResult(
            valid=False,
            text="",
            confidence=0.0,
            warnings=[],
            error=f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt"
        )


def validate_job_description(input_value: str) -> ValidationResult:
    """
    Validate a job description.
    
    Can be either:
    - A file path to a text file
    - Raw job description text
    
    Args:
        input_value: File path or raw text
        
    Returns:
        ValidationResult with job description text
    """
    # Check if it's a file path
    if os.path.isfile(input_value):
        path = Path(input_value)
        
        if path.suffix.lower() not in ['.txt', '.md', '']:
            return ValidationResult(
                valid=False,
                text="",
                confidence=0.0,
                warnings=[],
                error=f"Job description file must be .txt or plain text, got: {path.suffix}"
            )
        
        try:
            with open(input_value, 'r', encoding='utf-8') as f:
                text = f.read()
            return ValidationResult(
                valid=True,
                text=text,
                confidence=1.0,
                warnings=[]
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                text="",
                confidence=0.0,
                warnings=[],
                error=f"Could not read job description file: {e}"
            )
    
    # Assume it's raw text
    text = input_value.strip()
    
    if not text:
        return ValidationResult(
            valid=False,
            text="",
            confidence=0.0,
            warnings=[],
            error="Job description is empty"
        )
    
    if len(text) < 50:
        return ValidationResult(
            valid=True,
            text=text,
            confidence=0.5,
            warnings=["Job description is very short (< 50 characters)"]
        )
    
    return ValidationResult(
        valid=True,
        text=text,
        confidence=1.0,
        warnings=[]
    )


def load_document(file_path: str) -> Tuple[str, float, Optional[str]]:
    """
    Load any supported document and return its text.
    
    Args:
        file_path: Path to document
        
    Returns:
        Tuple of (text, confidence, error_message)
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        result = validate_resume_file(file_path)
    elif suffix == '.docx':
        result = validate_resume_file(file_path)
    elif suffix in ['.txt', '.md', '']:
        result = validate_job_description(file_path)
    else:
        return ("", 0.0, f"Unsupported file type: {suffix}")
    
    return (result.text, result.confidence, result.error)
