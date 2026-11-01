"""
Tests for document parsers.
"""

import pytest
from pathlib import Path
import tempfile
import os

# Import parsers
from src.parsers.pdf_parser import extract_text_from_pdf
from src.parsers.docx_parser import extract_text_from_docx
from src.utils.validators import validate_resume_file, validate_job_description


class TestPDFParser:
    """Tests for PDF parsing functionality"""
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file"""
        result = extract_text_from_pdf("/nonexistent/file.pdf")
        assert not result.success
        assert "not found" in result.error.lower()
    
    def test_wrong_extension(self):
        """Test handling of wrong file extension"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not a PDF")
            temp_path = f.name
        
        try:
            result = extract_text_from_pdf(temp_path)
            assert not result.success
            assert "not a pdf" in result.error.lower()
        finally:
            os.unlink(temp_path)


class TestDOCXParser:
    """Tests for DOCX parsing functionality"""
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file"""
        result = extract_text_from_docx("/nonexistent/file.docx")
        assert not result.success
        assert "not found" in result.error.lower()
    
    def test_wrong_extension(self):
        """Test handling of wrong file extension"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not a DOCX")
            temp_path = f.name
        
        try:
            result = extract_text_from_docx(temp_path)
            assert not result.success
            assert "not a docx" in result.error.lower()
        finally:
            os.unlink(temp_path)


class TestValidators:
    """Tests for input validators"""
    
    def test_validate_empty_job_description(self):
        """Test validation of empty job description"""
        result = validate_job_description("")
        assert not result.valid
        assert "empty" in result.error.lower()
    
    def test_validate_short_job_description(self):
        """Test validation of very short job description"""
        result = validate_job_description("Python dev")
        assert result.valid  # Still valid but with warning
        assert len(result.warnings) > 0
    
    def test_validate_normal_job_description(self):
        """Test validation of normal job description text"""
        job_desc = """
        Senior Python Developer
        
        We are looking for an experienced Python developer with:
        - 5+ years of Python experience
        - Knowledge of Django or FastAPI
        - Experience with PostgreSQL
        """
        result = validate_job_description(job_desc)
        assert result.valid
        assert result.confidence == 1.0
    
    def test_validate_nonexistent_resume(self):
        """Test validation of non-existent resume file"""
        result = validate_resume_file("/nonexistent/resume.pdf")
        assert not result.valid
        assert "not found" in result.error.lower()


class TestTextFileHandling:
    """Tests for plain text file handling"""
    
    def test_load_text_resume(self):
        """Test loading a plain text resume"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
            f.write("""
John Doe
Software Engineer

Skills: Python, JavaScript, SQL
Experience: 5 years at TechCorp
            """)
            temp_path = f.name
        
        try:
            result = validate_resume_file(temp_path)
            assert result.valid
            assert "John Doe" in result.text
            assert result.confidence == 1.0
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
