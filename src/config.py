"""
Configuration management for the Resume Screening Agent.
Loads settings from environment variables with sensible defaults.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

# Load environment variables from .env file
load_dotenv()


class Config(BaseModel):
    """Application configuration"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # API Configuration - Support both GROQ and GEMINI keys for backward compatibility
    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    )
    
    # Model Configuration - Support both GROQ and GEMINI model env vars
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL") or os.getenv("GEMINI_MODEL", "llama-3.1-8b-instant")
    )
    
    # Processing Configuration
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    
    # Logging
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    def validate_api_key(self) -> bool:
        """Check if API key is configured"""
        key = self.groq_api_key
        invalid_placeholders = [
            "",
            "your_api_key_here",
            "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ]
        return bool(key and key not in invalid_placeholders)
    
    # Aliases for backward compatibility with code expecting gemini_* names
    @property
    def gemini_api_key(self) -> str:
        return self.groq_api_key
    
    @property
    def gemini_model(self) -> str:
        return self.groq_model


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def setup_logging():
    """Configure logging based on config"""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)
