"""
Job Requirements Models

Pydantic models for structured job description data extracted by the Job Analyzer Agent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RequirementType(str, Enum):
    """Type of job requirement"""
    REQUIRED = "required"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


class RequirementCategory(str, Enum):
    """Category of requirement"""
    TECHNICAL_SKILL = "technical_skill"
    SOFT_SKILL = "soft_skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    OTHER = "other"


class Requirement(BaseModel):
    """Individual job requirement"""
    description: str = Field(description="The requirement description")
    requirement_type: RequirementType = Field(
        default=RequirementType.REQUIRED,
        description="Whether required, preferred, or nice-to-have"
    )
    category: RequirementCategory = Field(
        default=RequirementCategory.OTHER,
        description="Category of requirement"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for matching"
    )
    years_experience: Optional[float] = Field(
        default=None,
        description="Years of experience if applicable"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Importance weight for scoring"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "5+ years of Python experience",
                "requirement_type": "required",
                "category": "technical_skill",
                "keywords": ["Python", "programming"],
                "years_experience": 5.0,
                "weight": 1.5
            }
        }


class JobRequirements(BaseModel):
    """
    Complete parsed job requirements.
    
    Output from the Job Analyzer Agent containing structured
    job requirements for matching against candidate profiles.
    """
    
    # Basic Information
    job_title: str = Field(description="Job title/position")
    company: Optional[str] = Field(default=None, description="Company name if mentioned")
    location: Optional[str] = Field(default=None, description="Job location")
    employment_type: Optional[str] = Field(
        default=None,
        description="Full-time, Part-time, Contract, etc."
    )
    remote_option: Optional[str] = Field(
        default=None,
        description="Remote, Hybrid, On-site"
    )
    
    # Experience Requirements
    min_years_experience: float = Field(
        default=0.0,
        description="Minimum years of experience required"
    )
    max_years_experience: Optional[float] = Field(
        default=None,
        description="Maximum years of experience if specified"
    )
    
    # Education Requirements
    required_education: List[str] = Field(
        default_factory=list,
        description="Required educational qualifications"
    )
    preferred_education: List[str] = Field(
        default_factory=list,
        description="Preferred educational qualifications"
    )
    
    # Skills
    required_skills: List[str] = Field(
        default_factory=list,
        description="Must-have technical skills"
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Nice-to-have technical skills"
    )
    
    # Structured Requirements
    requirements: List[Requirement] = Field(
        default_factory=list,
        description="All requirements with metadata"
    )
    
    # Responsibilities
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key job responsibilities"
    )
    
    # Additional Context
    team_size: Optional[str] = Field(
        default=None,
        description="Team size if mentioned"
    )
    reporting_to: Optional[str] = Field(
        default=None,
        description="Who this role reports to"
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Salary range if mentioned"
    )
    
    # Analysis Metadata
    seniority_level: Optional[str] = Field(
        default=None,
        description="Inferred seniority: Entry, Mid, Senior, Lead, Principal"
    )
    role_type: Optional[str] = Field(
        default=None,
        description="Individual Contributor, Manager, etc."
    )
    extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction quality"
    )
    
    def get_required_requirements(self) -> List[Requirement]:
        """Get only required requirements"""
        return [r for r in self.requirements if r.requirement_type == RequirementType.REQUIRED]
    
    def get_preferred_requirements(self) -> List[Requirement]:
        """Get preferred requirements"""
        return [r for r in self.requirements if r.requirement_type == RequirementType.PREFERRED]
    
    def get_all_keywords(self) -> List[str]:
        """Get all keywords from requirements"""
        keywords = set()
        keywords.update(self.required_skills)
        keywords.update(self.preferred_skills)
        for req in self.requirements:
            keywords.update(req.keywords)
        return list(keywords)
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior Backend Engineer",
                "company": "TechCorp",
                "location": "Bangalore, India",
                "min_years_experience": 5.0,
                "required_skills": ["Python", "PostgreSQL", "REST APIs"],
                "preferred_skills": ["Kubernetes", "AWS"],
                "seniority_level": "Senior",
                "extraction_confidence": 0.88
            }
        }
