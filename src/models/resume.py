"""
Resume Data Models

Pydantic models for structured resume data extracted by the Resume Parser Agent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class Education(BaseModel):
    """Educational qualification"""
    institution: str = Field(description="Name of university/college/school")
    degree: str = Field(description="Degree obtained (e.g., B.Tech, MBA, PhD)")
    field_of_study: Optional[str] = Field(
        default=None, 
        description="Major/specialization"
    )
    graduation_year: Optional[int] = Field(
        default=None,
        description="Year of graduation"
    )
    gpa: Optional[float] = Field(
        default=None,
        description="GPA if mentioned"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "institution": "Indian Institute of Technology Delhi",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "graduation_year": 2020,
                "gpa": 8.5
            }
        }


class WorkExperience(BaseModel):
    """Work experience entry"""
    company: str = Field(description="Company/organization name")
    title: str = Field(description="Job title/role")
    start_date: str = Field(description="Start date (YYYY-MM or YYYY)")
    end_date: Optional[str] = Field(
        default=None, 
        description="End date (YYYY-MM, YYYY, or 'Present')"
    )
    location: Optional[str] = Field(
        default=None,
        description="Job location"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities"
    )
    achievements: List[str] = Field(
        default_factory=list,
        description="Notable achievements with metrics if available"
    )
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies/tools used in this role"
    )
    
    @property
    def is_current(self) -> bool:
        """Check if this is a current position"""
        return self.end_date is None or self.end_date.lower() == "present"
    
    class Config:
        json_schema_extra = {
            "example": {
                "company": "Google",
                "title": "Senior Software Engineer",
                "start_date": "2021-06",
                "end_date": "Present",
                "location": "Bangalore, India",
                "responsibilities": [
                    "Led development of ML pipeline",
                    "Mentored team of 5 engineers"
                ],
                "achievements": [
                    "Reduced inference latency by 40%",
                    "Shipped feature used by 10M+ users"
                ],
                "technologies": ["Python", "TensorFlow", "Kubernetes"]
            }
        }


class Project(BaseModel):
    """Personal or academic project"""
    name: str = Field(description="Project name")
    description: str = Field(description="Brief description")
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies used"
    )
    url: Optional[str] = Field(
        default=None,
        description="Project URL (GitHub, demo, etc.)"
    )


class ParsedResume(BaseModel):
    """
    Complete parsed resume data.
    
    Output from the Resume Parser Agent containing all structured
    information extracted from the resume document.
    """
    
    # Personal Information
    full_name: str = Field(description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="Current location")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(default=None, description="GitHub profile URL")
    portfolio_url: Optional[str] = Field(default=None, description="Portfolio website")
    
    # Professional Summary
    summary: Optional[str] = Field(
        default=None,
        description="Professional summary/objective if present"
    )
    
    # Skills
    skills: List[str] = Field(
        default_factory=list,
        description="List of all skills mentioned"
    )
    technical_skills: List[str] = Field(
        default_factory=list,
        description="Technical/hard skills"
    )
    soft_skills: List[str] = Field(
        default_factory=list,
        description="Soft skills"
    )
    
    # Experience
    work_experience: List[WorkExperience] = Field(
        default_factory=list,
        description="Work experience entries, most recent first"
    )
    total_years_experience: float = Field(
        default=0.0,
        description="Total years of professional experience"
    )
    
    # Education
    education: List[Education] = Field(
        default_factory=list,
        description="Educational qualifications"
    )
    
    # Additional
    certifications: List[str] = Field(
        default_factory=list,
        description="Professional certifications"
    )
    projects: List[Project] = Field(
        default_factory=list,
        description="Notable projects"
    )
    languages: List[str] = Field(
        default_factory=list,
        description="Languages spoken"
    )
    
    # Metadata
    extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction quality (0.0-1.0)"
    )
    extraction_notes: List[str] = Field(
        default_factory=list,
        description="Notes about extraction issues or assumptions"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "Rahul Sharma",
                "email": "rahul.sharma@email.com",
                "phone": "+91-9876543210",
                "location": "Bangalore, India",
                "skills": ["Python", "Machine Learning", "AWS", "Docker"],
                "technical_skills": ["Python", "TensorFlow", "SQL"],
                "work_experience": [],
                "total_years_experience": 5.5,
                "education": [],
                "extraction_confidence": 0.92
            }
        }
