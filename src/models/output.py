"""
Output Models

Pydantic models for the final screening output that matches
the assignment requirements exactly.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class SkillMatch(BaseModel):
    """Individual skill matching result"""
    skill: str = Field(description="The skill being evaluated")
    required: bool = Field(description="Whether this is a required skill")
    found_in_resume: bool = Field(description="Whether skill was found")
    match_type: str = Field(
        description="exact, partial, related, or not_found"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Where in resume this was found"
    )


class SkillsMatchResult(BaseModel):
    """Output from Skills Matcher Agent"""
    overall_score: float = Field(
        ge=0.0, 
        le=1.0,
        description="Overall skills match score"
    )
    required_skills_match_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentage of required skills matched"
    )
    preferred_skills_match_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentage of preferred skills matched"
    )
    matched_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Detailed skill matching results"
    )
    missing_required_skills: List[str] = Field(
        default_factory=list,
        description="Required skills not found"
    )
    missing_preferred_skills: List[str] = Field(
        default_factory=list,
        description="Preferred skills not found"
    )
    additional_skills: List[str] = Field(
        default_factory=list,
        description="Candidate skills not in requirements"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in matching accuracy"
    )


class ExperienceEvalResult(BaseModel):
    """Output from Experience Evaluator Agent"""
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall experience relevance score"
    )
    years_experience_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score based on years of experience"
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score based on experience relevance"
    )
    progression_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Career progression score"
    )
    meets_experience_requirement: bool = Field(
        description="Whether minimum experience is met"
    )
    total_relevant_years: float = Field(
        description="Years of relevant experience"
    )
    relevant_roles: List[str] = Field(
        default_factory=list,
        description="Roles most relevant to the job"
    )
    experience_gaps: List[str] = Field(
        default_factory=list,
        description="Experience areas that are lacking"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Experience strengths"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in evaluation"
    )


class ScreeningOutput(BaseModel):
    """
    Final JSON output matching assignment requirements.
    
    This is the main output schema that the system produces
    for each resume evaluation.
    """
    
    # Required fields from assignment
    match_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Overall match score (0.0 = no fit, 1.0 = perfect fit)"
    )
    recommendation: Literal[
        "Proceed to interview", 
        "Reject", 
        "Needs manual review"
    ] = Field(description="Recommended next step")
    requires_human: bool = Field(
        description="Whether human review is recommended"
    )
    confidence: float = Field(
        ge=0.0, 
        le=1.0,
        description="System confidence in this decision"
    )
    reasoning_summary: str = Field(
        description="Human-readable explanation of the decision"
    )
    
    # Additional context (enhances explainability)
    strengths: List[str] = Field(
        default_factory=list,
        description="Candidate's key strengths"
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="Areas of concern"
    )
    skill_gaps: List[str] = Field(
        default_factory=list,
        description="Missing skills"
    )
    
    # Detailed scores (for transparency)
    skills_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score from skills matching"
    )
    experience_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score from experience evaluation"
    )
    
    # Metadata
    candidate_name: Optional[str] = Field(
        default=None,
        description="Candidate name for reference"
    )
    job_title: Optional[str] = Field(
        default=None,
        description="Job title evaluated against"
    )
    evaluation_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When this evaluation was performed"
    )
    
    # Human review reasons
    human_review_reasons: List[str] = Field(
        default_factory=list,
        description="Why human review is recommended"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "match_score": 0.76,
                "recommendation": "Proceed to interview",
                "requires_human": True,
                "confidence": 0.81,
                "reasoning_summary": "Strong backend skills with Python and Django. Limited exposure to large-scale system design. Recommend interview to assess scaling knowledge.",
                "strengths": [
                    "7 years Python experience",
                    "Strong API development background"
                ],
                "concerns": [
                    "No Kubernetes experience"
                ],
                "skill_gaps": ["Kubernetes", "Large-scale system design"]
            }
        }


def create_error_output(error_message: str, candidate_name: str = None) -> ScreeningOutput:
    """Create an error output when processing fails"""
    return ScreeningOutput(
        match_score=0.0,
        recommendation="Needs manual review",
        requires_human=True,
        confidence=0.0,
        reasoning_summary=f"Processing error: {error_message}",
        candidate_name=candidate_name,
        human_review_reasons=[f"System error: {error_message}"]
    )
