"""
LangGraph State Definition

This module defines the shared state that evolves as each agent
contributes its analysis. The state is the "baton" passed between agents.
"""

from typing import TypedDict, Annotated, Optional, List, Any
import operator


class ResumeScreeningState(TypedDict):
    """
    State that evolves through the multi-agent workflow.
    
    Each agent reads from and writes to this shared state,
    enabling structured data passing between components.
    """
    
    # ===== INPUTS =====
    # Raw text extracted from documents
    resume_text: str
    job_description: str
    
    # Original file paths (for reference)
    resume_path: Optional[str]
    job_path: Optional[str]
    
    # ===== AGENT OUTPUTS =====
    # Each agent populates its section
    
    # From Resume Parser Agent
    parsed_resume: Optional[dict]
    
    # From Job Analyzer Agent  
    job_requirements: Optional[dict]
    
    # From Skills Matcher Agent
    skills_match: Optional[dict]
    
    # From Experience Evaluator Agent
    experience_eval: Optional[dict]
    
    # From Decision Maker Agent
    final_output: Optional[dict]
    
    # ===== CONTROL FLOW =====
    # Confidence scores from each agent (aggregated)
    confidence_scores: Annotated[List[float], operator.add]
    
    # Issues/warnings to flag for review
    flags: Annotated[List[str], operator.add]
    
    # Whether human review is required
    requires_human_review: bool
    
    # Current processing stage
    current_stage: str
    
    # ===== AUDIT TRAIL =====
    # Log of agent actions for explainability
    agent_logs: Annotated[List[dict], operator.add]
    
    # Error messages if any
    errors: Annotated[List[str], operator.add]


def create_initial_state(
    resume_text: str,
    job_description: str,
    resume_path: str = None,
    job_path: str = None
) -> ResumeScreeningState:
    """
    Factory function to create a properly initialized state.
    
    Args:
        resume_text: Extracted text from resume
        job_description: Job description text
        resume_path: Optional path to resume file
        job_path: Optional path to job description file
    
    Returns:
        Initialized ResumeScreeningState
    """
    return ResumeScreeningState(
        # Inputs
        resume_text=resume_text,
        job_description=job_description,
        resume_path=resume_path,
        job_path=job_path,
        
        # Agent outputs (start as None)
        parsed_resume=None,
        job_requirements=None,
        skills_match=None,
        experience_eval=None,
        final_output=None,
        
        # Control flow
        confidence_scores=[],
        flags=[],
        requires_human_review=False,
        current_stage="initialized",
        
        # Audit trail
        agent_logs=[],
        errors=[]
    )
