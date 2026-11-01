"""
Job Analyzer Agent

Agent 2 in the pipeline. Parses job description to extract
structured requirements for matching.

Responsibilities:
- Extract required and preferred skills
- Identify experience requirements
- Parse education requirements
- Categorize and weight requirements
"""

import logging
from typing import Dict, Any
from datetime import datetime

from src.models.job import JobRequirements, Requirement, RequirementType, RequirementCategory
from src.models.state import ResumeScreeningState
from src.llm.groq_client import get_gemini_client, GeminiAPIError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert job description analyzer with deep knowledge of hiring requirements.
Your job is to parse job descriptions and extract structured requirements that can be used for candidate matching.

Guidelines:
1. Distinguish clearly between REQUIRED and PREFERRED qualifications
2. Extract specific skills with their context (e.g., "Python" vs "5+ years Python")
3. Identify both technical skills and soft skills
4. Parse experience requirements carefully (minimum years, domain experience)
5. Weight requirements appropriately:
   - Core technical skills: 1.5
   - Required experience: 1.3
   - Required soft skills: 1.0
   - Preferred skills: 0.8
   - Nice-to-have: 0.5

Be precise and don't infer requirements that aren't stated."""


ANALYSIS_PROMPT = """Analyze the following job description and extract all requirements:

JOB DESCRIPTION:
{job_description}

Extract:
1. Job title and company (if mentioned)
2. Location and work arrangement (remote/hybrid/onsite)
3. Required skills - MUST-HAVE qualifications
4. Preferred skills - NICE-TO-HAVE qualifications
5. Minimum years of experience required
6. Education requirements
7. Key responsibilities
8. Seniority level (Entry/Mid/Senior/Lead/Principal) based on requirements

For each requirement, provide:
- Clear description
- Whether it's required, preferred, or nice-to-have
- Category (technical_skill, soft_skill, experience, education, certification, domain_knowledge)
- Keywords for matching
- Weight (1.0-2.0 for required, 0.5-1.0 for preferred)

Rate your extraction confidence 0.0-1.0."""


class JobAnalyzerAgent:
    """
    Agent that analyzes job descriptions to extract structured requirements.
    
    Runs in parallel with or after the Resume Parser, providing
    requirements that the Skills Matcher will use.
    """
    
    def __init__(self):
        self.client = get_gemini_client()
        self.name = "JobAnalyzerAgent"
    
    def run(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Execute the job analysis agent.
        
        Args:
            state: Current workflow state containing job_description
            
        Returns:
            Dict with updates to apply to state
        """
        logger.info(f"[{self.name}] Starting job description analysis")
        
        job_description = state.get("job_description", "")
        
        if not job_description or not job_description.strip():
            logger.error(f"[{self.name}] No job description provided")
            return {
                "job_requirements": {
                    "extraction_confidence": 0.0,
                    "error": "No job description provided"
                },
                "flags": ["No job description to analyze"],
                "requires_human_review": True,
                "current_stage": "job_analysis_failed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "analyze_job",
                    "status": "failed",
                    "reason": "Empty job description",
                    "timestamp": datetime.now().isoformat()
                }]
            }
        
        try:
            # Generate prompt
            prompt = ANALYSIS_PROMPT.format(
                job_description=job_description[:10000]  # Limit length
            )
            
            # Call LLM for structured extraction
            requirements = self.client.generate_structured(
                prompt=prompt,
                output_schema=JobRequirements,
                temperature=0.1,
                system_instruction=SYSTEM_PROMPT
            )
            
            logger.info(
                f"[{self.name}] Analyzed job: {requirements.job_title}"
            )
            logger.info(
                f"[{self.name}] Found {len(requirements.required_skills)} required skills, "
                f"{len(requirements.preferred_skills)} preferred skills, "
                f"min experience: {requirements.min_years_experience} years"
            )
            
            # Build state updates
            updates = {
                "job_requirements": requirements.model_dump(),
                "confidence_scores": [requirements.extraction_confidence],
                "current_stage": "job_analyzed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "analyze_job",
                    "status": "success",
                    "job_title": requirements.job_title,
                    "required_skills_count": len(requirements.required_skills),
                    "preferred_skills_count": len(requirements.preferred_skills),
                    "min_experience": requirements.min_years_experience,
                    "seniority": requirements.seniority_level,
                    "confidence": requirements.extraction_confidence,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            # Add flags for potential issues
            flags = []
            
            if requirements.extraction_confidence < 0.7:
                flags.append(f"Low job analysis confidence: {requirements.extraction_confidence:.2f}")
            
            if not requirements.required_skills:
                flags.append("No required skills identified in job description")
            
            if requirements.min_years_experience == 0 and requirements.seniority_level in ["Senior", "Lead", "Principal"]:
                flags.append("Senior role but no minimum experience specified")
            
            if flags:
                updates["flags"] = flags
            
            return updates
            
        except GeminiAPIError as e:
            logger.error(f"[{self.name}] LLM API error: {e}")
            return {
                "job_requirements": {
                    "extraction_confidence": 0.0,
                    "error": f"LLM API error: {e.message}"
                },
                "flags": ["Job analysis failed due to API error"],
                "errors": [f"JobAnalyzer API error: {e.message}"],
                "requires_human_review": True,
                "current_stage": "job_analysis_failed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "analyze_job",
                    "status": "failed",
                    "reason": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            }
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}")
            return {
                "job_requirements": {
                    "extraction_confidence": 0.0,
                    "error": str(e)
                },
                "flags": ["Job analysis failed unexpectedly"],
                "errors": [f"JobAnalyzer error: {str(e)}"],
                "requires_human_review": True,
                "current_stage": "job_analysis_failed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "analyze_job",
                    "status": "error",
                    "reason": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            }


# Function interface for LangGraph
def run_job_analyzer(state: ResumeScreeningState) -> Dict[str, Any]:
    """Function wrapper for LangGraph node"""
    agent = JobAnalyzerAgent()
    return agent.run(state)
