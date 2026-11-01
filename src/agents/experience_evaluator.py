"""
Experience Evaluator Agent

Agent 4 in the pipeline. Evaluates the relevance and quality
of candidate's work experience.

Responsibilities:
- Evaluate experience relevance to job requirements
- Check if minimum experience requirements are met
- Assess career progression
- Identify experience gaps and strengths
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from src.models.output import ExperienceEvalResult
from src.models.state import ResumeScreeningState
from src.llm.groq_client import get_gemini_client, GeminiAPIError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert HR professional with years of experience evaluating candidates.
Your job is to assess work experience relevance and quality for job matching.

Guidelines:
1. Years of experience evaluation:
   - Exact match: Meets or exceeds requirement
   - Close match: Within 1-2 years of requirement
   - Under-qualified: More than 2 years under requirement
   
2. Relevance factors:
   - Industry relevance (same industry highly valuable)
   - Role relevance (similar responsibilities)
   - Technology overlap (using same/similar tech)
   - Scale relevance (company size, team size)
   
3. Career progression indicators:
   - Increasing responsibility over time
   - Promotions and title advancement
   - Expanding scope of work
   - Leadership experience
   
4. Be fair to career changers - valuable transferable experience counts."""


EVALUATION_PROMPT = """Evaluate the candidate's experience against job requirements:

CANDIDATE WORK EXPERIENCE:
{work_experience}

TOTAL YEARS OF EXPERIENCE: {total_years}

JOB REQUIREMENTS:
- Title: {job_title}
- Minimum Experience: {min_experience} years
- Seniority Level: {seniority_level}
- Key Responsibilities: {responsibilities}
- Required Skills Context: {required_skills}

Evaluate:
1. years_experience_score (0-1): Does experience meet/exceed requirements?
2. relevance_score (0-1): How relevant is the experience to this role?
3. progression_score (0-1): Quality of career progression
4. meets_experience_requirement: True/False

Identify:
- relevant_roles: Which roles are most relevant to this job
- experience_gaps: What experience is lacking
- strengths: Key experience strengths

Calculate overall_score as weighted average:
- years_experience_score: 30%
- relevance_score: 50%
- progression_score: 20%

Rate your confidence 0.0-1.0."""


class ExperienceEvaluatorAgent:
    """
    Agent that evaluates candidate work experience relevance and quality.
    
    Assesses whether the candidate's background prepares them for the role.
    """
    
    def __init__(self):
        self.client = get_gemini_client()
        self.name = "ExperienceEvaluatorAgent"
    
    def run(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Execute the experience evaluation agent.
        
        Args:
            state: Current workflow state with parsed_resume and job_requirements
            
        Returns:
            Dict with updates to apply to state
        """
        logger.info(f"[{self.name}] Starting experience evaluation")
        
        parsed_resume = state.get("parsed_resume", {})
        job_requirements = state.get("job_requirements", {})
        
        # Validate inputs
        if not parsed_resume or parsed_resume.get("extraction_confidence", 0) == 0:
            logger.error(f"[{self.name}] No valid parsed resume")
            return self._create_error_response("No valid parsed resume available")
        
        if not job_requirements or job_requirements.get("extraction_confidence", 0) == 0:
            logger.error(f"[{self.name}] No valid job requirements")
            return self._create_error_response("No valid job requirements available")
        
        # Extract relevant data
        work_experience = parsed_resume.get("work_experience", [])
        total_years = parsed_resume.get("total_years_experience", 0)
        
        min_experience = job_requirements.get("min_years_experience", 0)
        job_title = job_requirements.get("job_title", "Unknown")
        seniority = job_requirements.get("seniority_level", "Not specified")
        responsibilities = job_requirements.get("responsibilities", [])
        required_skills = job_requirements.get("required_skills", [])
        
        # Format work experience for prompt
        exp_text = self._format_experience(work_experience)
        
        try:
            # Generate prompt
            prompt = EVALUATION_PROMPT.format(
                work_experience=exp_text if exp_text else "No work experience listed",
                total_years=total_years,
                job_title=job_title,
                min_experience=min_experience,
                seniority_level=seniority,
                responsibilities=", ".join(responsibilities[:5]) if responsibilities else "Not specified",
                required_skills=", ".join(required_skills[:10]) if required_skills else "Not specified"
            )
            
            # Call LLM for evaluation
            eval_result = self.client.generate_structured(
                prompt=prompt,
                output_schema=ExperienceEvalResult,
                temperature=0.2,
                system_instruction=SYSTEM_PROMPT
            )
            
            logger.info(
                f"[{self.name}] Experience evaluation complete: "
                f"overall={eval_result.overall_score:.2f}, "
                f"years={eval_result.years_experience_score:.2f}, "
                f"relevance={eval_result.relevance_score:.2f}, "
                f"meets_req={eval_result.meets_experience_requirement}"
            )
            
            # Build state updates
            updates = {
                "experience_eval": eval_result.model_dump(),
                "confidence_scores": [eval_result.confidence],
                "current_stage": "experience_evaluated",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "evaluate_experience",
                    "status": "success",
                    "overall_score": eval_result.overall_score,
                    "years_score": eval_result.years_experience_score,
                    "relevance_score": eval_result.relevance_score,
                    "meets_requirement": eval_result.meets_experience_requirement,
                    "relevant_years": eval_result.total_relevant_years,
                    "confidence": eval_result.confidence,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            # Add flags for concerning results
            flags = []
            
            if not eval_result.meets_experience_requirement:
                flags.append(
                    f"Does not meet experience requirement "
                    f"({total_years:.1f} years vs {min_experience} required)"
                )
            
            if eval_result.relevance_score < 0.5:
                flags.append(
                    f"Low experience relevance: {eval_result.relevance_score:.0%}"
                )
            
            if eval_result.experience_gaps:
                flags.append(
                    f"Experience gaps: {', '.join(eval_result.experience_gaps[:3])}"
                )
            
            if eval_result.confidence < 0.7:
                flags.append(
                    f"Low evaluation confidence: {eval_result.confidence:.2f}"
                )
            
            if flags:
                updates["flags"] = flags
            
            return updates
            
        except GeminiAPIError as e:
            logger.error(f"[{self.name}] LLM API error: {e}")
            return self._create_error_response(f"LLM API error: {e.message}")
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}")
            return self._create_error_response(str(e))
    
    def _format_experience(self, work_experience: List[Dict]) -> str:
        """Format work experience list for prompt"""
        if not work_experience:
            return ""
        
        formatted = []
        for i, exp in enumerate(work_experience, 1):
            exp_text = f"""
{i}. {exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')}
   Duration: {exp.get('start_date', '?')} - {exp.get('end_date', 'Present')}
   Location: {exp.get('location', 'Not specified')}
   Responsibilities: {', '.join(exp.get('responsibilities', [])[:3])}
   Technologies: {', '.join(exp.get('technologies', [])[:5])}
"""
            formatted.append(exp_text)
        
        return "\n".join(formatted)
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "experience_eval": {
                "overall_score": 0.0,
                "years_experience_score": 0.0,
                "relevance_score": 0.0,
                "progression_score": 0.0,
                "meets_experience_requirement": False,
                "confidence": 0.0,
                "error": error_message
            },
            "flags": [f"Experience evaluation failed: {error_message}"],
            "errors": [f"ExperienceEvaluator error: {error_message}"],
            "requires_human_review": True,
            "current_stage": "experience_eval_failed",
            "agent_logs": [{
                "agent": self.name,
                "action": "evaluate_experience",
                "status": "failed",
                "reason": error_message,
                "timestamp": datetime.now().isoformat()
            }]
        }


# Function interface for LangGraph
def run_experience_evaluator(state: ResumeScreeningState) -> Dict[str, Any]:
    """Function wrapper for LangGraph node"""
    agent = ExperienceEvaluatorAgent()
    return agent.run(state)
