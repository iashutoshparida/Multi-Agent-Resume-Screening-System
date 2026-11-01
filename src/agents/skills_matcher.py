"""
Skills Matcher Agent

Agent 3 in the pipeline. Performs semantic matching between
candidate skills and job requirements.

Responsibilities:
- Match candidate skills to required skills
- Handle skill variations (e.g., "JS" = "JavaScript")
- Identify related/transferable skills
- Calculate skills match score
- Identify skill gaps
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from src.models.output import SkillsMatchResult, SkillMatch
from src.models.state import ResumeScreeningState
from src.llm.groq_client import get_gemini_client, GeminiAPIError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert technical recruiter with deep knowledge of skills and technologies.
Your job is to match candidate skills against job requirements accurately.

Guidelines:
1. Consider skill variations and aliases:
   - "JS" = "JavaScript"
   - "React.js" = "React" = "ReactJS"
   - "Postgres" = "PostgreSQL"
   - "ML" = "Machine Learning"
   
2. Consider related/transferable skills:
   - "PyTorch" experience suggests general ML framework knowledge
   - "AWS" experience suggests cloud infrastructure knowledge
   - "Django" experience suggests Python web development
   
3. Match types:
   - exact: Skill explicitly mentioned with same name/alias
   - partial: Related skill that partially covers the requirement
   - related: Transferable skill from same domain
   - not_found: No matching skill found

4. Be fair but accurate - don't over-credit or under-credit skills."""


MATCHING_PROMPT = """Match the candidate's skills against the job requirements:

CANDIDATE SKILLS:
{candidate_skills}

REQUIRED SKILLS (MUST HAVE):
{required_skills}

PREFERRED SKILLS (NICE TO HAVE):
{preferred_skills}

For each required and preferred skill:
1. Check if the candidate has this skill (exact match)
2. Check for variations/aliases of the skill
3. Check for related/transferable skills
4. Provide match_type: exact, partial, related, or not_found
5. Provide evidence of where the skill was found (if applicable)

Then calculate:
- overall_score: Weighted average (required skills weighted higher)
- required_skills_match_rate: % of required skills matched (exact + partial)
- preferred_skills_match_rate: % of preferred skills matched

List:
- missing_required_skills: Required skills not found at all
- missing_preferred_skills: Preferred skills not found
- additional_skills: Candidate skills not in requirements (may be valuable)

Rate your confidence in the matching accuracy 0.0-1.0."""


class SkillsMatcherAgent:
    """
    Agent that matches candidate skills against job requirements.
    
    Uses semantic understanding to handle skill variations and
    identify transferable skills.
    """
    
    def __init__(self):
        self.client = get_gemini_client()
        self.name = "SkillsMatcherAgent"
    
    def run(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Execute the skills matching agent.
        
        Args:
            state: Current workflow state with parsed_resume and job_requirements
            
        Returns:
            Dict with updates to apply to state
        """
        logger.info(f"[{self.name}] Starting skills matching")
        
        parsed_resume = state.get("parsed_resume", {})
        job_requirements = state.get("job_requirements", {})
        
        # Validate inputs
        if not parsed_resume or parsed_resume.get("extraction_confidence", 0) == 0:
            logger.error(f"[{self.name}] No valid parsed resume")
            return self._create_error_response("No valid parsed resume available")
        
        if not job_requirements or job_requirements.get("extraction_confidence", 0) == 0:
            logger.error(f"[{self.name}] No valid job requirements")
            return self._create_error_response("No valid job requirements available")
        
        # Extract skills lists
        candidate_skills = parsed_resume.get("skills", [])
        technical_skills = parsed_resume.get("technical_skills", [])
        all_candidate_skills = list(set(candidate_skills + technical_skills))
        
        required_skills = job_requirements.get("required_skills", [])
        preferred_skills = job_requirements.get("preferred_skills", [])
        
        if not all_candidate_skills:
            logger.warning(f"[{self.name}] No candidate skills found")
        
        if not required_skills and not preferred_skills:
            logger.warning(f"[{self.name}] No job skills requirements found")
        
        try:
            # Generate prompt
            prompt = MATCHING_PROMPT.format(
                candidate_skills=", ".join(all_candidate_skills) if all_candidate_skills else "None listed",
                required_skills=", ".join(required_skills) if required_skills else "None specified",
                preferred_skills=", ".join(preferred_skills) if preferred_skills else "None specified"
            )
            
            # Call LLM for matching
            match_result = self.client.generate_structured(
                prompt=prompt,
                output_schema=SkillsMatchResult,
                temperature=0.1,
                system_instruction=SYSTEM_PROMPT
            )
            
            logger.info(
                f"[{self.name}] Skills matching complete: "
                f"overall={match_result.overall_score:.2f}, "
                f"required_match={match_result.required_skills_match_rate:.2%}, "
                f"preferred_match={match_result.preferred_skills_match_rate:.2%}"
            )
            
            # Build state updates
            updates = {
                "skills_match": match_result.model_dump(),
                "confidence_scores": [match_result.confidence],
                "current_stage": "skills_matched",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "match_skills",
                    "status": "success",
                    "overall_score": match_result.overall_score,
                    "required_match_rate": match_result.required_skills_match_rate,
                    "preferred_match_rate": match_result.preferred_skills_match_rate,
                    "missing_required": match_result.missing_required_skills,
                    "confidence": match_result.confidence,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            # Add flags for concerning results
            flags = []
            
            if match_result.required_skills_match_rate < 0.5:
                flags.append(
                    f"Low required skills match: {match_result.required_skills_match_rate:.0%}"
                )
            
            if match_result.missing_required_skills:
                flags.append(
                    f"Missing required skills: {', '.join(match_result.missing_required_skills[:5])}"
                )
            
            if match_result.confidence < 0.7:
                flags.append(
                    f"Low skills matching confidence: {match_result.confidence:.2f}"
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
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "skills_match": {
                "overall_score": 0.0,
                "required_skills_match_rate": 0.0,
                "preferred_skills_match_rate": 0.0,
                "confidence": 0.0,
                "error": error_message
            },
            "flags": [f"Skills matching failed: {error_message}"],
            "errors": [f"SkillsMatcher error: {error_message}"],
            "requires_human_review": True,
            "current_stage": "skills_match_failed",
            "agent_logs": [{
                "agent": self.name,
                "action": "match_skills",
                "status": "failed",
                "reason": error_message,
                "timestamp": datetime.now().isoformat()
            }]
        }


# Function interface for LangGraph
def run_skills_matcher(state: ResumeScreeningState) -> Dict[str, Any]:
    """Function wrapper for LangGraph node"""
    agent = SkillsMatcherAgent()
    return agent.run(state)
