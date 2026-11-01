"""
Decision Maker Agent

Agent 5 (Final) in the pipeline. Synthesizes all prior agent
outputs into a final screening recommendation.

Responsibilities:
- Aggregate scores from skills matching and experience evaluation
- Apply business rules for decision making
- Determine if human review is needed
- Generate clear reasoning summary
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from src.models.output import ScreeningOutput
from src.models.state import ResumeScreeningState
from src.llm.groq_client import get_gemini_client, GeminiAPIError
from src.config import get_config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior hiring manager making final screening decisions.
Your job is to synthesize all evaluation data and provide a clear recommendation.

Decision Guidelines:

1. PROCEED TO INTERVIEW (match_score >= 0.7):
   - Meets most/all required skills
   - Meets experience requirements
   - Good career progression
   - High confidence in evaluation
   
2. REJECT (match_score < 0.4):
   - Missing critical required skills
   - Significantly under-qualified
   - No relevant experience
   - Clear misfit for the role
   
3. NEEDS MANUAL REVIEW (0.4 <= match_score < 0.7 OR uncertainty):
   - Mixed signals (strong in some areas, weak in others)
   - Career changer with transferable skills
   - Close to requirements but not quite there
   - Low confidence in any evaluation
   - Conflicting data

Human Review Triggers:
- Confidence < 0.7 in any component
- Score variance > 0.3 between skills and experience
- Critical missing skills but otherwise strong candidate
- Edge cases (career changers, overqualified, etc.)

Be objective, fair, and explain your reasoning clearly."""


DECISION_PROMPT = """Make a final screening decision based on all evaluations:

CANDIDATE: {candidate_name}
JOB: {job_title}

SKILLS MATCHING RESULT:
- Overall Score: {skills_score}
- Required Skills Match Rate: {required_match_rate}
- Preferred Skills Match Rate: {preferred_match_rate}
- Missing Required Skills: {missing_required}
- Confidence: {skills_confidence}

EXPERIENCE EVALUATION RESULT:
- Overall Score: {experience_score}
- Years Experience Score: {years_score}
- Relevance Score: {relevance_score}
- Meets Experience Requirement: {meets_requirement}
- Total Relevant Years: {relevant_years}
- Strengths: {experience_strengths}
- Gaps: {experience_gaps}
- Confidence: {experience_confidence}

FLAGS FROM PROCESSING:
{flags}

Make a decision:
1. Calculate match_score (weighted average of skills and experience)
2. Choose recommendation: "Proceed to interview", "Reject", or "Needs manual review"
3. Determine if human review is needed and why
4. Rate your confidence in this decision
5. Write a clear reasoning_summary explaining the decision
6. List key strengths, concerns, and skill_gaps"""


class DecisionMakerAgent:
    """
    Final agent that synthesizes all evaluations into a recommendation.
    
    This is the output agent - its result is the final screening decision.
    """
    
    def __init__(self):
        self.client = get_gemini_client()
        self.name = "DecisionMakerAgent"
        self.config = get_config()
    
    def run(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Execute the decision maker agent.
        
        Args:
            state: Current workflow state with all prior agent outputs
            
        Returns:
            Dict with updates including final_output
        """
        logger.info(f"[{self.name}] Making final screening decision")
        
        # Gather all prior evaluations
        parsed_resume = state.get("parsed_resume", {})
        job_requirements = state.get("job_requirements", {})
        skills_match = state.get("skills_match", {})
        experience_eval = state.get("experience_eval", {})
        flags = state.get("flags", [])
        
        # Check for missing data
        if not skills_match or skills_match.get("confidence", 0) == 0:
            logger.warning(f"[{self.name}] No valid skills match data")
        
        if not experience_eval or experience_eval.get("confidence", 0) == 0:
            logger.warning(f"[{self.name}] No valid experience evaluation data")
        
        # Extract key metrics
        candidate_name = parsed_resume.get("full_name", "Unknown Candidate")
        job_title = job_requirements.get("job_title", "Unknown Position")
        
        skills_score = skills_match.get("overall_score", 0)
        experience_score = experience_eval.get("overall_score", 0)
        skills_confidence = skills_match.get("confidence", 0)
        experience_confidence = experience_eval.get("confidence", 0)
        
        # Check for conflicting scores
        score_variance = abs(skills_score - experience_score)
        if score_variance > 0.3:
            flags.append(
                f"Score variance ({score_variance:.2f}): "
                f"skills={skills_score:.2f} vs experience={experience_score:.2f}"
            )
        
        try:
            # Generate prompt
            prompt = DECISION_PROMPT.format(
                candidate_name=candidate_name,
                job_title=job_title,
                skills_score=f"{skills_score:.2f}",
                required_match_rate=f"{skills_match.get('required_skills_match_rate', 0):.0%}",
                preferred_match_rate=f"{skills_match.get('preferred_skills_match_rate', 0):.0%}",
                missing_required=", ".join(skills_match.get("missing_required_skills", [])) or "None",
                skills_confidence=f"{skills_confidence:.2f}",
                experience_score=f"{experience_score:.2f}",
                years_score=f"{experience_eval.get('years_experience_score', 0):.2f}",
                relevance_score=f"{experience_eval.get('relevance_score', 0):.2f}",
                meets_requirement=experience_eval.get("meets_experience_requirement", False),
                relevant_years=f"{experience_eval.get('total_relevant_years', 0):.1f}",
                experience_strengths=", ".join(experience_eval.get("strengths", [])) or "None identified",
                experience_gaps=", ".join(experience_eval.get("experience_gaps", [])) or "None",
                experience_confidence=f"{experience_confidence:.2f}",
                flags="\n".join(f"- {f}" for f in flags) if flags else "None"
            )
            
            # Call LLM for final decision
            decision = self.client.generate_structured(
                prompt=prompt,
                output_schema=ScreeningOutput,
                temperature=0.2,
                system_instruction=SYSTEM_PROMPT
            )
            
            # Apply business rules to override if needed
            decision = self._apply_business_rules(
                decision, skills_match, experience_eval, flags
            )
            
            # Add metadata
            decision.candidate_name = candidate_name
            decision.job_title = job_title
            decision.skills_score = skills_score
            decision.experience_score = experience_score
            
            logger.info(
                f"[{self.name}] Decision: {decision.recommendation} "
                f"(score={decision.match_score:.2f}, confidence={decision.confidence:.2f})"
            )
            
            # Build state updates
            updates = {
                "final_output": decision.model_dump(),
                "requires_human_review": decision.requires_human,
                "current_stage": "decision_complete",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "make_decision",
                    "status": "success",
                    "recommendation": decision.recommendation,
                    "match_score": decision.match_score,
                    "confidence": decision.confidence,
                    "requires_human": decision.requires_human,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            return updates
            
        except GeminiAPIError as e:
            logger.error(f"[{self.name}] LLM API error: {e}")
            return self._create_error_response(
                f"LLM API error: {e.message}",
                candidate_name,
                job_title
            )
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}")
            return self._create_error_response(
                str(e),
                candidate_name,
                job_title
            )
    
    def _apply_business_rules(
        self,
        decision: ScreeningOutput,
        skills_match: Dict,
        experience_eval: Dict,
        flags: List[str]
    ) -> ScreeningOutput:
        """Apply business rules to ensure consistency"""
        
        human_review_reasons = []
        
        # Rule 1: Low confidence triggers human review
        if decision.confidence < self.config.confidence_threshold:
            decision.requires_human = True
            human_review_reasons.append(
                f"Low decision confidence: {decision.confidence:.2f}"
            )
        
        # Rule 2: Score variance triggers human review
        skills_score = skills_match.get("overall_score", 0)
        exp_score = experience_eval.get("overall_score", 0)
        if abs(skills_score - exp_score) > 0.3:
            decision.requires_human = True
            human_review_reasons.append(
                f"Conflicting scores: skills={skills_score:.2f}, experience={exp_score:.2f}"
            )
        
        # Rule 3: Many flags trigger human review
        if len(flags) >= 3:
            decision.requires_human = True
            human_review_reasons.append(f"Multiple flags raised: {len(flags)}")
        
        # Rule 4: Edge case - high experience but low skills
        if exp_score > 0.7 and skills_score < 0.5:
            decision.requires_human = True
            human_review_reasons.append(
                "Strong experience but skills gap - potential for training"
            )
        
        # Rule 5: "Needs manual review" always sets requires_human
        if decision.recommendation == "Needs manual review":
            decision.requires_human = True
        
        # Update human review reasons
        if human_review_reasons:
            decision.human_review_reasons = human_review_reasons
        
        return decision
    
    def _create_error_response(
        self,
        error_message: str,
        candidate_name: str,
        job_title: str
    ) -> Dict[str, Any]:
        """Create error response"""
        from src.models.output import create_error_output
        
        error_output = create_error_output(error_message, candidate_name)
        error_output.job_title = job_title
        
        return {
            "final_output": error_output.model_dump(),
            "requires_human_review": True,
            "errors": [f"DecisionMaker error: {error_message}"],
            "current_stage": "decision_failed",
            "agent_logs": [{
                "agent": self.name,
                "action": "make_decision",
                "status": "failed",
                "reason": error_message,
                "timestamp": datetime.now().isoformat()
            }]
        }


# Function interface for LangGraph
def run_decision_maker(state: ResumeScreeningState) -> Dict[str, Any]:
    """Function wrapper for LangGraph node"""
    agent = DecisionMakerAgent()
    return agent.run(state)
