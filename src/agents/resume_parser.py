"""
Resume Parser Agent

Agent 1 in the pipeline. Extracts structured information from
resume text using LLM-powered analysis.

Responsibilities:
- Parse resume text into structured format
- Extract skills, experience, education
- Calculate total years of experience
- Provide extraction confidence score
"""

import logging
from typing import Dict, Any
from datetime import datetime

from src.models.resume import ParsedResume
from src.models.state import ResumeScreeningState
from src.llm.groq_client import get_gemini_client, GeminiAPIError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert resume parser with years of experience in HR and recruiting.
Your job is to extract structured information from resume text accurately and completely.

Guidelines:
1. Extract ALL skills mentioned, including programming languages, frameworks, tools, and soft skills
2. Parse work experience with accurate dates and responsibilities
3. Calculate total years of experience by summing up work periods
4. If information is unclear or missing, note it but don't make up data
5. For extraction_confidence:
   - 0.9-1.0: Clear, well-formatted resume with complete information
   - 0.7-0.9: Good resume but some sections unclear
   - 0.5-0.7: Poorly formatted or missing significant information
   - <0.5: Very difficult to parse, major issues

Be thorough but only include information that is explicitly stated or can be directly inferred."""


EXTRACTION_PROMPT = """Parse the following resume and extract all relevant information:

RESUME TEXT:
{resume_text}

Extract the following:
1. Personal Information: name, email, phone, location, LinkedIn, GitHub
2. Professional Summary if present
3. ALL Skills mentioned (technical and soft skills)
4. Work Experience: company, title, dates, responsibilities, achievements, technologies used
5. Education: institution, degree, field, graduation year, GPA if mentioned
6. Certifications
7. Projects if mentioned
8. Calculate total years of professional experience

Be accurate with dates and calculate experience correctly. If currently employed, use today's date ({today}).
Rate your extraction confidence 0.0-1.0 based on resume quality and completeness."""


class ResumeParserAgent:
    """
    Agent that parses resume text into structured data.
    
    This is the first agent in the pipeline and its output
    feeds into all subsequent agents.
    """
    
    def __init__(self):
        self.client = get_gemini_client()
        self.name = "ResumeParserAgent"
    
    def run(self, state: ResumeScreeningState) -> Dict[str, Any]:
        """
        Execute the resume parsing agent.
        
        Args:
            state: Current workflow state containing resume_text
            
        Returns:
            Dict with updates to apply to state
        """
        logger.info(f"[{self.name}] Starting resume parsing")
        
        resume_text = state.get("resume_text", "")
        
        if not resume_text or not resume_text.strip():
            logger.error(f"[{self.name}] No resume text provided")
            return {
                "parsed_resume": {
                    "extraction_confidence": 0.0,
                    "error": "No resume text provided"
                },
                "flags": ["No resume text to parse"],
                "requires_human_review": True,
                "current_stage": "parse_failed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "parse_resume",
                    "status": "failed",
                    "reason": "Empty resume text",
                    "timestamp": datetime.now().isoformat()
                }]
            }
        
        # Check text length
        text_length = len(resume_text)
        if text_length < 100:
            logger.warning(f"[{self.name}] Resume text very short: {text_length} chars")
        
        try:
            # Generate prompt
            prompt = EXTRACTION_PROMPT.format(
                resume_text=resume_text[:15000],  # Limit to avoid token limits
                today=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Call LLM for structured extraction
            parsed = self.client.generate_structured(
                prompt=prompt,
                output_schema=ParsedResume,
                temperature=0.1,
                system_instruction=SYSTEM_PROMPT
            )
            
            logger.info(
                f"[{self.name}] Successfully parsed resume for: {parsed.full_name}"
            )
            logger.info(
                f"[{self.name}] Found {len(parsed.skills)} skills, "
                f"{len(parsed.work_experience)} work experiences, "
                f"{parsed.total_years_experience:.1f} years total experience"
            )
            
            # Build state updates
            updates = {
                "parsed_resume": parsed.model_dump(),
                "confidence_scores": [parsed.extraction_confidence],
                "current_stage": "resume_parsed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "parse_resume",
                    "status": "success",
                    "candidate_name": parsed.full_name,
                    "skills_found": len(parsed.skills),
                    "experience_years": parsed.total_years_experience,
                    "confidence": parsed.extraction_confidence,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            # Add flags for low confidence
            if parsed.extraction_confidence < 0.6:
                updates["flags"] = [
                    f"Low parsing confidence: {parsed.extraction_confidence:.2f}"
                ]
            
            # Add notes if any
            if parsed.extraction_notes:
                updates["agent_logs"][0]["notes"] = parsed.extraction_notes
            
            return updates
            
        except GeminiAPIError as e:
            logger.error(f"[{self.name}] LLM API error: {e}")
            return {
                "parsed_resume": {
                    "extraction_confidence": 0.0,
                    "error": f"LLM API error: {e.message}"
                },
                "flags": ["Resume parsing failed due to API error"],
                "errors": [f"ResumeParser API error: {e.message}"],
                "requires_human_review": True,
                "current_stage": "parse_failed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "parse_resume",
                    "status": "failed",
                    "reason": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            }
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}")
            return {
                "parsed_resume": {
                    "extraction_confidence": 0.0,
                    "error": str(e)
                },
                "flags": ["Resume parsing failed unexpectedly"],
                "errors": [f"ResumeParser error: {str(e)}"],
                "requires_human_review": True,
                "current_stage": "parse_failed",
                "agent_logs": [{
                    "agent": self.name,
                    "action": "parse_resume",
                    "status": "error",
                    "reason": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            }


# Function interface for LangGraph
def run_resume_parser(state: ResumeScreeningState) -> Dict[str, Any]:
    """Function wrapper for LangGraph node"""
    agent = ResumeParserAgent()
    return agent.run(state)
