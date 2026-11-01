"""
LangGraph Workflow Definition

Defines the multi-agent workflow as a graph with:
- Nodes for each agent
- Conditional edges for routing based on confidence
- Human-in-the-loop checkpoints
"""

import logging
from typing import Literal, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.state import ResumeScreeningState, create_initial_state
from src.agents.resume_parser import run_resume_parser
from src.agents.job_analyzer import run_job_analyzer
from src.agents.skills_matcher import run_skills_matcher
from src.agents.experience_evaluator import run_experience_evaluator
from src.agents.decision_maker import run_decision_maker
from src.config import get_config

logger = logging.getLogger(__name__)


# ============= ROUTING FUNCTIONS =============

def route_after_parsing(state: ResumeScreeningState) -> Literal["analyze_job", "early_exit"]:
    """
    Route after resume parsing based on extraction confidence.
    
    Decision Point 1: Did we successfully parse the resume?
    """
    parsed = state.get("parsed_resume", {})
    confidence = parsed.get("extraction_confidence", 0)
    
    if confidence < 0.3:
        logger.warning(f"Resume parsing failed (confidence={confidence:.2f}), routing to early exit")
        return "early_exit"
    
    return "analyze_job"


def route_after_job_analysis(state: ResumeScreeningState) -> Literal["match_skills", "early_exit"]:
    """
    Route after job analysis based on extraction quality.
    
    Decision Point 2: Did we successfully parse the job description?
    """
    job_req = state.get("job_requirements", {})
    confidence = job_req.get("extraction_confidence", 0)
    
    if confidence < 0.3:
        logger.warning(f"Job analysis failed (confidence={confidence:.2f}), routing to early exit")
        return "early_exit"
    
    return "match_skills"


def route_to_decision(state: ResumeScreeningState) -> Literal["make_decision"]:
    """
    Route to decision maker after evaluations.
    Always proceeds to decision - this is a synchronization point.
    """
    return "make_decision"


# ============= NODE FUNCTIONS =============

def early_exit_node(state: ResumeScreeningState) -> Dict[str, Any]:
    """
    Handle early exit when parsing fails.
    Creates a "needs human review" output.
    """
    from src.models.output import create_error_output
    from datetime import datetime
    
    errors = state.get("errors", [])
    flags = state.get("flags", [])
    
    error_msg = "; ".join(errors) if errors else "Processing failed early in pipeline"
    
    output = create_error_output(error_msg)
    output.human_review_reasons = flags
    
    return {
        "final_output": output.model_dump(),
        "requires_human_review": True,
        "current_stage": "early_exit",
        "agent_logs": [{
            "agent": "Workflow",
            "action": "early_exit",
            "status": "terminated",
            "reason": error_msg,
            "timestamp": datetime.now().isoformat()
        }]
    }


def output_node(state: ResumeScreeningState) -> Dict[str, Any]:
    """
    Final output node - just passes through the state.
    """
    return {"current_stage": "complete"}


# ============= WORKFLOW BUILDER =============

def build_workflow() -> StateGraph:
    """
    Build the multi-agent workflow graph.
    
    Architecture:
    
    START
      │
      ▼
    [Resume Parser] ──(low confidence)──► [Early Exit] ──► END
      │
      │ (success)
      ▼
    [Job Analyzer] ──(low confidence)──► [Early Exit] ──► END
      │
      │ (success)
      ├─────────────────┐
      ▼                 ▼
    [Skills Matcher] [Experience Evaluator]
      │                 │
      └────────┬────────┘
               ▼
    [Decision Maker]
               │
               ▼
           [Output]
               │
               ▼
              END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Building resume screening workflow graph")
    
    # Create the graph builder
    builder = StateGraph(ResumeScreeningState)
    
    # Add nodes
    builder.add_node("parse_resume", run_resume_parser)
    builder.add_node("analyze_job", run_job_analyzer)
    builder.add_node("match_skills", run_skills_matcher)
    builder.add_node("evaluate_experience", run_experience_evaluator)
    builder.add_node("make_decision", run_decision_maker)
    builder.add_node("early_exit", early_exit_node)
    builder.add_node("output", output_node)
    
    # Define edges
    
    # Start with resume parsing
    builder.add_edge(START, "parse_resume")
    
    # Conditional routing after parsing
    builder.add_conditional_edges(
        "parse_resume",
        route_after_parsing,
        {
            "analyze_job": "analyze_job",
            "early_exit": "early_exit"
        }
    )
    
    # Conditional routing after job analysis
    builder.add_conditional_edges(
        "analyze_job",
        route_after_job_analysis,
        {
            "match_skills": "match_skills",
            "early_exit": "early_exit"
        }
    )
    
    # Skills matching leads to experience evaluation (sequential for simplicity)
    builder.add_edge("match_skills", "evaluate_experience")
    
    # Experience evaluation leads to decision
    builder.add_edge("evaluate_experience", "make_decision")
    
    # Decision leads to output
    builder.add_edge("make_decision", "output")
    
    # Early exit leads to output (with error state)
    builder.add_edge("early_exit", "output")
    
    # Output leads to END
    builder.add_edge("output", END)
    
    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    logger.info("Workflow graph built successfully")
    return graph


def run_screening(
    resume_text: str,
    job_description: str,
    resume_path: str = None,
    job_path: str = None,
    thread_id: str = None
) -> Dict[str, Any]:
    """
    Run the complete screening workflow.
    
    Args:
        resume_text: Extracted text from resume
        job_description: Job description text
        resume_path: Optional path to resume file
        job_path: Optional path to job description file
        thread_id: Optional thread ID for checkpointing
    
    Returns:
        Final screening output dict
    """
    import uuid
    
    # Build workflow
    workflow = build_workflow()
    
    # Create initial state
    initial_state = create_initial_state(
        resume_text=resume_text,
        job_description=job_description,
        resume_path=resume_path,
        job_path=job_path
    )
    
    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = f"screening-{uuid.uuid4().hex[:8]}"
    
    # Execute workflow
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"Starting screening workflow (thread_id={thread_id})")
    
    try:
        final_state = workflow.invoke(initial_state, config=config)
        
        logger.info(f"Workflow completed (thread_id={thread_id})")
        
        return {
            "success": True,
            "output": final_state.get("final_output", {}),
            "requires_human_review": final_state.get("requires_human_review", False),
            "flags": final_state.get("flags", []),
            "agent_logs": final_state.get("agent_logs", []),
            "thread_id": thread_id
        }
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "output": {
                "match_score": 0.0,
                "recommendation": "Needs manual review",
                "requires_human": True,
                "confidence": 0.0,
                "reasoning_summary": f"Workflow error: {e}"
            },
            "thread_id": thread_id
        }
