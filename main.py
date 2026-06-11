#!/usr/bin/env python3
"""
Agentic Resume Screening Assistant

CLI entry point for running resume screening.

Usage:
    python main.py <resume_file> <job_description_file> [options]
    
Example:
    python main.py resume.pdf job.txt -o output.json
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

from src.config import setup_logging, get_config
from src.utils.validators import validate_resume_file, validate_job_description
from src.workflow.graph import run_screening


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Agentic Resume Screening Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python main.py resume.pdf job_description.txt
    
    # Save output to file
    python main.py resume.pdf job.txt -o result.json
    
    # Verbose output
    python main.py resume.pdf job.txt -v
    
    # Process with raw job description
    python main.py resume.pdf "Senior Python Developer with 5+ years experience"
        """
    )
    
    parser.add_argument(
        "resume",
        help="Path to resume file (PDF, DOCX, or TXT)"
    )
    
    parser.add_argument(
        "job_description",
        help="Path to job description file OR raw job description text"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: print to stdout)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show agent execution logs"
    )
    
    return parser.parse_args()


def print_colored(text: str, color: str = None, bold: bool = False, no_color: bool = False):
    """Print with optional color formatting"""
    if no_color or not sys.stdout.isatty():
        print(text)
        return
    
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
        "bold": "\033[1m"
    }
    
    prefix = ""
    if bold:
        prefix += colors["bold"]
    if color and color in colors:
        prefix += colors[color]
    
    suffix = colors["reset"]
    
    print(f"{prefix}{text}{suffix}")


def format_output(result: dict, show_logs: bool = False, no_color: bool = False) -> str:
    """Format the screening result for display"""
    output = result.get("output", {})
    
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("RESUME SCREENING RESULT")
    lines.append("=" * 60)
    
    # Main recommendation
    recommendation = output.get("recommendation", "Unknown")
    match_score = output.get("match_score", 0)
    confidence = output.get("confidence", 0)
    
    lines.append(f"\nCandidate: {output.get('candidate_name', 'Unknown')}")
    lines.append(f"Position:  {output.get('job_title', 'Unknown')}")
    lines.append("")
    
    # Color-coded recommendation
    rec_color = {
        "Proceed to interview": "green",
        "Reject": "red",
        "Needs manual review": "yellow"
    }.get(recommendation, "white")
    
    lines.append(f"Recommendation: {recommendation}")
    lines.append(f"Match Score:    {match_score:.0%}")
    lines.append(f"Confidence:     {confidence:.0%}")
    lines.append(f"Human Review:   {'Yes' if output.get('requires_human', False) else 'No'}")
    
    # Reasoning
    lines.append("\n" + "-" * 40)
    lines.append("REASONING")
    lines.append("-" * 40)
    lines.append(output.get("reasoning_summary", "No reasoning provided"))
    
    # Strengths
    strengths = output.get("strengths", [])
    if strengths:
        lines.append("\n" + "-" * 40)
        lines.append("STRENGTHS")
        lines.append("-" * 40)
        for s in strengths:
            lines.append(f"  ✓ {s}")
    
    # Concerns
    concerns = output.get("concerns", [])
    if concerns:
        lines.append("\n" + "-" * 40)
        lines.append("CONCERNS")
        lines.append("-" * 40)
        for c in concerns:
            lines.append(f"  ⚠ {c}")
    
    # Skill gaps
    skill_gaps = output.get("skill_gaps", [])
    if skill_gaps:
        lines.append("\n" + "-" * 40)
        lines.append("SKILL GAPS")
        lines.append("-" * 40)
        for g in skill_gaps:
            lines.append(f"  ✗ {g}")
    
    # Flags
    flags = result.get("flags", [])
    if flags:
        lines.append("\n" + "-" * 40)
        lines.append("FLAGS")
        lines.append("-" * 40)
        for f in flags:
            lines.append(f"  ⚑ {f}")
    
    # Human review reasons
    human_reasons = output.get("human_review_reasons", [])
    if human_reasons:
        lines.append("\n" + "-" * 40)
        lines.append("HUMAN REVIEW REASONS")
        lines.append("-" * 40)
        for r in human_reasons:
            lines.append(f"  • {r}")
    
    # Agent logs (optional)
    if show_logs:
        agent_logs = result.get("agent_logs", [])
        if agent_logs:
            lines.append("\n" + "-" * 40)
            lines.append("AGENT LOGS")
            lines.append("-" * 40)
            for log in agent_logs:
                lines.append(f"  [{log.get('agent', 'Unknown')}] {log.get('message', '')}")
    
    lines.append("\n" + "=" * 60)
    lines.append(f"Timestamp: {output.get('evaluation_timestamp', datetime.now().isoformat())}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check API key
    config = get_config()
    if not config.validate_api_key():
        print_colored(
            "ERROR: GROQ_API_KEY not configured.\n\n"
            "Setup (2 minutes, FREE, no credit card):\n"
            "  1. Go to: https://console.groq.com\n"
            "  2. Sign up and create an API key\n"
            "  3. Set the key:\n"
            "     export GROQ_API_KEY=gsk_your_key_here\n\n"
            "Or add to .env file:\n"
            "     GROQ_API_KEY=gsk_your_key_here",
            color="red",
            bold=True,
            no_color=args.no_color
        )
        sys.exit(1)
    
    # Validate resume
    print_colored(f"Loading resume: {args.resume}", color="cyan", no_color=args.no_color)
    resume_result = validate_resume_file(args.resume)
    
    if not resume_result.valid:
        print_colored(f"ERROR: {resume_result.error}", color="red", bold=True, no_color=args.no_color)
        sys.exit(1)
    
    if resume_result.warnings:
        for warning in resume_result.warnings:
            print_colored(f"WARNING: {warning}", color="yellow", no_color=args.no_color)
    
    print_colored(
        f"Resume loaded: {len(resume_result.text)} chars, confidence: {resume_result.confidence:.0%}",
        color="green",
        no_color=args.no_color
    )
    
    # Validate job description
    print_colored(f"Loading job description...", color="cyan", no_color=args.no_color)
    job_result = validate_job_description(args.job_description)
    
    if not job_result.valid:
        print_colored(f"ERROR: {job_result.error}", color="red", bold=True, no_color=args.no_color)
        sys.exit(1)
    
    if job_result.warnings:
        for warning in job_result.warnings:
            print_colored(f"WARNING: {warning}", color="yellow", no_color=args.no_color)
    
    print_colored(
        f"Job description loaded: {len(job_result.text)} chars",
        color="green",
        no_color=args.no_color
    )
    
    # Run screening
    print_colored("\nRunning screening workflow...", color="cyan", bold=True, no_color=args.no_color)
    print_colored("This may take a minute due to multiple LLM calls.\n", color="white", no_color=args.no_color)
    
    try:
        result = run_screening(
            resume_text=resume_result.text,
            job_description=job_result.text,
            resume_path=args.resume,
            job_path=args.job_description if Path(args.job_description).exists() else None
        )
    except Exception as e:
        print_colored(f"ERROR: Screening failed: {e}", color="red", bold=True, no_color=args.no_color)
        logger.exception("Screening failed")
        sys.exit(1)
    
    # Check for success
    if not result.get("success", False):
        print_colored(
            f"ERROR: Screening failed: {result.get('error', 'Unknown error')}",
            color="red",
            bold=True,
            no_color=args.no_color
        )
        sys.exit(1)
    
    # Output results
    if args.output:
        # Save to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result["output"], f, indent=2)
        print_colored(f"\nResults saved to: {output_path}", color="green", no_color=args.no_color)
    
    # Print formatted output
    formatted = format_output(result, show_logs=args.show_logs, no_color=args.no_color)
    print(formatted)
    
    # Print JSON for easy copying
    if args.verbose:
        print("\nJSON Output:")
        print(json.dumps(result["output"], indent=2))
    
    # Exit code based on recommendation
    recommendation = result["output"].get("recommendation", "")
    if recommendation == "Reject":
        sys.exit(2)  # Different exit code for rejection
    elif recommendation == "Needs manual review":
        sys.exit(3)  # Different exit code for manual review
    
    sys.exit(0)


if __name__ == "__main__":
    main()
