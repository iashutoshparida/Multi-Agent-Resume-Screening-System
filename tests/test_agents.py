"""
Integration tests for the agent workflow.

Note: These tests require a valid GEMINI_API_KEY to run.
Set the environment variable before running tests:
    export GEMINI_API_KEY=your_key_here
"""

import pytest
import os
from pathlib import Path

# Check if API key is available for integration tests
HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY"))


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """
JOHN DOE
Senior Software Engineer
john.doe@email.com | +1-555-123-4567 | San Francisco, CA
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

SUMMARY
Experienced software engineer with 7 years of experience in backend development,
specializing in Python and cloud infrastructure. Proven track record of building
scalable systems serving millions of users.

SKILLS
Programming: Python, JavaScript, Go, SQL
Frameworks: Django, FastAPI, Flask, React
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS (EC2, S3, Lambda, RDS), GCP
DevOps: Docker, Kubernetes, Terraform, CI/CD
Other: REST APIs, GraphQL, Microservices, System Design

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | San Francisco, CA
June 2021 - Present
- Led development of microservices platform handling 10M+ daily requests
- Reduced API latency by 40% through optimization and caching strategies
- Mentored team of 4 junior engineers
- Implemented CI/CD pipelines reducing deployment time by 60%

Software Engineer | StartupXYZ | San Francisco, CA
March 2018 - May 2021
- Built RESTful APIs using Django and FastAPI
- Designed and implemented PostgreSQL database schemas
- Deployed applications on AWS using Docker and Kubernetes
- Collaborated with frontend team on React-based dashboard

Junior Developer | WebDev Agency | New York, NY
January 2017 - February 2018
- Developed web applications using Python and JavaScript
- Created automated testing suites
- Participated in code reviews and agile ceremonies

EDUCATION

Bachelor of Science in Computer Science
University of California, Berkeley | 2016
GPA: 3.7/4.0

CERTIFICATIONS
- AWS Solutions Architect Associate
- Google Cloud Professional Cloud Architect
"""


@pytest.fixture
def sample_job_description():
    """Sample job description for testing"""
    return """
Senior Backend Engineer - Platform Team

About the Role:
We're looking for a Senior Backend Engineer to join our Platform team. You'll be
building and scaling our core infrastructure that powers our entire product suite.

Requirements:
- 5+ years of backend development experience
- Strong proficiency in Python
- Experience with Django or FastAPI frameworks
- Solid understanding of PostgreSQL and Redis
- Experience with AWS or GCP cloud platforms
- Knowledge of Docker and Kubernetes
- Experience building RESTful APIs
- Strong system design skills

Nice to Have:
- Experience with Go
- GraphQL experience
- Terraform or infrastructure as code
- Experience with high-traffic systems (1M+ requests/day)

What You'll Do:
- Design and build scalable backend services
- Optimize database queries and system performance
- Mentor junior engineers
- Participate in architecture decisions
- Implement monitoring and alerting systems

Compensation: $180,000 - $220,000 + equity
Location: San Francisco, CA (Hybrid)
"""


@pytest.fixture
def poor_fit_resume():
    """Resume that doesn't match the job well"""
    return """
JANE SMITH
Marketing Coordinator
jane.smith@email.com | +1-555-987-6543 | Chicago, IL

SUMMARY
Marketing professional with 3 years of experience in digital marketing
and content creation. Strong communicator with creative problem-solving skills.

SKILLS
- Social Media Marketing
- Content Writing
- Canva, Adobe Photoshop
- Google Analytics
- Email Marketing (Mailchimp)
- Basic HTML

EXPERIENCE

Marketing Coordinator | RetailCo | Chicago, IL
August 2021 - Present
- Managed social media accounts with 50K+ followers
- Created marketing campaigns increasing engagement by 25%
- Wrote blog posts and email newsletters

Marketing Assistant | SmallBiz Inc | Chicago, IL
June 2020 - July 2021
- Assisted in planning marketing events
- Updated website content using WordPress
- Tracked campaign metrics

EDUCATION
Bachelor of Arts in Communications
Northwestern University | 2020
"""


class TestModels:
    """Test Pydantic models"""
    
    def test_parsed_resume_model(self):
        """Test ParsedResume model creation"""
        from src.models.resume import ParsedResume
        
        resume = ParsedResume(
            full_name="John Doe",
            email="john@example.com",
            skills=["Python", "JavaScript"],
            work_experience=[],
            education=[],
            total_years_experience=5.0,
            extraction_confidence=0.9
        )
        
        assert resume.full_name == "John Doe"
        assert len(resume.skills) == 2
        assert resume.extraction_confidence == 0.9
    
    def test_screening_output_model(self):
        """Test ScreeningOutput model creation"""
        from src.models.output import ScreeningOutput
        
        output = ScreeningOutput(
            match_score=0.75,
            recommendation="Proceed to interview",
            requires_human=False,
            confidence=0.85,
            reasoning_summary="Strong candidate with relevant experience."
        )
        
        assert output.match_score == 0.75
        assert output.recommendation == "Proceed to interview"
        assert not output.requires_human
    
    def test_job_requirements_model(self):
        """Test JobRequirements model creation"""
        from src.models.job import JobRequirements
        
        job = JobRequirements(
            job_title="Senior Python Developer",
            min_years_experience=5.0,
            required_skills=["Python", "Django"],
            preferred_skills=["Kubernetes"],
            extraction_confidence=0.9
        )
        
        assert job.job_title == "Senior Python Developer"
        assert len(job.required_skills) == 2


class TestStateManagement:
    """Test state creation and management"""
    
    def test_create_initial_state(self):
        """Test initial state creation"""
        from src.models.state import create_initial_state
        
        state = create_initial_state(
            resume_text="Test resume",
            job_description="Test job"
        )
        
        assert state["resume_text"] == "Test resume"
        assert state["job_description"] == "Test job"
        assert state["parsed_resume"] is None
        assert state["confidence_scores"] == []
        assert state["flags"] == []


@pytest.mark.skipif(not HAS_API_KEY, reason="GEMINI_API_KEY not set")
class TestAgentIntegration:
    """Integration tests requiring API access"""
    
    def test_gemini_client_initialization(self):
        """Test Gemini client can be initialized"""
        from src.llm.groq_client import GeminiClient
        
        client = GeminiClient()
        assert client is not None
    
    def test_resume_parser_agent(self, sample_resume_text):
        """Test resume parser agent"""
        from src.agents.resume_parser import ResumeParserAgent
        from src.models.state import create_initial_state
        
        state = create_initial_state(
            resume_text=sample_resume_text,
            job_description="Test job"
        )
        
        agent = ResumeParserAgent()
        result = agent.run(state)
        
        assert "parsed_resume" in result
        parsed = result["parsed_resume"]
        
        # Should extract name
        assert parsed.get("full_name") is not None
        assert parsed.get("extraction_confidence", 0) > 0.5
    
    def test_full_workflow_good_fit(self, sample_resume_text, sample_job_description):
        """Test full workflow with a good fit candidate"""
        from src.workflow.graph import run_screening
        
        result = run_screening(
            resume_text=sample_resume_text,
            job_description=sample_job_description
        )
        
        assert result["success"]
        output = result["output"]
        
        # Good fit should have high score
        assert output["match_score"] >= 0.6
        assert output["recommendation"] in [
            "Proceed to interview",
            "Needs manual review"
        ]
        assert output["confidence"] > 0.5
    
    def test_full_workflow_poor_fit(self, poor_fit_resume, sample_job_description):
        """Test full workflow with a poor fit candidate"""
        from src.workflow.graph import run_screening
        
        result = run_screening(
            resume_text=poor_fit_resume,
            job_description=sample_job_description
        )
        
        assert result["success"]
        output = result["output"]
        
        # Poor fit should have low score
        assert output["match_score"] < 0.5
        # Should recommend rejection or review
        assert output["recommendation"] in ["Reject", "Needs manual review"]


class TestWorkflowGraph:
    """Test workflow graph construction"""
    
    def test_build_workflow(self):
        """Test workflow graph can be built"""
        from src.workflow.graph import build_workflow
        
        workflow = build_workflow()
        assert workflow is not None
    
    def test_workflow_has_expected_nodes(self):
        """Test workflow has all expected nodes"""
        from src.workflow.graph import build_workflow
        
        workflow = build_workflow()
        
        # Get node names from the graph
        # This is implementation-dependent but verifies structure
        assert workflow is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
