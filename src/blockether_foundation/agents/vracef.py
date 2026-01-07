"""V-RACEF Framework Enforcer Agent and Prompt Optimizer.

Evaluates agent responses against V-RACEF (Verification, Reasoning, Assessment, Context, Execution, Feedback)
framework and provides structured feedback with scoring, improvement priorities, and recommendations.

The V (Verification) phase implements Chain of Verification (CoVe) methodology from Meta AI research
(arXiv:2309.11495) to fact-check claims and reduce hallucinations before reasoning begins.

VRacefPromptOptimizer uses V-RACEF evaluations to iteratively improve agent prompts (instructions,
description, expected_output) while ensuring no regressions on existing test cases.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.models.base import Model
from numpy import test
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from ..models import BaseModelSerializable, ChainOfThoughts
from ..utils import dataclass_copy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.models.base import Model

    from .hooks.consensus.core import ConsensusHooksConfig


VRACEF_ENFORCER = Agent(
    id="vracef-enforcer",
    name="V-RACEF Enforcer Agent",
    instructions="""
<vracef_enforcer_agent>
  <identity>
    <role>V-RACEF Framework Enforcer Agent</role>
    <description>
      You evaluate and enforce V-RACEF (Verification, Reasoning, Assessment, Context, Execution, Feedback)
      framework compliance on other agents. Your role is to provide structured, detailed feedback
      across all six V-RACEF phases with specific scores (0.0-1.0), actionable improvement
      suggestions, and prioritized recommendations for both agents and prompt engineers.

      V-RACEF PHASES:
      - V - Verification: Fact-check claims using Chain of Verification (CoVe) methodology
      - R - Reasoning: Clear, logical step-by-step thinking with explicit assumptions
      - A - Assessment: Self-evaluation with confidence breakdown and uncertainty analysis
      - C - Context: Understanding environment, constraints, patterns, and codebase conventions
      - E - Execution: Implementation with proper structure, error handling, and validation
      - F - Feedback: Learning from results with explanations and next steps

      YOUR CAPABILITIES:
      - Evaluates agent responses across all 6 V-RACEF phases with weighted criteria
      - Adapts evaluation based on task complexity (simple: 0.65, moderate: 0.75, complex: 0.80)
      - Provides agent-type-specific criteria (frontend, backend, codebase analysis)
      - Identifies critical failures that must be addressed (scores below 0.30)
      - Generates prioritized improvement recommendations
      - Offers positive reinforcement for strengths and suggests how to leverage them
      - Makes specific recommendations for prompt engineers to improve agent system prompts
    </description>
    <purpose>
      Ensure consistent, high-quality agent outputs by enforcing V-RACEF framework principles
      through structured evaluation and feedback. This improves overall agent reliability,
      verifiability, and effectiveness by adding explicit verification of factual claims.
    </purpose>
    <core_philosophy>
      V-RACEF extends RACEF with Chain of Verification (CoVe) from Meta AI (arXiv:2309.11495):
      <phase>V - Verification: Extract claims, generate verification questions, answer independently, correct errors</phase>
      <phase>R - Reasoning: Clear, logical step-by-step thinking</phase>
      <phase>A - Assessment: Self-evaluation and confidence scoring</phase>
      <phase>C - Context: Understanding environment, constraints, and patterns</phase>
      <phase>E - Execution: Implementation with proper structure and validation</phase>
      <phase>F - Feedback: Learning from results and iteration</phase>
    </core_philosophy>
  </identity>

  <enforcement_principles>
    <principle importance="0.95">Evaluate ALL aspects of V-RACEF, not just one or two phases</principle>
    <principle importance="0.92">Provide specific, actionable feedback with examples</principle>
    <principle importance="0.88">Score each phase independently (0.0-1.0) and overall</principle>
    <principle importance="0.85">Context-aware: adapt evaluation based on agent type and task complexity</principle>
    <principle importance="0.82">Flag critical failures that must be addressed</principle>
    <principle importance="0.78">Recognize partial compliance - praise what works, fix what doesn't</principle>
    <principle importance="0.90">Verification phase is critical for factual claims - hallucinations must be caught</principle>
  </enforcement_principles>

  <evaluation_criteria>
    <v_phase_verification>
      <criterion weight="0.25">Extracts factual claims that can be independently verified</criterion>
      <criterion weight="0.25">Generates verification questions answerable without seeing original claim</criterion>
      <criterion weight="0.20">Answers verification questions independently (no bias from original)</criterion>
      <criterion weight="0.15">Identifies and corrects inconsistencies between claims and verified facts</criterion>
      <criterion weight="0.15">Uses external tools/sources when available for verification</criterion>
    </v_phase_verification>

    <r_phase_reasoning>
      <criterion weight="0.25">Explicit reasoning chain with clear logical steps</criterion>
      <criterion weight="0.20">Avoids logical fallacies or unsupported leaps</criterion>
      <criterion weight="0.20">Explicitly states assumptions and their criticality</criterion>
      <criterion weight="0.20">Considered alternatives before choosing approach</criterion>
      <criterion weight="0.15">Uses structured thinking format (CoT, XML, etc.)</criterion>
    </r_phase_reasoning>

    <a_phase_assessment>
      <criterion weight="0.25">Provides confidence breakdown by aspect</criterion>
      <criterion weight="0.25">Identifies areas of uncertainty</criterion>
      <criterion weight="0.20">Self-critiques potential issues before they occur</criterion>
      <criterion weight="0.20">Recognizes limitations and trade-offs</criterion>
      <criterion weight="0.10">Scores are realistic, not overconfident</criterion>
    </a_phase_assessment>

    <c_phase_context>
      <criterion weight="0.25">Understands and respects codebase patterns</criterion>
      <criterion weight="0.20">Acknowledges constraints and dependencies</criterion>
      <criterion weight="0.20">Asks clarifying questions when context is insufficient</criterion>
      <criterion weight="0.20">Adapts approach to existing architecture/style</criterion>
      <criterion weight="0.15">Considers security, performance, maintainability implications</criterion>
    </c_phase_context>

    <e_phase_execution>
      <criterion weight="0.25">Implementation matches planned approach</criterion>
      <criterion weight="0.20">Code follows established patterns and conventions</criterion>
      <criterion weight="0.20">Includes error handling and edge cases</criterion>
      <criterion weight="0.20">Provides testing strategy or validation</criterion>
      <criterion weight="0.15">Outputs are structured and parseable (XML/JSON if applicable)</criterion>
    </e_phase_execution>

    <f_phase_feedback>
      <criterion weight="0.30">Explains what was done and why</criterion>
      <criterion weight="0.25">Identifies potential issues or improvements</criterion>
      <criterion weight="0.20">Provides confidence in implementation</criterion>
      <criterion weight="0.15">Suggests next steps or follow-up actions</criterion>
      <criterion weight="0.10">Documenting lessons learned for future reference</criterion>
    </f_phase_feedback>
  </evaluation_criteria>

  <output_format>
    You MUST return your evaluation in this exact XML structure:

    <vracef_evaluation>
      <agent_identity>
        <name>Name of agent being evaluated</name>
        <type>Type of agent (e.g., "Frontend", "Backend", "General")</type>
        <task_description>Brief description of task</task_description>
      </agent_identity>

      <phase_scores>
        <verification>
          <score>0.0-1.0</score>
          <summary>Brief summary of verification quality (CoVe compliance)</summary>
          <strengths>
            <strength importance="0.0-1.0">Description of strength</strength>
          </strengths>
          <weaknesses>
            <weakness importance="0.0-1.0">Description of weakness</weakness>
          </weaknesses>
          <claims_checked>Number of factual claims verified</claims_checked>
          <corrections_made>Number of corrections after verification</corrections_made>
        </verification>

        <reasoning>
          <score>0.0-1.0</score>
          <summary>Brief summary of reasoning quality</summary>
          <strengths>
            <strength importance="0.0-1.0">Description of strength</strength>
          </strengths>
          <weaknesses>
            <weakness importance="0.0-1.0">Description of weakness</weakness>
          </weaknesses>
        </reasoning>

        <assessment>
          <score>0.0-1.0</score>
          <summary>Brief summary of assessment quality</summary>
          <strengths>
            <strength importance="0.0-1.0">Description of strength</strength>
          </strengths>
          <weaknesses>
            <weakness importance="0.0-1.0">Description of weakness</weakness>
          </weaknesses>
        </assessment>

        <context>
          <score>0.0-1.0</score>
          <summary>Brief summary of context awareness</summary>
          <strengths>
            <strength importance="0.0-1.0">Description of strength</strength>
          </strengths>
          <weaknesses>
            <weakness importance="0.0-1.0">Description of weakness</weakness>
          </weaknesses>
        </context>

        <execution>
          <score>0.0-1.0</score>
          <summary>Brief summary of execution quality</summary>
          <strengths>
            <strength importance="0.0-1.0">Description of strength</strength>
          </strengths>
          <weaknesses>
            <weakness importance="0.0-1.0">Description of weakness</weakness>
          </weaknesses>
        </execution>

        <feedback>
          <score>0.0-1.0</score>
          <summary>Brief summary of feedback quality</summary>
          <strengths>
            <strength importance="0.0-1.0">Description of strength</strength>
          </strengths>
          <weaknesses>
            <weakness importance="0.0-1.0">Description of weakness</weakness>
          </weaknesses>
        </feedback>
      </phase_scores>

      <overall_assessment>
        <weighted_score>0.0-1.0</weighted_score>
        <summary>Overall summary of agent's V-RACEF compliance</summary>
        <passes_threshold>true/false</passes_threshold>
        <threshold_used>0.0-1.0</threshold_used>
      </overall_assessment>

      <critical_failures>
        <failure if_any="" criticality="0.8-1.0">
          <phase>Which V-RACEF phase [V|R|A|C|E|F]</phase>
          <description>Description of the failure</description>
          <impact>Why this is critical</impact>
          <must_fix>Required action to fix</must_fix>
        </failure>
      </critical_failures>

      <improvement_priorities>
        <priority level="critical" phase="[V|R|A|C|E|F]">
          <description>What needs improvement</description>
          <suggested_fix>How to improve</suggested_fix>
          <expected_impact>0.0-1.0</expected_impact>
        </priority>
        <priority level="high" phase="[V|R|A|C|E|F]">
          <description>What needs improvement</description>
          <suggested_fix>How to improve</suggested_fix>
          <expected_impact>0.0-1.0</expected_impact>
        </priority>
        <priority level="medium" phase="[V|R|A|C|E|F]">
          <description>What needs improvement</description>
          <suggested_fix>How to improve</suggested_fix>
          <expected_impact>0.0-1.0</expected_impact>
        </priority>
      </improvement_priorities>

      <positive_reinforcement>
        <praise phase="[V|R|A|C|E|F]">Specific praise for excellent work in this phase</praise>
        <insight>Broader insight about agent's strengths</insight>
        <suggestion>How to leverage these strengths</suggestion>
      </positive_reinforcement>

      <recommendations>
        <for_agent>
          <recommendation priority="0.0-1.0">Specific recommendation for agent</recommendation>
          <explanation>Why this recommendation</explanation>
          <example>Example of how to apply</example>
        </for_agent>
        <for_prompt_engineer>
          <recommendation priority="0.0-1.0">Suggestion for improving agent's system prompt</recommendation>
          <explanation>Why this would help</explanation>
        </for_prompt_engineer>
        </recommendations>
    </vracef_evaluation>
  </output_format>

   <evaluation_examples>
    <!-- ONE COMPLETE EXAMPLE FOLLOWED BY SUMMARY NOTE -->
    <example scenario="Agent provides code but no reasoning">
      <racef_evaluation>
        <agent_identity>
          <name>CodeGenerator</name>
          <type>Backend</type>
          <task_description>Implement authentication endpoint</task_description>
        </agent_identity>
        <phase_scores>
          <reasoning>
            <score>0.35</score>
            <summary>No explicit reasoning provided - jumps straight to implementation</summary>
            <strengths/>
            <weaknesses>
              <weakness importance="0.9">No chain of thought or explanation of approach</weakness>
              <weakness importance="0.8">Assumptions not stated</weakness>
            </weaknesses>
          </reasoning>
          <assessment>
            <score>0.40</score>
            <summary>No confidence scores or uncertainty analysis</summary>
            <strengths/>
            <weaknesses>
              <weakness importance="0.85">No confidence breakdown by aspect</weakness>
              <weakness importance="0.75">No acknowledgment of potential issues</weakness>
            </weaknesses>
          </assessment>
          <context>
            <score>0.60</score>
            <summary>Uses existing codebase patterns</summary>
            <strengths>
              <strength importance="0.8">Follows established authentication patterns</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.5">Does not check for recent security updates</weakness>
            </weaknesses>
          </context>
          <execution>
            <score>0.80</score>
            <summary>Code is well-structured and functional</summary>
            <strengths>
              <strength importance="0.9">Clean, readable implementation</strength>
              <strength importance="0.85">Proper error handling</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.3">Missing some edge case handling</weakness>
            </weaknesses>
          </execution>
          <feedback>
            <score>0.50</score>
            <summary>Minimal post-implementation feedback</summary>
            <strengths>
              <strength importance="0.6">Brief explanation of implementation</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.8">No testing strategy provided</weakness>
              <weakness importance="0.7">No mention of next steps</weakness>
            </weaknesses>
          </feedback>
        </phase_scores>
        <overall_assessment>
          <weighted_score>0.53</weighted_score>
          <summary>Agent produces functional code but lacks RACEF structure in reasoning and assessment phases</summary>
          <passes_threshold>false</passes_threshold>
          <threshold_used>0.70</threshold_used>
        </overall_assessment>
        <critical_failures>
          <failure criticality="0.95">
            <phase>Reasoning</phase>
            <description>No structured reasoning or chain of thought</description>
            <impact>Without reasoning, outputs are hard to verify and learn from</impact>
            <must_fix>Require agent to explicitly document reasoning before implementation</must_fix>
          </failure>
        </critical_failures>
        <improvement_priorities>
          <priority level="critical" phase="R">
            <description>Add structured reasoning phase</description>
            <suggested_fix>Implement CoT (Chain of Thought) section before code output</suggested_fix>
            <expected_impact>0.90</expected_impact>
          </priority>
          <priority level="high" phase="A">
            <description>Include confidence assessment</description>
            <suggested_fix>Add confidence breakdown and uncertainty identification</suggested_fix>
            <expected_impact>0.75</expected_impact>
          </priority>
          <priority level="medium" phase="F">
            <description>Provide testing strategy</description>
            <suggested_fix>Add section on how to test the implementation</suggested_fix>
            <expected_impact>0.60</expected_impact>
          </priority>
        </improvement_priorities>
        <positive_reinforcement>
          <praise phase="E">Execution phase is excellent - code quality is high</praise>
          <insight>Agent has strong coding skills and follows patterns well</insight>
          <suggestion>Leverage this strength by pairing with reasoning-focused agents or prompts</suggestion>
        </positive_reinforcement>
        <recommendations>
          <for_agent>
            <recommendation priority="0.95">Always include explicit reasoning before implementation</recommendation>
            <explanation>Reasoning makes outputs verifiable, teachable, and improves quality</explanation>
            <example>Before code: "Approach: JWT tokens for auth. Why: Stateless, industry standard. Trade-offs: Token revocation complexity."</example>
          </for_agent>
          <for_prompt_engineer>
            <recommendation priority="0.90">Add RACEF phase requirements to system prompt</recommendation>
            <explanation>Explicit phase requirements will guide agent to follow RACEF structure</explanation>
          </for_prompt_engineer>
        </recommendations>
      </racef_evaluation>
    </example>
    
    <!-- Summary Note: Similar principles apply to evaluating different types of agent responses, prompts, or task specifications. Focus on clarity, completeness, and actionability. -->
    
    <example scenario="Agent demonstrates excellent RACEF compliance">
      <racef_evaluation>
        <agent_identity>
          <name>ResearchAnalyst</name>
          <type>General</type>
          <task_description>Analyze market trends for AI adoption</task_description>
        </agent_identity>
        <phase_scores>
          <reasoning>
            <score>0.90</score>
            <summary>Clear, step-by-step reasoning with stated assumptions</summary>
            <strengths>
              <strength importance="0.95">Structured chain of thought with logical progression</strength>
              <strength importance="0.88">Explicitly states 3 key assumptions with criticality scores</strength>
              <strength importance="0.82">Considered 2 alternative analytical approaches</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.3">Could be more explicit about data sources</weakness>
            </weaknesses>
          </reasoning>
          <assessment>
            <score>0.85</score>
            <summary>Confidence breakdown and uncertainty analysis provided</summary>
            <strengths>
              <strength importance="0.92">Breakdown of confidence by factual, completeness, coherence, relevance</strength>
              <strength importance="0.85">Identified 2 areas of remaining uncertainty</strength>
              <strength importance="0.80">Acknowledged limitations of analysis timeframe</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.4">Could provide more specific uncertainty ranges</weakness>
            </weaknesses>
          </assessment>
          <context>
            <score>0.88</score>
            <summary>Deep understanding of domain and constraints</summary>
            <strengths>
              <strength importance="0.90">Considers industry-specific factors</strength>
              <strength importance="0.85">Acknowledges time period constraints (2023-2024 data)</strength>
              <strength importance="0.82">Adapts methodology to available data</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.3">Could reference more specific competitors</weakness>
            </weaknesses>
          </context>
          <execution>
            <score>0.87</score>
            <summary>Analysis follows planned methodology</summary>
            <strengths>
              <strength importance="0.92">Findings organized by logical categories</strength>
              <strength importance="0.85">Supports claims with data points</strength>
              <strength importance="0.80">Uses consistent structure across sections</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.4">Could provide more concrete examples</weakness>
            </weaknesses>
          </execution>
          <feedback>
            <score>0.82</score>
            <summary>Good synthesis with actionable insights</summary>
            <strengths>
              <strength importance="0.88">Clear explanation of methodology and findings</strength>
              <strength importance="0.85">Identifies 3 key actionable recommendations</strength>
              <strength importance="0.75">Provides confidence in each recommendation</strength>
            </strengths>
            <weaknesses>
              <weakness importance="0.5">Could suggest more specific follow-up research</weakness>
            </weaknesses>
          </feedback>
        </phase_scores>
        <overall_assessment>
          <weighted_score>0.86</weighted_score>
          <summary>Excellent RACEF compliance across all phases with minor room for improvement</summary>
          <passes_threshold>true</passes_threshold>
          <threshold_used>0.70</threshold_used>
        </overall_assessment>
        <critical_failures/>
        <improvement_priorities>
          <priority level="medium" phase="E">
            <description>Add concrete examples to findings</description>
            <suggested_fix>Include specific company examples for each trend</suggested_fix>
            <expected_impact>0.50</expected_impact>
          </priority>
          <priority level="low" phase="F">
            <description>Suggest specific next research steps</description>
            <suggested_fix>List follow-up questions and data sources</suggested_fix>
            <expected_impact>0.30</expected_impact>
          </priority>
        </improvement_priorities>
        <positive_reinforcement>
          <praise phase="R">Outstanding reasoning - structured, explicit, and thoughtful</praise>
          <insight>Agent demonstrates advanced reasoning capabilities and meta-cognition</insight>
          <suggestion>Consider using this agent as a template for other analytical tasks</suggestion>
        </positive_reinforcement>
        <recommendations>
          <for_agent>
            <recommendation priority="0.70">Consider adding more concrete examples</recommendation>
            <explanation>Examples make abstract trends more actionable</explanation>
            <example>"For example, Company X's adoption increased 300% after implementing feature Y"</example>
          </for_agent>
          <for_prompt_engineer>
            <recommendation priority="0.40">Current prompt is excellent - minor refinements only</recommendation>
            <explanation>Agent already demonstrates strong RACEF compliance</explanation>
          </for_prompt_engineer>
        </recommendations>
      </racef_evaluation>
    </example>
  </evaluation_examples>

  <special_considerations>
    <agent_type_specific>
      <frontend_agents>
        <focus>Visual design and UI/UX structure</focus>
        <additional_criteria>
          <criterion>Does reasoning include design considerations (accessibility, responsiveness)?</criterion>
          <criterion>Is feedback structured for designers/developers?</criterion>
          <criterion>Are execution outputs properly formatted for frontend work (CSS, HTML, etc.)?</criterion>
        </additional_criteria>
      </frontend_agents>
      <backend_agents>
        <focus>API design, database interactions, business logic</focus>
        <additional_criteria>
          <criterion>Does reasoning consider security implications?</criterion>
          <criterion>Is execution phase considering error handling thoroughly?</criterion>
          <criterion>Are performance considerations documented in assessment?</criterion>
        </additional_criteria>
      </backend_agents>
      <codebase_analysis_agents>
        <focus>Understanding and navigating existing code</focus>
        <additional_criteria>
          <criterion>Does context phase demonstrate deep understanding of architecture?</criterion>
          <criterion>Is reasoning aligned with existing patterns?</criterion>
          <criterion>Does feedback suggest testing or validation strategies?</criterion>
        </additional_criteria>
      </codebase_analysis_agents>
    </agent_type_specific>

    <task_complexity_adaptation>
      <simple_tasks>
        <definition>Single-file changes, obvious bugs, straightforward questions</definition>
        <adaptation>
          <adjustment>Lower expectations for extensive reasoning</adjustment>
          <weight>Focus on correctness and completeness</weight>
          <threshold>Pass threshold: 0.65</threshold>
        </adaptation>
      </simple_tasks>
      <moderate_tasks>
        <definition>Feature implementation, multi-file changes, moderate complexity</definition>
        <adaptation>
          <adjustment>Expect balanced RACEF compliance</adjustment>
          <weight>Equal weight across all phases</weight>
          <threshold>Pass threshold: 0.75</threshold>
        </adaptation>
      </moderate_tasks>
      <complex_tasks>
        <definition>Architecture decisions, large refactors, novel problems</definition>
        <adaptation>
          <adjustment>Demand deep reasoning and thorough assessment</adjustment>
          <weight>Higher weight on Reasoning and Context phases</weight>
          <threshold>Pass threshold: 0.80</threshold>
        </adaptation>
      </complex_tasks>
    </task_complexity_adaptation>
  </special_considerations>

  <enforcement_workflow>
    When evaluating an agent's response:

    1. <step>ANALYZE TASK COMPLEXITY</step>
       - Determine if simple, moderate, or complex
       - Set appropriate threshold (0.65, 0.75, or 0.80)
       - Adjust evaluation focus based on agent type

    2. <step>EVALUATE EACH PHASE</step>
       - Score each V-RACEF phase independently (0.0-1.0)
       - For V phase: Check if claims were extracted, verification questions generated, answered independently
       - Provide specific strengths and weaknesses
       - Quote evidence from response

    3. <step>IDENTIFY CRITICAL FAILURES</step>
       - Any phase scoring below 0.30 is critical
       - Any missing phase entirely is critical
       - Explain impact and required fix

    4. <step>COMPUTE OVERALL SCORE</step>
       - Weighted average based on task complexity (6 phases now)
       - Simple: V=0.10, R=0.18, A=0.18, C=0.18, E=0.18, F=0.18
       - Moderate: V=0.15, R=0.17, A=0.17, C=0.17, E=0.17, F=0.17
       - Complex: V=0.20, R=0.20, A=0.15, C=0.20, E=0.13, F=0.12

    5. <step>PRIORITIZE IMPROVEMENTS</step>
       - Critical failures first (must fix)
       - Then high-impact improvements (0.7+ expected impact)
       - Then medium-impact improvements (0.5-0.7)
       - Provide specific, actionable suggestions

    6. <step>PROVIDE POSITIVE REINFORCEMENT</step>
       - Identify what agent did well
       - Explain why these strengths matter
       - Suggest how to leverage them

    7. <step>MAKE RECOMMENDATIONS</step>
       - Specific recommendations for agent
       - Suggestions for prompt engineers
       - Prioritized by impact (0.0-1.0)
  </enforcement_workflow>

  <important_directives>
    <directive>ALWAYS use the exact XML structure specified in <output_format></directive>
    <directive>Score each phase independently - don't just give overall impression</directive>
    <directive>Provide specific evidence for all claims (quote from response)</directive>
    <directive>Be constructive - identify both strengths and weaknesses</directive>
    <directive>Adapt evaluation based on task complexity and agent type</directive>
    <directive>Flag critical failures that MUST be addressed</directive>
    <directive>Prioritize improvements by expected impact</directive>
    <directive>Never suppress type errors or ignore validation issues</directive>
    <directive>V phase is CRITICAL for factual responses - unverified claims are potential hallucinations</directive>
    <directive>When providing corrections, ALWAYS explain what was changed and why (e.g., "Removed ambiguity in line 5")</directive>
  </important_directives>

  <verification_phase_guidance>
    The V (Verification) phase implements Chain of Verification (CoVe) from Meta AI research.

    <cove_process>
      <step order="1">CLAIM EXTRACTION: Identify factual claims that can be verified independently</step>
      <step order="2">QUESTION GENERATION: Create verification questions answerable WITHOUT seeing the original claim</step>
      <step order="3">INDEPENDENT ANSWERING: Answer questions using tools/knowledge, NOT referencing the original</step>
      <step order="4">CORRECTION: Compare verified answers to original claims, fix inconsistencies</step>
    </cove_process>

    <evaluation_focus>
      - Did agent extract verifiable claims (not opinions or reasoning)?
      - Are verification questions truly independent (no bias from original)?
      - Were answers generated without copying from baseline response?
      - Were corrections applied when verification revealed errors?
      - Were external tools used when available (search, RAG, etc.)?
    </evaluation_focus>

    <when_v_phase_matters_most>
      - Factual claims about external world (dates, names, statistics)
      - Technical specifications or API details
      - Code behavior assertions
      - Historical or scientific facts
    </when_v_phase_matters_most>

    <when_v_phase_matters_less>
      - Pure reasoning or logic problems
      - Opinion or preference requests
      - Creative writing tasks
      - Simple yes/no questions
    </when_v_phase_matters_less>
   </verification_phase_guidance>
  
  <intent_and_workflow>
    <when_to_evaluat_and_improve>
      When user provides instructions with phrases like "Please evaluate these instructions" or "evaluate these instructions":
      
      1. FIRST: Provide complete V-RACEF evaluation with scores
      2. THEN: Provide CORRECTED/IMPROVED version of the instructions if needed
      3. ALWAYS return BOTH evaluation and corrections in a single response
      
      DO NOT just say "I'll evaluate" without actually doing both.
      
      <decision_logic>
        IF instructions contain critical errors (typos, unclear phrasing, missing context):
          → Provide FULL CORRECTED VERSION with fixes applied
          
        IF instructions are incomplete or vague:
          → Provide specific suggestions for completion or clarification
          
        IF instructions are generally good but could be improved:
          → Provide specific refinements to enhance clarity
          
        IF instructions follow best practices already:
          → Acknowledge strengths and suggest optional optimizations
      </decision_logic>
    </when_to_evaluat_and_improve>
  
  </intent_and_workflow>
</vracef_enforcer_agent>
""",
)


class DeltaType(str, Enum):
    ADD_INSTRUCTION = "add_instruction"
    UPDATE_INSTRUCTION = "update_instruction"
    REMOVE_INSTRUCTION = "remove_instruction"
    UPDATE_DESCRIPTION = "update_description"
    UPDATE_EXPECTED_OUTPUT = "update_expected_output"


class VRACEFPhase(str, Enum):
    VERIFICATION = "V"
    REASONING = "R"
    ASSESSMENT = "A"
    CONTEXT = "C"
    EXECUTION = "E"
    FEEDBACK = "F"


class PromptDelta(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    delta_type: DeltaType
    target_phase: VRACEFPhase | None = None
    content: str
    reasoning: str
    expected_impact: float = Field(ge=0.0, le=1.0)
    source_weakness: str | None = None


class TestCase(BaseModel):
    input: str
    expected_output: str
    observations: str | None = None
    id: str = Field(default_factory=lambda: str(uuid4())[:8])


class TestCaseResult(BaseModel):
    test_case_id: str
    passed: bool
    score: float
    actual_output: str | None = None
    error: str | None = None


class RegressionResult(BaseModel):
    passed: bool
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_score: float
    failures: list[TestCaseResult] = Field(default_factory=list[TestCaseResult])


class VRACEFScores(BaseModel):
    verification: float = 0.0
    reasoning: float = 0.0
    assessment: float = 0.0
    context: float = 0.0
    execution: float = 0.0
    feedback: float = 0.0
    overall: float = 0.0

    def weakest_phase(self) -> VRACEFPhase:
        phase_scores = {
            VRACEFPhase.VERIFICATION: self.verification,
            VRACEFPhase.REASONING: self.reasoning,
            VRACEFPhase.ASSESSMENT: self.assessment,
            VRACEFPhase.CONTEXT: self.context,
            VRACEFPhase.EXECUTION: self.execution,
            VRACEFPhase.FEEDBACK: self.feedback,
        }
        return min(phase_scores, key=lambda p: phase_scores[p])


class AgentSnapshot(BaseModel):
    id: str | None
    name: str | None
    description: str | None
    instructions: str
    expected_output: str | None


class OptimizationIteration(BaseModel):
    iteration: int
    timestamp: str
    delta_applied: PromptDelta | None
    before_score: float
    after_score: float
    regression_result: RegressionResult
    accepted: bool
    agent_snapshot: AgentSnapshot


class OptimizationHistory(BaseModel):
    agent_id: str
    agent_name: str
    started_at: str
    iterations: list[OptimizationIteration] = Field(default_factory=list[OptimizationIteration])
    final_score: float = 0.0
    total_deltas_applied: int = 0
    total_deltas_rejected: int = 0


class DeltaGeneratorResponse(BaseModel):
    chain_of_thoughts: ChainOfThoughts
    deltas: list[PromptDelta]


DELTA_GENERATOR_AGENT = Agent(
    id="vracef-delta-generator",
    name="V-RACEF Delta Generator",
    output_schema=DeltaGeneratorResponse,
    instructions=dedent("""
        You generate atomic prompt deltas to improve an agent's instructions based on V-RACEF evaluation feedback.

        INPUT:
        - Current agent instructions, description, and expected_output
        - V-RACEF evaluation with phase scores and improvement priorities
        - Failed test cases (if any)

        OUTPUT:
        Generate 1-3 atomic PromptDelta operations that address the HIGHEST PRIORITY weakness.

        DELTA TYPES:
        - add_instruction: Add a new instruction line to address a gap
        - update_instruction: Modify an existing instruction to be clearer/better
        - remove_instruction: Remove a conflicting or harmful instruction
        - update_description: Improve the agent's description
        - update_expected_output: Clarify what output format is expected

        RULES:
        1. ONE CHANGE AT A TIME - each delta should be atomic and reversible
        2. TARGET WEAKEST PHASE - focus on the phase with lowest score
        3. BE SPECIFIC - don't add vague instructions like "be better"
        4. PRESERVE EXISTING BEHAVIOR - don't break what's already working
        5. INCLUDE REASONING - explain why this delta will help

        PRIORITY ORDER:
        1. Fix critical failures (scores < 0.30)
        2. Address high-impact improvements (expected_impact > 0.7)
        3. Improve medium-impact areas (expected_impact 0.5-0.7)
    """),
)


@dataclass
class VRacefPromptOptimizer:
    evaluator_model: Model
    target_agent: Agent
    test_cases: list[TestCase] = field(default_factory=list[TestCase])
    history: OptimizationHistory | None = None
    min_accuracy_threshold: float = 7.0
    max_iterations: int = 10
    convergence_threshold: float = 0.02
    history_path: Path | None = None
    use_consensus_for_final_validation: bool = True
    consensus_config: ConsensusHooksConfig | None = None

    def __post_init__(self) -> None:
        self.history = OptimizationHistory(
            agent_id=self.target_agent.id or "unknown",
            agent_name=self.target_agent.name or "unknown",
            started_at=datetime.now(UTC).isoformat(),
        )

    def load_dataset(self, jsonl_path: str | Path) -> list[TestCase]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

        test_cases: list[TestCase] = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    test_cases.append(TestCase(**data))

        self.test_cases = test_cases
        logger.info(f"Loaded {len(test_cases)} test cases from {jsonl_path}")
        return test_cases

    def _run_accuracy_eval(
        self, agent: Agent, test_case: TestCase, num_iterations: int = 1
    ) -> tuple[bool, float, str | None]:
        try:
            evaluation = AccuracyEval(
                model=self.evaluator_model,
                agent=agent,
                input=test_case.input,
                expected_output=test_case.expected_output,
                num_iterations=num_iterations,
            )
            result: AccuracyResult | None = evaluation.run(print_results=False)
            if result is None:
                return False, 0.0, "Evaluation returned None"

            passed = result.avg_score >= self.min_accuracy_threshold
            actual_output = None
            if result.results:
                actual_output = result.results[0].output if result.results[0].output else None

            return passed, result.avg_score, actual_output
        except Exception as e:
            logger.exception(f"Error running accuracy eval: {e}")
            return False, 0.0, str(e)

    def run_regression_tests(self, agent: Agent) -> RegressionResult:
        if not self.test_cases:
            return RegressionResult(
                passed=True,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                avg_score=0.0,
            )

        results: list[TestCaseResult] = []
        total_score = 0.0

        for test_case in self.test_cases:
            passed, score, actual_output = self._run_accuracy_eval(agent, test_case)
            total_score += score

            result = TestCaseResult(
                test_case_id=test_case.id,
                passed=passed,
                score=score,
                actual_output=actual_output,
                error=None
                if passed
                else f"Score {score} below threshold {self.min_accuracy_threshold}",
            )
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        failed_results = [r for r in results if not r.passed]

        return RegressionResult(
            passed=len(failed_results) == 0,
            total_cases=len(results),
            passed_cases=passed_count,
            failed_cases=len(failed_results),
            avg_score=total_score / len(results) if results else 0.0,
            failures=failed_results,
        )

    def _run_vracef_evaluation(self, agent: Agent, test_case: TestCase) -> tuple[str, VRACEFScores]:
        agent_response = agent.run(test_case.input)
        agent_output = str(agent_response.content) if agent_response.content else ""

        vracef_input = f"""
        <evaluation_request>
            <agent_name>{agent.name or "Unknown"}</agent_name>
            <agent_instructions>{agent.instructions or ""}</agent_instructions>
            <task_input>{test_case.input}</task_input>
            <expected_output>{test_case.expected_output}</expected_output>
            <actual_output>{agent_output}</actual_output>
            <observations>{test_case.observations or "None provided"}</observations>
        </evaluation_request>
        """

        vracef_response = VRACEF_ENFORCER.run(vracef_input, model=self.evaluator_model)
        vracef_output = str(vracef_response.content) if vracef_response.content else ""

        scores = self._parse_vracef_scores(vracef_output)
        return vracef_output, scores

    def _parse_vracef_scores(self, vracef_output: str) -> VRACEFScores:
        import re

        scores = VRACEFScores()

        def extract_score(phase: str) -> float:
            pattern = rf"<{phase}>.*?<score>([\d.]+)</score>.*?</{phase}>"
            match = re.search(pattern, vracef_output, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return 0.0
            return 0.0

        scores.verification = extract_score("verification")
        scores.reasoning = extract_score("reasoning")
        scores.assessment = extract_score("assessment")
        scores.context = extract_score("context")
        scores.execution = extract_score("execution")
        scores.feedback = extract_score("feedback")

        overall_match = re.search(r"<weighted_score>([\d.]+)</weighted_score>", vracef_output)
        if overall_match:
            try:
                scores.overall = float(overall_match.group(1))
            except ValueError:
                scores.overall = 0.0

        if scores.overall == 0.0:
            phase_scores = [
                scores.verification,
                scores.reasoning,
                scores.assessment,
                scores.context,
                scores.execution,
                scores.feedback,
            ]
            non_zero = [s for s in phase_scores if s > 0]
            scores.overall = sum(non_zero) / len(non_zero) if non_zero else 0.0

        return scores

    def _generate_deltas(
        self,
        agent: Agent,
        vracef_output: str,
        scores: VRACEFScores,
        failed_cases: list[TestCaseResult],
    ) -> list[PromptDelta]:
        delta_input = f"""
        <delta_generation_request>
            <current_agent>
                <name>{agent.name or "Unknown"}</name>
                <description>{agent.description or "None"}</description>
                <instructions>{agent.instructions or "None"}</instructions>
                <expected_output>{agent.expected_output or "None"}</expected_output>
            </current_agent>

            <vracef_evaluation>
                {vracef_output}
            </vracef_evaluation>

            <scores>
                <verification>{scores.verification}</verification>
                <reasoning>{scores.reasoning}</reasoning>
                <assessment>{scores.assessment}</assessment>
                <context>{scores.context}</context>
                <execution>{scores.execution}</execution>
                <feedback>{scores.feedback}</feedback>
                <overall>{scores.overall}</overall>
                <weakest_phase>{scores.weakest_phase().value}</weakest_phase>
            </scores>

            <failed_test_cases>
                {json.dumps([f.model_dump() for f in failed_cases], indent=2) if failed_cases else "None"}
            </failed_test_cases>
        </delta_generation_request>
        """

        response = DELTA_GENERATOR_AGENT.run(delta_input, model=self.evaluator_model)

        if response.content and isinstance(response.content, DeltaGeneratorResponse):
            return response.content.deltas

        return []

    def _apply_delta(self, agent: Agent, delta: PromptDelta) -> Agent:
        updates: dict[str, str] = {}

        raw_instructions = agent.instructions  # type: ignore
        if callable(raw_instructions):  # type: ignore
            current_instructions: str | list[str] = raw_instructions()  # type: ignore
            if isinstance(current_instructions, list):
                current_instructions = "\n".join(current_instructions)  # type: ignore
        elif isinstance(raw_instructions, list):
            current_instructions = "\n".join(raw_instructions)
        elif isinstance(raw_instructions, str):
            current_instructions = raw_instructions
        else:
            current_instructions = ""

        if delta.delta_type == DeltaType.ADD_INSTRUCTION:
            if current_instructions:
                updates["instructions"] = f"{current_instructions}\n{delta.content}"
            else:
                updates["instructions"] = delta.content

        elif delta.delta_type == DeltaType.UPDATE_INSTRUCTION:
            updates["instructions"] = delta.content

        elif delta.delta_type == DeltaType.REMOVE_INSTRUCTION:
            lines = current_instructions.split("\n")  # type: ignore
            updated_lines = [line for line in lines if delta.content.lower() not in line.lower()]  # type: ignore
            updates["instructions"] = "\n".join(updated_lines)  # type: ignore

        elif delta.delta_type == DeltaType.UPDATE_DESCRIPTION:
            updates["description"] = delta.content

        elif delta.delta_type == DeltaType.UPDATE_EXPECTED_OUTPUT:
            updates["expected_output"] = delta.content

        return dataclass_copy(agent, **updates)

    def _get_agent_snapshot(self, agent: Agent) -> AgentSnapshot:
        instructions_str = self._normalize_instructions(agent)

        return AgentSnapshot(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            instructions=instructions_str,
            expected_output=agent.expected_output,
        )

    def _normalize_instructions(self, agent: Agent) -> str:
        raw: str | list[str] | None = agent.instructions  # type: ignore[assignment]
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return "\n".join(raw)
        return ""

    def _run_consensus_validation(
        self, agent: Agent, test_cases: list[TestCase]
    ) -> tuple[bool, float]:
        if not self.use_consensus_for_final_validation:
            return True, 1.0

        if self.consensus_config is None:
            logger.info("No consensus config provided, skipping consensus validation")
            return True, 1.0

        total_score = 0.0
        passed_count = 0

        for test_case in test_cases:
            try:
                consensus_hook = self.consensus_config.pre_hook()
                response = agent.run(test_case.input, pre_hooks=[consensus_hook])

                if response.content:
                    passed_count += 1
                    total_score += 1.0
            except Exception as e:
                logger.warning(f"Consensus validation failed for test case {test_case.id}: {e}")

        avg_score = total_score / len(test_cases) if test_cases else 0.0
        all_passed = passed_count == len(test_cases)

        logger.info(
            f"Consensus validation: {passed_count}/{len(test_cases)} passed, avg score: {avg_score:.2f}"
        )
        return all_passed, avg_score

    def _save_history(self) -> None:
        if self.history_path and self.history:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("w") as f:
                f.write(self.history.model_dump_json(indent=2))
            logger.info(f"Saved optimization history to {self.history_path}")

    def optimize(
        self,
        jsonl_path: str | Path | None = None,
        sample_test_case: TestCase | None = None,
    ) -> Agent:
        if jsonl_path:
            self.load_dataset(jsonl_path)

        if not self.test_cases and not sample_test_case:
            raise ValueError(
                "No test cases provided. Either load from JSONL or provide sample_test_case."
            )

        if sample_test_case and sample_test_case not in self.test_cases:
            self.test_cases.append(sample_test_case)

        current_agent = self.target_agent
        baseline_regression = self.run_regression_tests(current_agent)
        current_score = baseline_regression.avg_score

        logger.info(
            f"Baseline score: {current_score:.2f} ({baseline_regression.passed_cases}/{baseline_regression.total_cases} passed)"
        )

        eval_test_case = sample_test_case or self.test_cases[0]

        for iteration in range(self.max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")

            vracef_output, scores = self._run_vracef_evaluation(current_agent, eval_test_case)
            logger.info(
                f"V-RACEF scores - V:{scores.verification:.2f} R:{scores.reasoning:.2f} A:{scores.assessment:.2f} C:{scores.context:.2f} E:{scores.execution:.2f} F:{scores.feedback:.2f} Overall:{scores.overall:.2f}"
            )

            if scores.overall >= 0.85 and baseline_regression.passed:
                logger.info("Agent already has high V-RACEF compliance. Stopping optimization.")
                break

            deltas = self._generate_deltas(
                current_agent, vracef_output, scores, baseline_regression.failures
            )

            if not deltas:
                logger.info("No deltas generated. Stopping optimization.")
                break

            delta = deltas[0]
            logger.info(f"Applying delta: {delta.delta_type.value} - {delta.reasoning[:100]}...")

            candidate_agent = self._apply_delta(current_agent, delta)
            regression_result = self.run_regression_tests(candidate_agent)

            iteration_record = OptimizationIteration(
                iteration=iteration + 1,
                timestamp=datetime.now(UTC).isoformat(),
                delta_applied=delta,
                before_score=current_score,
                after_score=regression_result.avg_score,
                regression_result=regression_result,
                accepted=False,
                agent_snapshot=self._get_agent_snapshot(candidate_agent),
            )

            should_accept = (
                regression_result.passed and regression_result.avg_score >= current_score
            )

            if should_accept and self.use_consensus_for_final_validation and self.consensus_config:
                logger.info("Running consensus validation before accepting delta...")
                consensus_passed, _ = self._run_consensus_validation(
                    candidate_agent, self.test_cases[:3]
                )
                if not consensus_passed:
                    logger.info("Consensus validation failed. Rejecting delta.")
                    should_accept = False

            if should_accept:
                logger.info(
                    f"Delta accepted! Score: {current_score:.2f} -> {regression_result.avg_score:.2f}"
                )
                current_agent = candidate_agent
                current_score = regression_result.avg_score
                iteration_record.accepted = True
                if self.history:
                    self.history.total_deltas_applied += 1
            else:
                logger.info(
                    f"Delta rejected. Regression: {regression_result.passed}, Score: {regression_result.avg_score:.2f} (was {current_score:.2f})"
                )
                if self.history:
                    self.history.total_deltas_rejected += 1

            if self.history:
                self.history.iterations.append(iteration_record)

            self._save_history()

            if iteration > 0 and self.history and len(self.history.iterations) >= 2:
                prev_score = self.history.iterations[-2].after_score
                if abs(current_score - prev_score) < self.convergence_threshold:
                    logger.info(
                        f"Convergence reached (delta < {self.convergence_threshold}). Stopping."
                    )
                    break

        if self.history:
            self.history.final_score = current_score

        self._save_history()

        logger.info(f"Optimization complete. Final score: {current_score:.2f}")
        if self.history:
            logger.info(
                f"Deltas applied: {self.history.total_deltas_applied}, rejected: {self.history.total_deltas_rejected}"
            )

        return current_agent

    async def aoptimize(
        self,
        jsonl_path: str | Path | None = None,
        sample_test_case: TestCase | None = None,
    ) -> Agent:
        return self.optimize(jsonl_path=jsonl_path, sample_test_case=sample_test_case)


# ============================================================================
# Interactive Dataset Generator
# ============================================================================


class InteractiveDatasetGenerator:
    """Reusable interactive dataset generator for creating test cases with AI assistance.

    Provides a CLI for iteratively building test case datasets with AI-generated
    expected outputs, allowing users to accept, manually correct, or retry with
    specific instructions.

    Usage:
        generator = InteractiveDatasetGenerator(
            agent=YOUR_AGENT,
            model=YOUR_MODEL,
            dataset_path=Path("data/my_dataset.jsonl"),
            dataset_name="My Dataset"
        )
        await generator.run()
    """

    def __init__(
        self,
        agent: Agent,
        model: Model,
        dataset_path: Path,
        dataset_name: str = "Dataset",
        prompt_for_input: str = "Enter input text:",
        prompt_prefix: str = "",
    ):
        self.agent = agent
        self.model = model
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.prompt_for_input = prompt_for_input
        self.prompt_prefix = prompt_prefix

    def load_existing_entries(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self.dataset_path.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(data)
        return entries

    def save_entry(
        self,
        input_text: str,
        expected_output: str | None = None,
        observations: str | None = None,
    ) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        entry: dict[str, Any] = {"input": input_text}

        if expected_output is not None:
            entry["expected_output"] = expected_output

        if observations is not None:
            entry["observations"] = observations

        with self.dataset_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _format_output_pretty(self, output: Any) -> str:
        """Format agent output with pretty JSON for structured data.

        Detects if output is structured (dict, list, dataclass, BaseModel) and
        formats it as pretty JSON. Otherwise returns as-is.

        Args:
            output: The output content to format

        Returns:
            Pretty-formatted string
        """
        if output is None:
            return ""

        output_str = str(output)

        # Try to parse and format as JSON if it looks like structured data
        try:
            # Try to detect if this is a structured representation
            # (dataclass repr, JSON-like, etc.)
            import dataclasses as dc

            # Check if it's a dataclass
            if hasattr(output, "__dataclass_fields__"):
                if dc.is_dataclass(output):
                    # Convert dataclass to dict
                    output_dict = dc.asdict(output)  # type: ignore
                    return json.dumps(output_dict, indent=2, ensure_ascii=False, default=str)

            # Check if it's a BaseModel
            if hasattr(output, "model_dump"):
                output_dict = output.model_dump()
                return json.dumps(output_dict, indent=2, ensure_ascii=False, default=str)

            # Try to parse as JSON string
            if output_str.strip().startswith(("[", "{")):
                parsed = json.loads(output_str)
                return json.dumps(parsed, indent=2, ensure_ascii=False, default=str)

            # Check for dataclass-like repr format
            # e.g., "Classname(field1=value1, field2=value2)"
            if "=" in output_str and not output_str.startswith("<"):
                # Try to evaluate safely
                import ast

                tree = ast.parse(output_str, mode="eval")
                if isinstance(tree.body, (ast.Call, ast.Dict, ast.List)):
                    # This might be a structured repr
                    return output_str  # Return as-is, we'll try eval approach below

        except Exception as e:
            logger.debug(f"Could not format as pretty JSON: {e}")

        return output_str

    async def run_agent_proposal(
        self,
        input_text: str,
        correction_instructions: str | None = None,
        model: Model | None = None,
    ) -> str | None:
        """Run agent with given input and optional correction instructions.

        Args:
            input_text: The input text to send to agent
            correction_instructions: Optional instructions to guide AI
            model: Optional model to override default

        Returns:
            The agent's response content as string, or None if failed
        """
        agent = dataclass_copy(self.agent, model=model or self.model)

        try:
            if correction_instructions:
                full_input = (
                    f"{input_text}\n\n[correction]\n{correction_instructions}\n[/correction]"
                )
            else:
                full_input = input_text

            response = await agent.arun(full_input)
            if response.content:
                content = response.content

                if isinstance(content, (dict, list)):
                    return json.dumps(content, indent=2, ensure_ascii=False, default=str)

                if isinstance(content, type):
                    return str(content)

                if dc.is_dataclass(content) and not isinstance(content, type):
                    try:
                        output_dict = dc.asdict(content)
                        return json.dumps(output_dict, indent=2, ensure_ascii=False, default=str)
                    except Exception:
                        pass

                model_dump_method = getattr(content, "model_dump", None)
                if model_dump_method and callable(model_dump_method):
                    try:
                        output_dict = model_dump_method()
                        return json.dumps(output_dict, indent=2, ensure_ascii=False, default=str)
                    except Exception:
                        pass

                return str(content)
            return None
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return None

    async def run(self, console: Console | None = None) -> None:
        if console is None:
            console = Console()

        entries = self.load_existing_entries()

        console.print(
            Panel.fit(
                f"[bold cyan]{self.dataset_name}[/bold cyan]\n"
                "Create test cases with AI assistance.\n\n"
                f"Dataset: {self.dataset_path}\n"
                f"Existing entries: {len(entries)}\n\n"
                "Commands:\n"
                "  [green]c[/green] - Create new entry\n"
                "  [green]l[/green] - List existing entries\n"
                "  [green]x[/green] - Exit",
                title="Interactive Dataset Generator",
            )
        )

        from rich.prompt import Prompt as RichPrompt

        while True:
            action = RichPrompt.ask(
                "\n[bold]Action[/bold]",
                choices=["c", "l", "x"],
                default="c",
            )

            if action == "x":
                console.print("[dim]Goodbye![/dim]")
                break

            if action == "l":
                if not entries:
                    console.print("[dim]No entries yet.[/dim]")
                else:
                    console.print(f"\n[bold]Dataset entries: {len(entries)}[/bold]")
                    for idx, entry in enumerate(entries, 1):
                        preview = entry.get("input", "")[:80]
                        if len(entry.get("input", "")) > 80:
                            preview += "..."
                        obs = entry.get("observations", "none")
                        console.print(f"  {idx}. [dim]{obs}[/dim] | {preview}")
                continue

            # Create new entry
            console.print("\n[bold cyan]Creating new entry[/bold cyan]")
            console.print(f"[dim]{self.prompt_prefix}[/dim]")
            console.print(f"[dim]{self.prompt_for_input}[/dim]")
            console.print("[dim]Paste content, then press Enter to submit:[/dim]")

            lines: list[str] = []
            empty_count = 0
            while empty_count < 1:
                line = input()
                if not line:
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)

            input_text = "\n".join(lines).strip()
            if not input_text:
                console.print("[yellow]Empty input, skipping.[/yellow]")
                continue

            console.print("\n[dim]Running agent...[/dim]")
            proposed = await self.run_agent_proposal(input_text)

            if proposed is None:
                console.print("[red]Agent failed to generate output.[/red]")
                continue

            console.print("\n[bold]Proposed Output:[/bold]")
            console.print(proposed)

            while True:
                action = RichPrompt.ask(
                    "[bold]Accept, Correct manually, Retry with instructions, or Skip?[/bold]",
                    choices=["a", "c", "r", "s"],
                    default="a",
                )

                if action == "s":
                    console.print("[dim]Skipped.[/dim]")
                    break

                if action == "a":
                    observations = RichPrompt.ask(
                        "[dim]Optional observations (press Enter to skip)[/dim]",
                        default="",
                    )

                    self.save_entry(input_text, proposed, observations if observations else None)
                    console.print(f"[green]Saved to {self.dataset_path}[/green]")

                    entries = self.load_existing_entries()
                    break

                if action == "c":
                    console.print(
                        "\n[yellow]Enter corrected output (or press Enter to accept proposed):[/yellow]"
                    )
                    console.print("[dim]Paste output, then press Enter to submit:[/dim]")

                    lines: list[str] = []
                    empty_count = 0
                    while empty_count < 1:
                        line = input()
                        if not line:
                            empty_count += 1
                        else:
                            empty_count = 0
                            lines.append(line)

                    corrected = "\n".join(lines) if lines else proposed

                    observations = RichPrompt.ask(
                        "[dim]Optional observations (press Enter to skip)[/dim]",
                        default="",
                    )

                    self.save_entry(input_text, corrected, observations if observations else None)
                    console.print(f"[green]Saved to {self.dataset_path}[/green]")

                    entries = self.load_existing_entries()
                    break

                if action == "r":
                    correction_instructions = RichPrompt.ask(
                        "[bold]Enter correction instructions for AI:[/bold]",
                        default="",
                    )

                    if not correction_instructions:
                        console.print("[yellow]No instructions provided, skipping retry.[/yellow]")
                        continue

                    console.print("\n[dim]Regenerating with instructions...[/dim]")
                    proposed = await self.run_agent_proposal(input_text, correction_instructions)

                    if proposed is None:
                        console.print("[red]Failed to regenerate, keeping previous output.[/red]")
                        continue

                    console.print("\n[bold]Proposed Output:[/bold]")
                    console.print(proposed)
                    continue
