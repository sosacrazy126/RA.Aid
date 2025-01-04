"""Prompts package for RA.Aid."""

# System prompts
WELCOME_MESSAGE = "👋 Welcome to RA.Aid! I'm your AI Development Assistant. How can I help you today?"

# Research prompts
RESEARCH_PROMPT = """Research Task: {base_task}

{research_only_note}

{expert_section}
{human_section}
{web_research_section}

Context from memory:
Key Facts: {key_facts}
Code Snippets: {code_snippets}
Related Files: {related_files}
"""

RESEARCH_ONLY_PROMPT = """Research Task: {base_task}

{expert_section}
{human_section}
{web_research_section}

Context from memory:
Key Facts: {key_facts}
Code Snippets: {code_snippets}
Related Files: {related_files}
"""

# Web research prompts
WEB_RESEARCH_PROMPT = """Web Research Task: {base_task}

{expert_section}
{human_section}

Context from memory:
Key Facts: {key_facts}
Code Snippets: {code_snippets}
Related Files: {related_files}
"""

# Implementation prompts
IMPLEMENTATION_PROMPT = """Implementation Task: {base_task}

{expert_section}
{human_section}
{web_research_section}

Context from memory:
Key Facts: {key_facts}
Code Snippets: {code_snippets}
Related Files: {related_files}
Task Details:
Current Task: {task}
All Tasks: {tasks}
Plan: {plan}
Work Log: {work_log}
"""

# Planning prompts
PLANNING_PROMPT = """Planning Task: {base_task}

{expert_section}
{human_section}
{web_research_section}

Context from memory:
Key Facts: {key_facts}
Code Snippets: {code_snippets}
Related Files: {related_files}
Research Notes: {research_notes}
"""

# Section prompts
EXPERT_PROMPT_SECTION_IMPLEMENTATION = """Expert Mode: Enabled
- Provide detailed technical analysis
- Consider advanced implementation patterns
- Include performance considerations
"""

HUMAN_PROMPT_SECTION_IMPLEMENTATION = """Human Interaction: Enabled
- Ask for clarification when needed
- Explain technical decisions
- Provide progress updates
"""

EXPERT_PROMPT_SECTION_RESEARCH = """Expert Mode: Enabled
- Deep technical analysis
- Consider multiple approaches
- Evaluate trade-offs
"""

HUMAN_PROMPT_SECTION_RESEARCH = """Human Interaction: Enabled
- Ask clarifying questions
- Explain findings clearly
- Seek feedback on direction
"""

EXPERT_PROMPT_SECTION_PLANNING = """Expert Mode: Enabled
- Detailed task breakdown
- Consider dependencies
- Risk assessment
"""

HUMAN_PROMPT_SECTION_PLANNING = """Human Interaction: Enabled
- Confirm priorities
- Clarify requirements
- Get feedback on plan
"""

# Web research sections
WEB_RESEARCH_PROMPT_SECTION_RESEARCH = """Web Research: Enabled
- Search online documentation
- Find relevant examples
- Check best practices
"""

WEB_RESEARCH_PROMPT_SECTION_CHAT = """Web Research: Enabled
- Find relevant information
- Check documentation
- Look for examples
"""

WEB_RESEARCH_PROMPT_SECTION_PLANNING = """Web Research: Enabled
- Research similar projects
- Find implementation guides
- Check common pitfalls
"""

# Chat prompts
CHAT_PROMPT = """Chat Task: {task}

{expert_section}
{human_section}
{web_research_section}

Context from memory:
Key Facts: {key_facts}
Code Snippets: {code_snippets}
Related Files: {related_files}
""" 