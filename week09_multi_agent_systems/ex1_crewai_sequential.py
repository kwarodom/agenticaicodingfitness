#creaai_test.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
load_dotenv()

qwen_llm = LLM(
    model="openrouter/qwen/qwen3.5-flash-02-23",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

zai_llm = LLM(
    model="openai/glm-5",
    base_url="https://api.z.ai/api/paas/v4/",
    api_key=os.getenv("ZAI_API_KEY"),
)

from crewai_tools import SerperDevTool
search_tool = SerperDevTool()

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find and summarize the latest trends in building energy"
    " optimization and smart HVAC systems",
    backstory="You are an expert energy analyst at AltoTech who"
    " tracks global trends in building efficiency, IoT"
    " sensors, and AI-driven HVAC optimization.",
    verbose=True,
    llm=zai_llm,
    allow_delegation=False,
    tools=[search_tool],
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write an engaging, informative blog post based on the"
    " research findings",
    backstory="You write for AltoTech's engineering blog."
    " Your audience is building managers and facility"
    " engineers. You make complex topics accessible.",
    verbose=True,
    llm=zai_llm,
    allow_delegation=False,
)

editor = Agent(
    role="Chief Editor",
    goal="Review and polish the article for clarity, accuracy,"
    " and AltoTech brand voice",
    backstory="You are the senior editor at AltoTech, ensuring"
    " all published content meets high standards of"
    " technical accuracy and readability.",
    verbose=True,
    llm=zai_llm,
    allow_delegation=False,
)

research_task = Task(
    description="Research the latest trends in AI-driven building"
    " energy optimization. Focus on: (1) new HVAC"
    " control strategies, (2) IoT sensor advances,"
    " (3) real-world case studies with measurable"
    " energy savings. Summarize findings in 5 bullet"
    " points with sources.",
    expected_output="A structured research brief with 5 key"
    " findings, each with a source citation.",
    agent=researcher,
)
writing_task = Task(
    description="Using the research brief, write a 600-word blog"
    " post titled This Week in Building Energy AI."
    " Include an introduction, 3 main sections, and"
    " a conclusion with a call-to-action for AltoTech.",
    expected_output="A complete 600-word blog post in markdown.",
    agent=writer,
)
editing_task = Task(
    description="Review the blog post for: (1) technical accuracy,"
    " (2) grammar and clarity, (3) AltoTech brand"
    " voice consistency, (4) engaging headline. Return"
    " the final polished version.",
    expected_output="The final, publication-ready blog post.",
    agent=editor,
    output_file='ex1_withtools_new_post.md'  # The final blog post will be saved here
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
print("\n" + "="*60)
print("FINAL OUTPUT:")
print("="*60)
print(result)

# output_path = os.path.join(os.path.dirname(__file__), "ex1_output.md")
# with open(output_path, "w") as f:
#     f.write(str(result))
# print(f"\nExported to {output_path}")
