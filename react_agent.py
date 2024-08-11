import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain import hub

from langchain.agents import (
    Tool,
    agent_iterator,
    AgentExecutor,
    initialize_agent,
    create_react_agent,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(find_dotenv(), override=True)
os.environ.get("OPENAI_API")

llm = ChatOpenAI(model_name="------------", temperature=0)

# Define prompt template
template = """ 
Answer the following questions as best as you can
Questions: {q}
"""
prompt_template = PromptTemplate.from_template(template)

# Load prompt from hub
prompt = hub.pull("----------")

# Print prompt details
print(type(prompt))
print(prompt.input_variables)
print(prompt.template)

# Define tools
python_repl = PythonAstREPLTool()
python_repl_tool = Tool(
    name="python Repl",
    func=python_repl.run,
    description="Useful when you need to use Python to answer questions. Please input Python code.",
)

api_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Useful for looking up information in Wikipedia.",
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for performing web searches.",
)

tools = [python_repl_tool, wikipedia_tool, duckduckgo_tool]

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

# Define question and invoke agent
Question = "generate the first 20 numbers in fibonacci series "
prompt_text = prompt_template.format(q=Question)
output = agent_executor.invoke({"input": prompt_text})

print(output)
