#%%
import os
from IPython.display import Markdown
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()


#%% Tools
# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

#%% define model
# https://console.groq.com/docs/models
MODEL = "mixtral-8x7b-32768"

#%% define other LLM
llm=ChatGroq(temperature=0,
             model_name=MODEL,
             api_key=os.environ["GROQ_API"])

# %% Agents
match_researcher = Agent(
    role="Match Researcher",
    goal="Find the match result history of previous matches between {country1} and {country2} in men's soccer. Consider also the recent trends and statistics of the countries' performance.",
    backstory="""
    You are a researcher who specializes in finding historical information about previous matches between two countries in men's soccer. You consider direct matches, but also the recent performance of the teams. Your work is the basis for Match Result Predictor to predict the outcome of the match.""",
    allow_delegation=False,
    llm=llm,
    max_iter = 2,
    tools=[search_tool, scrape_tool],
	verbose=True
)

match_predictor = Agent(
    role="Match Result Predictor",
    goal="Predict the outcome of the match between {country1} and {country2} in men's soccer",
    backstory="""You are a match predictor who predicts the outcome of the match between two countries. You base your prediction on the results of previous matches and recent trends and statistics of the countries' performance.""",
    allow_delegation=False,
    context=["researcher_direct_match", "researcher_recent_trend"],
    llm=llm,
    max_iter = 2,
    verbose=True
)

# %% Tasks
analyze_matches = Task(
    description=(
        "1. Find historical soccer results of previous matches between {country1} and {country2}.\n"
    ),
    expected_output="A list of previous matches between the two countries.",
    agent=match_researcher,
)

class OutputFormat(BaseModel):
    country1: str
    country2: str
    prediction: str

predict_match_outcome = Task(
    description=(
        "Predict the outcome of the match between {country1} and {country2} based on direct match results and recent trends and statistics.\n" 
        "Only provide one most likely result.\n"),
    expected_output="A markdown document on the teams and the match result prediction.",
    agent=match_predictor,
    output_format="markdown",
    output_format_model=OutputFormat,
    output_format_description=(
        "The output format is a markdown file with the following structure:\n"
        "1. {country1}\n"
        "2. {country2}\n"
        "3. Match Prediction\n"
    ),
    output_file = "match_prediction.md"
    )
#%% Crew
crew = Crew(
    agents=[match_researcher, match_predictor],
    tasks=[analyze_matches, predict_match_outcome],
    verbose=2
)

result = crew.kickoff(inputs={"country1": "Germany", "country2": "Scotland"})

# %%
Markdown(result)

# %%
