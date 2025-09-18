import dspy
from pydantic import BaseModel, Field
from typing import Any
from aicodetools import CodeInstance

from .. import dspy_program
from .super_utils import create_runtime, get_runtime_tools


class FinishResponse(BaseModel):
    success: bool = Field(..., description="Indicates whether the task was completed successfully.")
    confidence: float = Field(..., description="Confidence level of the task completion, ranging from 0.0 to 1.0.")
    reasoning: str = Field(..., description="Explanation of the reasoning or process followed for the task.")
    structured_output: Any = Field(
        ...,
        description="The main output or result of the task"
    )
    summary: str = Field(..., description="Summary of the steps taken to complete the task.")


class SuperReactAgent(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, max_iters=100):
        super().__init__()
        self.max_iters = max_iters
        
    def get_fresh_tools(self):
        return get_runtime_tools()
    
    def forward(self, query, **kwargs):
        tools = self.get_fresh_tools()

        # Extract additional fields from kwargs if they exist in the example
        github_repo = kwargs.get('github_repo', '')
        git_commit = kwargs.get('git_commit', '')
        instance_id = kwargs.get('instance_id', '')

        react = dspy.ReAct(
            "query, github_repo, git_commit -> result: FinishResponse",
            tools=tools,
            max_iters=self.max_iters
        )

        result = react(
            query=query,
            github_repo=github_repo,
            git_commit=git_commit
        )

        return result