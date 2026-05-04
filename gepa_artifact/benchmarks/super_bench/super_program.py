import dspy
from pydantic import BaseModel, Field
from typing import Any

from .. import dspy_program
import uuid
from dspy import Tool

from .super_utils import get_runtime_tools, FinishResponse, close_runtime_tools

from ..dspy_program import LangProBeDSPyMetaProgram
class SuperResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""
    query : str = dspy.InputField()
    github_repo : str = dspy.InputField()
    git_commit : str = dspy.InputField()
    result : FinishResponse = dspy.OutputField()

class SuperReactAgent(LangProBeDSPyMetaProgram,dspy.Module):
    def __init__(self, max_iters=100):
        
        super().__init__()
        self.max_iters = max_iters
        self.tool_object = None
        self.tools = self.get_fresh_tools(str(uuid.uuid4()))

        
        self.react = dspy.ReAct(
            signature=SuperResponse,
            tools=self.tools,
            max_iters= self.max_iters
            )
        

    def get_fresh_tools(self, id):
        tools, tool_object =  get_runtime_tools(id)

        # add run check using a tool

        self.tool_object = tool_object
        self.tools = tools
        return self.tools
    

    def close_tools(self, id) : 
        rc = close_runtime_tools(id)
        return rc 

    
    def forward(self, query, **kwargs):
        github_repo = kwargs.get('github_repo', '')
        git_commit = kwargs.get('git_commit', '')
        instance_id = kwargs.get('instance_id', '')
        tools = self.get_fresh_tools(instance_id)

        # reason to give dummy tools prior is to make sure the dspy prompts are well compiled etc as dspy dumps the docstrings to prompt
        # n the tools are independent for each task : handled it externally by reference it shouldn't effect any performance
        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        react_tools = self.react.tools
        # tools = {**react_tools, **{tool.name: tool for tool in tools}}
        for tool in tools : 
            react_tools[tool.name] = tool 
    
        result = self.react(
            query=query,
            github_repo=github_repo,
            git_commit=git_commit
        )

        self.close_tools(instance_id)
        # self.close_tools(uid created)

        return result

