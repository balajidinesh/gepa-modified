import dspy
from pydantic import BaseModel, Field
from typing import Any

from .. import dspy_program
import uuid
from dspy import Tool

from ..dspy_program import LangProBeDSPyMetaProgram

class RCBResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""
    query : str = dspy.InputField()
    github_repo : str = dspy.InputField()
    git_commit : str = dspy.InputField()
    result : str = dspy.OutputField()

class RCB(LangProBeDSPyMetaProgram,dspy.Module):
    def __init__(self):
        
        super().__init__()
        self.model = dspy.Predict(RCBResponse)
        

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

        return result