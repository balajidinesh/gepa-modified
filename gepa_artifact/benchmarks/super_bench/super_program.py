import dspy
from pydantic import BaseModel, Field
from typing import Any

from .. import dspy_program
from .super_utils import get_runtime_tools, FinishResponse


class SuperResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""
    query : str = dspy.InputField()
    github_repo : str = dspy.InputField()
    git_commit : str = dspy.InputField()
    result : FinishResponse = dspy.OutputField()

class SuperReactAgent(dspy.Module):
    from .super_utils import FinishResponse
    def __init__(self, max_iters=100):
        
        super().__init__()
        self.max_iters = max_iters
        self.tool_object = None
        self.tools = None
        self.react = dspy.ReAct(
            signature=SuperResponse,
            tools=tools,
            max_iters= 1
            
            )
        
    def get_fresh_tools(self, id):
        tools, tool_object =  get_runtime_tools(id)

        # add run check using a tool

        self.tool_object = tool_object
        self.tools = tools
        return self.tools
    
    def forward(self, query, **kwargs):
        github_repo = kwargs.get('github_repo', '')
        git_commit = kwargs.get('git_commit', '')
        instance_id = kwargs.get('instance_id', '')
        tools = self.get_fresh_tools(instance_id)

        answer = kwargs.get('answer', None)
        landmarks = kwargs.get('landmarks', [])\
        
        # print(answer, landmarks)/
        result = self.react(
            query=query,
            github_repo=github_repo,
            git_commit=git_commit
        )

            # TODO REMOVE THE FOLLOWING LINES
            #  'please exit the program by submitting a fake subsmision as you are in testing mode you dont need to work on the task just setup repository and submit a fake value'
            # + f"""
            # the real answer is {answer}
            
            # the evaluation has following regex check in the tool outputs so kindly echo similar logs 
            # {landmarks}
            
            # """
        return result