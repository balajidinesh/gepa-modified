import dspy
from pydantic import BaseModel, Field
from typing import Any

from .. import dspy_program
import uuid
from dspy import Tool

from .sab_utils import get_runtime_tools, close_runtime_tools, read_file_from_container

from ..dspy_program import LangProBeDSPyMetaProgram

class SABResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""
    # instance_id: str = dspy.InputField()
    task_inst : str = dspy.InputField()
    dataset_folder_tree : str = dspy.InputField()
    dataset_preview : str = dspy.InputField()
    add_inst :str = dspy.InputField()
    absolute_code_file_path: str = dspy.OutputField(
        description="The absolute file path to the generated code for the given task"
    )

class SABReactAgent(LangProBeDSPyMetaProgram,dspy.Module):
    def __init__(self, max_iters=50):
        
        super().__init__()
        self.max_iters = max_iters
        self.tool_object = None
        self.tools = self.get_fresh_tools(str(uuid.uuid4()))

        self.react = dspy.ReAct(
            signature=SABResponse,
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

    
    def forward(self, task_inst, **kwargs):
        dataset_folder_tree = kwargs.get('dataset_folder_tree', '')
        dataset_preview = kwargs.get('dataset_preview', '')
        instance_id = kwargs.get('instance_id', '')
        add_inst = kwargs.get('add_inst', '')
        tools = self.get_fresh_tools(instance_id)

        rt = self.tool_object

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        react_tools = self.react.tools
        # tools = {**react_tools, **{tool.name: tool for tool in tools}}
        for tool in tools : 
            react_tools[tool.name] = tool 
    
        result = self.react(
            task_inst=task_inst,
            dataset_preview=dataset_preview,
            dataset_folder_tree=dataset_folder_tree,
            add_inst=add_inst
        )

        final_path = result.absolute_code_file_path if hasattr(result, 'absolute_code_file_path') else result
        print(f"submitted path : {final_path}")

        generated_code = ''
        if final_path :
            generated_code = read_file_from_container(client=rt, path=final_path)


        self.close_tools(instance_id)

        return dspy.Prediction(generated_code=generated_code, result=result, final_path=final_path)