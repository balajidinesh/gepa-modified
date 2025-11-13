import dspy
from pydantic import BaseModel, Field
from typing import Any , List, Dict

from .. import dspy_program
import uuid

from dspy import Tool
from ..dspy_program import LangProBeDSPyMetaProgram

class RCBResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""
    
    instance_id: str = dspy.InputField()
    paper_id: str = dspy.InputField()
    snippet_name: str = dspy.InputField()
    masked_file: str = dspy.InputField()
    context_files: list = dspy.InputField()
    paper: str = dspy.InputField()

    result: str = dspy.OutputField()

class RCB(LangProBeDSPyMetaProgram,dspy.Module):
    def __init__(self):
        
        super().__init__()
        self.model = dspy.Predict(RCBResponse)
    
    def forward(self, **kwargs):
        instance_id = kwargs.get('instance_id', '')
        paper_id = kwargs.get('paper_id', '')
        snippet_name = kwargs.get('snippet_name', '')
        masked_file = kwargs.get('masked_file', '')
        context_files = kwargs.get('context_files', [])
        paper = kwargs.get('paper', '')

        # Ensure context_files is a list of dicts
        if isinstance(context_files, dict):
            context_files = [context_files]
        elif not isinstance(context_files, list):
            context_files = []

        result = self.model(
            instance_id=instance_id,
            paper_id=paper_id,
            snippet_name=snippet_name,
            masked_file=masked_file,
            context_files=context_files,
            paper=paper
        )

        return result