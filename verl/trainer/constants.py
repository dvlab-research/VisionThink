GENERAL_QA_SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{}."""

SHORT_SYSTEM_PROMPT="""You are a helpful assistant."""

TOOL_AGENT_SYSTEM_PROMPT="""You are a helpful assistant.

# Tools

You may call the function tool shown below to assist with the user query.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "resize_image", "name": "resize_image", "description": "Resize the image resolution.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `resize`: Double the resolution of the current image. You should only use this tool if you are unable to obtain the critical information needed to answer the question from the current resolution.", "enum": ["resize"], "type": "string"}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
For each function call, return a json object with the function name and the corresponding argument within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

SYSTEM_PROMPT_MAP={
    "short": SHORT_SYSTEM_PROMPT,
    "general_qa": GENERAL_QA_SYSTEM_PROMPT,
    "tool_agent": TOOL_AGENT_SYSTEM_PROMPT,
}

TOOL_CALL_MULTI_TRUN_PROMPT="Please carefully analyze the content returned from the image resize tool in combination with the original question and image from the user, continue your reasoning process inside <think> and </think> and then write your final answer inside <answer> and </answer>."

ERROR_INFO_MULTI_TURN_PROMPT="Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."