
MMLU_QUESTION = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} "

MATH_TASK_SYSTEM_PROMPT = "Imagine you are an expert skilled in solving mathematical problems and are confident in your answer and often persuades other agents to believe in you. Please keep this in mind."

CODE_COMPLETION_SYSTEM_PROMPT = f"You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer! You can only respond with comments and actual code, no free-flowing text (unless in a comment)." # from https://github.com/getcursor/eval.git

interaction_prompt = {
    "mmlu":{
        "question": "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents: ",
            "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
        ],
        "reflection": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response.",
    },
    "math":{
        "question": "Here is a math problem written in LaTeX:{}\nPlease carefully consider it and explain your reasoning. Put your answer in the form \\boxed{{answer}}, at the end of your response.",
        "debate": [
            "These are the solutions to the problem from other agents:",
            "\n\nUsing the reasoning from other agents as additional information and referring to your historical answers, can you give an updated answer? Put your answer in the form \\boxed{{answer}}, at the end of your response."
        ],
        "reflection": "Can you double check that your answer is correct? Please reiterate your answer, with your answer in the form \\boxed{{answer}}, at the end of your response.",
    },
    "chess":{
        "question": "Given the chess game \"{}\", give one valid destination square for the chess piece at \"{}\". Give a one line explanation of why your destination square is a valid move. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]. ",
        "debate": [
            "Here are destination square suggestions from other agents:",
            "\n\nCan you double check that your destination square is a valid move? Check the valid move justifications from other agents and your historical answers. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."
        ],
        "reflection": "Can you double check that your destination square is a valid move? Check the valid move justifications from your historical answers. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8].",
    },
    "gsm":{
        "question" : "Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. "
    }
}


def construct_message(question, qtype):
    if qtype == "code_completion":
        qtemplate = "```python\n{}\n```"
        qtemplate = '''You must complete the python function I give you.
Be sure to use the same indentation I specified. Furthermore, you may only write your response in code/comments.
[function impl]:
{}\nOnce more, please follow the template by repeating the original function, then writing the completion.'''.format(qtemplate.format(question))
        return {"role": "user", "content": qtemplate}
    elif qtype == "mmlu":
        return {"role": "user", "content": question}
    elif qtype == "math":
        return {"role": "user", "content": question}
    elif qtype == "chess":
        return {"role": "user", "content": question}
    elif qtype == "gsm":
        return {"role": "user", "content": question}
    elif qtype == "istask":
        return {"role": "user", "content": question}
    elif qtype == "sstask":
        return {"role": "user", "content": question}
    else:
        raise NotImplementedError
