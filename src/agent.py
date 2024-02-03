from utils import get_llama_ip


class Agent:
    def __init__(self, role, mtype, ans_parser, qtype, nums=1, temperature=1, top_p=1):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.question = None
        self.llm_ip = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.nums = nums
        self.temperature = temperature
        self.top_p = top_p
        if mtype == "gpt-35-turbo":
            self.model = "gpt-35-turbo"
        elif mtype == "gpt-4":
            self.model = "gpt-4"
        elif mtype == "gpt-4-1106-Preview":
            self.model = "gpt-4-1106-Preview"
        elif mtype == "gpt-35-turbo-1106":
            self.model = "gpt-35-turbo-1106"
        else:
            self.model = mtype
            self.llm_ip = get_llama_ip()

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def preprocess(self, question):
        self.question = question
        contexts = self.get_context()
        return contexts

    def postprocess(self, completion, question):
        self.reply = completion
        self.answer, _ = self.ans_parser(self.reply, question)

    def get_context(self):
        if self.qtype == "code_completion":
            # sys_prompt = prompt_lib.CODE_COMPLETION_SYSTEM_PROMPT # TODO
            sys_prompt = ""
        elif self.qtype == "mmlu":
            sys_prompt = ""
        elif self.qtype == "math":
            # sys_prompt = prompt_lib.MATH_TASK_SYSTEM_PROMPT # TODO
            sys_prompt = ""
        elif self.qtype == "chess":
            sys_prompt = ""
        elif self.qtype == "gsm":
            sys_prompt = ""
        elif self.qtype == "istask":
            sys_prompt = ""
        elif self.qtype == "sstask":
            sys_prompt = ""
        else:
            raise NotImplementedError("Error init question type")
        contexts = [{"role": "system", "content": sys_prompt}]
        return contexts
