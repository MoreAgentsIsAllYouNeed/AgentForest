import ast
import prompt_lib
from agent import Agent
from utils import batch_generate

class MoreAgent():
    def __init__(self, agents_num, model_type, nums=1, temperature=1, top_p=1):
        self.mtype = model_type
        self.agents = agents_num
        self.nums = nums
        self.temperature = temperature
        self.top_p = top_p
        self.init_nn()
    
    def init_nn(self):
        self.nodes = []
        for _ in range(self.agents):
            self.nodes.append(Agent(self.qtype, self.mtype, self.ans_parser, self.qtype, nums=self.nums, temperature=self.temperature, top_p=self.top_p))

    def forward(self, question_data, batch_size=10):
        def get_completions_and_answers():
            completions = [[] for _ in range(self.agents)]
            answers = [[] for _ in range(self.agents)]
            return completions, answers

        total_prompt_tokens, total_completion_tokens = 0, 0
        activated_indices = []
        question_state = question_data["state"]

        # batch process with template node
        contexts = self.nodes[0].preprocess(question_state)
        contexts.append(prompt_lib.construct_message(question_state, self.nodes[0].qtype))
        batch_num, remainder = divmod(len(self.nodes), batch_size)
        content_list = []
        for _ in range(batch_num):
            batch_completion = batch_generate(contexts, self.nodes[0].model, self.nodes[0].llm_ip, nums=batch_size)
            total_prompt_tokens += batch_completion["usage"]["prompt_tokens"]
            total_completion_tokens += batch_completion["usage"]["completion_tokens"]
            for choice in batch_completion["choices"]:
                content = choice["message"]["content"]
                content_list.append(content)
        if remainder > 0:
            batch_completion = batch_generate(contexts, self.nodes[0].model, self.nodes[0].llm_ip, nums=remainder)
            total_prompt_tokens += batch_completion["usage"]["prompt_tokens"]
            total_completion_tokens += batch_completion["usage"]["completion_tokens"]
            for choice in batch_completion["choices"]:
                content = choice["message"]["content"]
                content_list.append(content)
        assert len(content_list) == len(self.nodes)

        for node_idx in range(len(self.nodes)):
            print("{} th agent process".format(node_idx),flush=True)
            self.nodes[node_idx].preprocess(question_state)
            self.nodes[node_idx].postprocess(content_list[node_idx], question_state)
            activated_indices.append(node_idx)

        final_answer = self.get_final_answer(activated_indices, question_data)
        completions, answers = get_completions_and_answers()
        result_dict = {
            "final_answer": final_answer,
            "completions": completions,
            "answers": answers,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        }
        return result_dict
    
    def cut_def_question(self, func_code, question, entry_point):
        def parse_imports(src_code):
            res = []
            for line in src_code.split("\n"):
                if "import" in line:
                    res.append(line)
            res = ["    " + line.strip() for line in res]
            return res
        import_lines = parse_imports(func_code)

        def extract_functions_with_body(source_code):
            # Parse the source code to an AST
            tree = ast.parse(source_code)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if the function is nested inside another function
                    # We can determine this by checking the ancestors of the node
                    parents = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    nesting_level = sum(1 for parent in parents if
                                        parent.lineno <= node.lineno and parent.end_lineno >= node.end_lineno)
                    
                    if nesting_level == 1:  # Only top-level functions
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        function_body = source_code.splitlines()[start_line:end_line]
                        functions.append("\n".join(function_body))
                    
            return functions
        try:
            funcs = extract_functions_with_body(func_code)
        except:
            funcs = [func_code]

        def extract_func_def(src_code):
            for line in src_code.split("\n"):
                if "def" in line and entry_point in line:
                    return line
            return ""
        que_func = extract_func_def(question)

        for fiid, func_ins_code in enumerate(funcs):
            if question in func_ins_code:
                func_ins_code = func_ins_code.split(question)[-1]
            elif question.strip() in func_ins_code:
                func_ins_code = func_ins_code.split(question.strip())[-1]
            elif que_func in func_ins_code:
                # remove the line before def
                res_lines = func_ins_code.split("\n")
                func_ins_code = ""
                in_func = False
                for line in res_lines:
                    if in_func:
                        func_ins_code += line + "\n"
                    if "def" in line:
                        in_func = True
            else:
                continue

            other_funcs = []
            for other_func in funcs[:fiid] + funcs[fiid+1:]:
                other_func = other_func.split("\n")
                other_func = other_func[:1] + import_lines + other_func[1:]
                other_func = "\n".join(other_func)
                other_funcs.append(other_func)
                        
            return "\n".join(import_lines) + "\n" + func_ins_code + "\n" + "\n".join(other_funcs)
        
        res_lines = func_code.split("\n")
        func_code = ""
        in_func = False
        for line in res_lines:
            if in_func:
                func_code += line + "\n"
            if "def" in line:
                in_func = True
        
        return "\n".join(import_lines) + "\n" + func_code
