import utils
from more_agent import MoreAgent
from human_eval.data import read_problems


class CodeCompletion(MoreAgent):
    def __init__(self, agents_nums, model_type, nums=1, temperature=1, top_p=1):
        self.qtype = "code_completion"
        self.ans_parser = utils.parse_code_completion
        super().__init__(agents_nums, model_type, nums, temperature, top_p)

    def get_question_datas(self):
        raw_problems = read_problems()
        question_datas = []
        for k, v in raw_problems.items():
            question_data = {
                "task_id": k,
                "state": v["prompt"],
                "entry_point": v["entry_point"],
            }
            question_datas.append(question_data)
        return question_datas
    
    def get_final_answer(self, idxs, question_data):
        entry_point = question_data["entry_point"]
        question_state = question_data["state"]
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        python_codes = []
        passed_codes = []
        for cand in candidates:
            result = utils.check_function_result(cand)
            if result["passed"]:
                passed_codes.append(cand)
            python_codes.append(cand)
        if len(python_codes) == 0:
            exit("with no python code generated")
        
        if len(passed_codes) != 0:
            passed_answers = []
            for passed_code in passed_codes:
                passed_code = self.cut_def_question(passed_code, question_state, entry_point)
                passed_answers.append(passed_code)
            _, most_similar_answer, max_score = utils.most_similar_code(passed_answers)
            print("PASSED: Most Similar Answer: {}\nMax bleu score: {}".format(most_similar_answer, max_score))
            return most_similar_answer
        
        pred_answers = []
        for python_code in python_codes:
            python_code = self.cut_def_question(python_code, question_state, entry_point)
            pred_answers.append(python_code)
        _, most_similar_answer, max_score = utils.most_similar_code(pred_answers)
        print("Most Similar Answer: {}\nMax bleu score: {}".format(most_similar_answer, max_score))
        return most_similar_answer

    def evaluation(self, df):
        return df.apply(utils.is_final_answer_correct, axis=1).mean()