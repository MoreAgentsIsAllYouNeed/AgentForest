import utils
import json
from more_agent import MoreAgent
from prompt_lib import interaction_prompt

class MATH(MoreAgent):
    def __init__(self, agents_num, model_type, nums=1, temperature=1, top_p=1):
        self.qtype = "math"
        self.ans_parser = utils.math_ans_parser
        self.format_prompt = "Please put your answer in the form \\boxed{{answer}}, at the end of your response."
        super().__init__(agents_num, model_type, nums, temperature, top_p)

    def get_question_datas(self):
        path = "../dataset/math_dataset/math_subset_20.json"
        sampledMathSet = json.load(open(path))
        question_datas = []
        for level in sampledMathSet.keys():
            for category in sampledMathSet[level].keys():
                for problem in sampledMathSet[level][category]:
                    solution, ok = utils.math_ans_parser(problem["solution"])
                    # print(f"idd: {idd}, solution: {solution}, ok: {ok}")
                    if solution is None:
                        raw_solution = problem["solution"]
                        # print(f"cannot extract \\box\ at level-{level}-category-{category}. Solution is:\n {raw_solution}\n\n")
                        continue
                    question_state = interaction_prompt["math"]["question"].format(problem["problem"])
                    question_data = {
                        "level": level,
                        "category": category,
                        "state": question_state,
                        "ground_truth": solution,
                    }
                    question_datas.append(question_data)
        return question_datas

    def get_final_answer(self, idxs, question_data):
        agent_answers = [self.nodes[idx].get_answer() for idx in idxs]
        consistent_answer = utils.get_majority_voting_answer_for_math(agent_answers)
        return consistent_answer

    def evaluation(self, df):
        return df.apply(utils.is_final_answer_correct, axis=1).mean()