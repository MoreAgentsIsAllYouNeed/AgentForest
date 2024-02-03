import utils
from more_agent import MoreAgent
from dataloader import sample_chess
from prompt_lib import interaction_prompt


class CHESS(MoreAgent):
    def __init__(self, agents_num, model_type, nums=1, temperature=1, top_p=1):
        self.qtype = "chess"
        self.ans_parser = utils.chess_ans_parser
        self.format_prompt = "Please state your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."
        super().__init__(agents_num, model_type, nums, temperature, top_p)

    def get_question_datas(self, question_num=100):
        chess_datas = sample_chess("../dataset/chess_dataset/task.json", question_num)
        question_datas = []
        for chess_data in chess_datas:
            question_state = interaction_prompt['chess']['question'].format(chess_data['input'], chess_data['move'])
            question_data = {
                "state": question_state,
                "ground_truth": chess_data["target"],
            }
            question_datas.append(question_data)
        return question_datas

    def get_final_answer(self, idxs, question_data):
        agent_answers = [self.nodes[idx].get_answer() for idx in idxs]
        consistent_answer = utils.get_majority_voting_answer(agent_answers)
        return consistent_answer

    def evaluation(self, df):
        return df.apply(utils.is_final_answer_in_ground_truth, axis=1).mean()