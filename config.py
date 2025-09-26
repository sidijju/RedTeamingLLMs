class PreAttackConfig:
    def __init__(self,
                 model_name = 'gpt-4o',
                 actor_num = 3,
                 behavior_csv = './data/harmbench.csv',
                 extract_prompt = './prompts/1_extract.txt',
                 network_prompt = './prompts/2_network.txt',
                 actor_prompt = './prompts/3_actor.txt',
                 query_prompt = './prompts/4_queries.txt',
                 json_format_prompt = './prompts/5_json_format.txt',
                 more_actor_prompt = './prompts/3_more_actor.txt'):
        self.model_name = model_name
        self.actor_num = actor_num
        self.behavior_csv = behavior_csv
        self.extract_prompt = extract_prompt
        self.network_prompt = network_prompt
        self.query_prompt = query_prompt
        self.actor_prompt = actor_prompt
        self.json_format_prompt = json_format_prompt
        self.more_actor_prompt = more_actor_prompt

class InAttackConfig:
    def __init__(self,
                 attack_model_name = 'gpt-4o',
                 target_model_name = 'gpt-4o',
                 pre_attack_data_path = '',
                 step_judge_prompt = './prompts/attack_step_judge.txt',
                 modify_prompt = './prompts/attack_modify.txt',
                 early_stop = True,
                 dynamic_modify = True,
                 evaluate = False,
                 num_actors = 3,
                 use_goat = False):
        self.attack_model_name = attack_model_name
        self.target_model_name = target_model_name
        self.pre_attack_data_path = pre_attack_data_path
        self.step_judge_prompt = step_judge_prompt
        self.modify_prompt = modify_prompt
        self.early_stop = early_stop
        self.dynamic_modify = dynamic_modify
        self.evaluate = evaluate
        self.num_actors = num_actors


        # GOAT Attacks
        self.goat_step_judge_prompt = './prompts/goat/attack_step_judge.txt'
        self.dual_response_prompt = './prompts/goat/attack_dual_response.txt'
        self.hypotheticals_prompt = './prompts/goat/attack_hypotheticals.txt'
        self.opposite_intent_prompt = './prompts/goat/attack_opposite_intent.txt'
        self.persona_modification_prompt = './prompts/goat/attack_persona_modification.txt'
        self.refusal_suppression_prompt = './prompts/goat/attack_refusal_suppression.txt'
        self.response_priming_prompt = './prompts/goat/attack_response_priming.txt'
        self.topic_splitting_prompt = './prompts/goat/attack_topic_splitting.txt'
        self.use_goat = use_goat
        
class DatasetConfig:
    def __init__(self,
                 attack_data_file = '', 
                 safe_prompt = 'prompts/get_safe_response.txt', 
                 safe_model_name = 'deepseek-chat'):
        self.attack_data_file = attack_data_file
        self.safe_prompt = safe_prompt
        self.safe_model_name = safe_model_name