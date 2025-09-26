### Prompt Explanation

1_extract.txt:
Used in preattack stage, identifies the harm_target and query_details. The harm_target is used to help create the actors later on, and generate the query attack chain. The query_details does not seem to be used further along in the attack. 

2_network.txt
Used in preattack stage, creating actors based on the harm_target.

3_actor.txt, 3_more_actor.txt
Used in preattack stage, chooses the top n most relevant actors to use. n is specified at the beginning (default is 6), and can go up to a max of 10. If not enough actors are generated, more_actor prompt is used.

4_queries.txt
Used in preattack stage, generating the attack chain queries based on the harm_target, actors, and relationship between actor and harm_target passed in.

5_json_format.txt
Used to format responses

attack_modify.txt
Used in inattack stage. This is the prompt used to help modfiy the attack queries if dynamic modify is being used. 

attack_step_judge.txt
Used in inattack stage. This is the prompt used to judge the responses of the target model.

get_safe_response.txt
Not useful, used in creating dataset to finetune target model to be safer. 
