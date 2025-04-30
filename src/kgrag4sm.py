import openai
from typing import Tuple, Optional, Union
from transformers import pipeline

class KGRAG_for_Schema_Matching:
    def generate_system_prompt(self) -> str:
        return """
                You are an expert in schema matching and data integration. 
                Your task is to analyze the attribute 1 with its textual description 1 and attribute 2 with its textual description 2 from source and target schema in the given question, and specify if the attribute 1 from source schema is semantically matched with attribute 2 from the target schema.
                In some questions, there is the knowledge graph context that might be helpful for you to answer. In this case, you will need to consider the provided context to make the correct decision. \n\n

                Here are two examples of the schema matching questions with correct answers and explanations that you need to learn before you start to analyze the potential mappings:
                Example 1:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death; a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table. 
                Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; beneficiary code. 
                Do attribute 1 and attribute 2 are semantically matched with each other?
                Here is the correct answer and the explanations for the above-given example question: 1 
                Explanation: they are semantically matched with each other because both of them are unique identifiers for each person. Even if the death-person_id refers to the unique identifier of the person in the death table and beneficiarysummary-desynpuf_id refers to the unique identifier of the person beneficiary from beneficiarysummary table, they are semantically matched with each other. \n\n
                        
                Example 2:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death.;a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table. 
                Attribute 2 beneficiarysummary-bene_birth_dt and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; date of birth. 
                Do attribute 1 and attribute 2 are semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question: 
                death (Q4), has part(s) of the class (P2670), date of death (Q18748141) -> date of death (Q18748141), opposite of (P461), date of birth (Q2389905) | human (Q5), has characteristic (P1552), age of a person (Q185836) -> age of a person (Q185836), uses (P2283), date of birth (Q2389905)
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because death-person_id is a unique identifier for each person in death table and bene_birth_dt is the date of birth of person in beneficiarysummary table. From the above context, we can found that date of death is opposite of date of birth, they are not semantically matched with each other.
                        
                Remember the following tips when you are analyzing the potential mappings.
                Tips:
                (1) Some letters are extracted from the full names and merged into an abbreviation word.
                (2) Schema information sometimes is also added as the prefix of abbreviation.
                (3) Please consider the abbreviation case. 
                (4) Please consider the knowledge graph context to make the correct decision when it is provided.
                """

    def generate_user_prompt(self, question: str, paths: Optional[str]) -> str:
        #f""" """ 是一种用于构造多行格式化字符串的方式,f-string
        prompt = f"""Based on the provided example and the following knowledge graph context, please answer the following schema matching question:
        
        {question}

        Knowledge Graph Context:
        {paths if paths and paths != ['null'] else "No available knowledge graph context, please make the decision yourself. "}

        Please respond with the label: 1 if attribute 1 and attribute 2 are semantically matched with each other, otherwise respond lable: 0.
        Do not mention that there is not enough information to decide.
        """
        return prompt
    #system_prompt :prompt offered by system, describe task or enviroment
    #user_prompt : prompt offered by user,describe actual question或要求模型生成响应的内容。
    def get_llm_response(self, system_prompt: str, user_prompt: str, model: Union[str, pipeline]) -> str:
        if isinstance(model, str) and model.startswith('gpt'):
            #build OpenAI client,this client is used to send request and receive response
            client = openai.Client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        else:
            # 仅修改这一行：添加jellyfish-7b的特殊提示格式
            if hasattr(model, 'model') and "jellyfish-7b" in str(model.model.config._name_or_path).lower():
                input_text = f"### System:\n{system_prompt}\n### User:\n{user_prompt}\n### Assistant:\n"
                response = model(input_text, max_new_tokens=4096)
                return response[0]['generated_text'].split("### Assistant:")[-1].strip()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            terminators = [
                #eos_token_id：This is tokenizer's end-of-sequence ID。
                #在生成任务中，模型会生成一个 token，当生成的 token 与 eos_token_id 匹配时，表示文本已经完成。
                model.tokenizer.eos_token_id,
                #自定义的结束符 ID，它通过将自定义 token "<|eot_id|>" 转换为 ID 来实现。
                model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            responses = model(
                messages,
                eos_token_id=terminators,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.5,
                top_k=1,
                top_p=0.9,
                pad_token_id=model.tokenizer.eos_token_id
            )
            answer = responses[0]['generated_text'][-1]["content"].strip()
            return answer
    
    def kgrag_query_for_schema_matching(self, question: str, paths: Optional[str], model) -> Tuple[str, str, str]:
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(question, paths)
        response = self.get_llm_response(system_prompt, user_prompt, model)
        return system_prompt, user_prompt, response