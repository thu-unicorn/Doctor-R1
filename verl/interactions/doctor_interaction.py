# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score import doctor_reward
import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from .base import BaseInteraction
from IPython import embed
import torch
import openai

BASE_URL = ""
API_KEY = ""
MODEL = ""

def simulate_patient_response(conversation_history: str, history: List[Dict[str, str]]) -> str:
    system_prompt = f"""You are an outpatient currently experiencing health issues. Your task is to simulate a patient-doctor interaction in the upcoming consultation dialogue.

In this simulated conversation, you will play the role of the patient, and the user will play the role of the doctor. Please follow these guidelines:

1. Simulate realistic patient behavior and reactions. Ensure your communication feels authentic. You may use natural expressions, including hesitation, pauses, or emotional fluctuations, to enhance the realism of the patient role.
2. Do not reveal all key information at once. Like a real patient, gradually disclose deeper concerns and core issues as the conversation progresses.

You are now the patient. No matter what the doctor asks, respond strictly in character as the patient.
"""
    
    INSTRUCTION_PROMPT = system_prompt + "\n"
    for msg in conversation_history:
        role = "Doctor" if msg["role"] == "assistant" else "Patient"
        INSTRUCTION_PROMPT += f"{role}：{msg['content']}\n"
    for msg in history:
        role = "Doctor" if msg["role"] == "assistant" else "Patient"
        INSTRUCTION_PROMPT += f"{role}：{msg['content']}\n"

    client = openai.OpenAI(
        api_key=API_KEY, 
        base_url=BASE_URL
    )
    
    response = client.completions.create(
        model=MODEL,
        prompt=INSTRUCTION_PROMPT,
        temperature=1.0,
        max_tokens=256,
    )

    outputs = response.choices[0].text
    # print(outputs)

    reply = outputs.split("Patient：")[-1].strip()
    return reply


class DoctorInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of doctor agent.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self, 
        instance_id: Optional[str] = None, 
        ground_truth: Optional[str] = None, 
        query: Optional[str] = None,  
        disease: Optional[str] = None, 
        max_turns: int = 5, 
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self.max_turns = max_turns
        self.conversation_history = query
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "asked_questions": set(),
            "turn_count": 0,
            "conversation_history": query,
            "disease": disease,
            "history": [], # To store conversation history for the patient model
        }
        if self.conversation_history == None:
            print("Error retrieve conversational history")
            embed()
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:

    
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break


        if not content: content = ""
        
        self._instance_dict[instance_id]["response"] = content
        print(f"[{instance_id}] Turn {self._instance_dict[instance_id]['turn_count'] + 1}. Received content: {content}")

        match1 = re.search(r"<think>(.*)</think>", content, re.DOTALL)
        match2 = re.search(r"<answer>(.*)</answer>", content, re.DOTALL)
        match_q = re.search(r"Question:\s*(.*)", content, re.DOTALL)

        think_content = match1.group(1).strip() if match1 else None
        answer_content = match2.group(1).strip() if match2 else None
        question_content = match_q.group(1).strip() if match_q else None
        
        if not think_content or not answer_content:
            response = f"Invalid response format. The correct format is: '<think> [your thinking] </think> <answer> [your reply] </answer>"
            return False, response, -0.2, {}

        self._instance_dict[instance_id]["turn_count"] += 1
        patient_reply = ""
        have_replied= False

        # Generate patient conversations
        if question_content:
            # Check for maximum number of turns
            if self._instance_dict[instance_id]["turn_count"] > self.max_turns:
                return True, "Failed to provide a treatment recommendation within the allowed number of turns.", -0.3, {}

            # Check for duplicates
            if question_content in self._instance_dict[instance_id]["asked_questions"]:
                response = f"Repeated question: '{question_content}'"
                return False, response, -0.2, {}

            # Effective questions
            self._instance_dict[instance_id]["asked_questions"].add(question_content)
            history = self._instance_dict[instance_id]["history"]
            history.append({"role": "assistant", "content": question_content})
            patient_reply = simulate_patient_response(
                conversation_history=self.conversation_history,
                history=self._instance_dict[instance_id]["history"]
            )
            history.append({"role": "user", "content": patient_reply})
            have_replied = True


        # Rewards for Calculating Thinking and Response
        if answer_content or think_content:
            reward = await self.calculate_score(instance_id)
            
            if answer_content and not have_replied:
                history = self._instance_dict[instance_id]["history"]
                history.append({"role": "assistant", "content": answer_content})
                patient_reply = simulate_patient_response(
                    conversation_history=self._instance_dict[instance_id]["conversation_history"],
                    history=self._instance_dict[instance_id]["history"]
                )
                history.append({"role": "user", "content": patient_reply})
            
            if "recommendation" in answer_content.lower():
                should_terminate_sequence = True
                return should_terminate_sequence, patient_reply, reward, {}
            else:
                should_terminate_sequence = False
                return should_terminate_sequence, patient_reply, reward, {}

        else:
            reward = -0.2
            response = "Incorrect answer format."
            should_terminate_sequence = False
            return should_terminate_sequence, response, reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return doctor_reward.compute_score(
            solution_str=self._instance_dict[instance_id]["response"],      # doctor agent's response
            ground_truth=self._instance_dict[instance_id]["ground_truth"],
            conversation_history=self._instance_dict[instance_id]["conversation_history"],
            disease=self._instance_dict[instance_id]["disease"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )


    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
