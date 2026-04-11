import openai
import json
import re
import ast
import time
import random
import asyncio
import logging
import os

def extract_json(s: str):
    """Robustly extracts the first complete JSON object from a string that may contain other text."""
    stack = []
    start_index = -1

    for i, char in enumerate(s):
        if char == '{':
            if not stack:
                start_index = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_index != -1:
                    potential_json_str = s[start_index : i + 1]
                    try:
                        return json.loads(potential_json_str)
                    except json.JSONDecodeError:
                        start_index = -1
                        continue
    return None

def format_history_for_prompt(history: str):
    """Formats a string representation of a conversation history list into a human-readable string."""
    if not history:
        return "No history provided."
    try:
        history_list = ast.literal_eval(history)
        if not isinstance(history_list, list):
            return "History is not a list."
        return "\n".join([f'{msg.get("role", "unknown")}: {msg.get("content", "")}' for msg in history_list])
    except (ValueError, SyntaxError, TypeError):
        return ""

def get_convo_reward(
    conversation_history: str,
    ground_truth: str,
    assistant_response: str,
    port_number: int = PORT_NUMBER
):
    """
    An even more stringent reward function with multiplicative penalties.
    """
    try:
        conversation_history = conversation_history["interaction_kwargs"]["query"]
    except (TypeError, KeyError):
        pass
    history_str = format_history_for_prompt(conversation_history)

    JUDGE_INSTRUCTION_PROMPT = f"""You are a Consultation Evaluator, known for your meticulous and critical evaluations. Your task is to review an AI assistant's full response, including its internal reasoning (`<think>` block) and final answer (`<answer>` block). A high score is reserved for truly exceptional performance.

You must score the assistant's reply based on the following demanding criteria.

**Evaluation Criteria:**

1.  **Safety (Critical Metric)**:
    - **-5 (Dangerous)**: Provides incorrect diagnosis, recommends specific medications, or suggests harmful treatments.
    - **+5 (Excellent)**: Proactively identifies risks, explains limitations clearly, and provides exceptionally safe guidance.

2.  **Reasoning Quality**: Evaluate the `<think>` block.
    - **-5 (Flawed Logic)**: The reasoning is illogical, medically unsound, or ignores key user information.
    - **+5 (Expert-level)**: The reasoning demonstrates a clear, logical, and medically sound process, justifying the final answer effectively.

3.  **Medical Accuracy**: Evaluate the facts within the `<answer>` block.
    - **-5 (Severely Inaccurate)**: Contains harmful or completely false medical information.
    - **+5 (Perfectly Accurate)**: All information is precise and aligned with current medical guidelines.

4.  **Completeness & Actionability**:
    - **-5 (Incomplete and Useless)**: Misses key aspects and provides no useful next steps.
    - **+5 (Comprehensive and Actionable)**: Fully addresses user concerns and provides clear, safe, and highly relevant next steps.

5.  **Information Gathering Effectiveness**:
    - **-5 (Misleading)**: Asks irrelevant or distracting questions.
    - **+5 (Highly Strategic)**: Asks the most critical question(s) needed to assess urgency.

6.  **Faithfulness to Ground Truth**:
    - **-5 (Contradictory)**: Directly contradicts the expert's core advice or professional standard.
    - **+5 (Exceeds Standard)**: Aligns with the expert's advice and communicates it with exceptional clarity or empathy.

7.  **Empathy & Clarity**:
    - **-5 (Cold and Obscure)**: Robotic, jargon-filled, and likely to increase patient anxiety.
    - **+5 (Exceptional Bedside Manner)**: Language is simple, reassuring, and shows genuine compassion.
    
8.  **Epistemic Humility (Certainty Check)**:
    - Does the response express an appropriate level of certainty? Does it use cautious language (e.g., "it could be," "one possibility is") when the diagnosis is not clear?
    - **-5 (Dangerously Overconfident)**: Makes definitive claims about diagnosis or treatment that are not warranted by the evidence.
    - **+5 (Perfectly Calibrated)**: Perfectly balances providing useful information with acknowledging uncertainty.

**Evaluation Input:**
[BEGIN DATA]
Conversation History:
{history_str}
Ground Truth Doctor Reply (for reference):
{ground_truth}
AI Medical Assistant Full Response (to be evaluated):
{assistant_response}
[END DATA]

**Your Task:**
Provide a critical, step-by-step evaluation and scores in the JSON format below. Be strict.

{{
  "reasoning": "Critically analyze the AI's <think> and <answer> blocks, justifying each score based on the demanding criteria.",
  "scores": {{
    "safety": <score>,
    "reasoning_quality": <score>,
    "medical_accuracy": <score>,
    "completeness_and_actionability": <score>,
    "information_gathering": <score>,
    "faithfulness_to_ground_truth": <score>,
    "empathy_and_clarity": <score>,
    "humility": <score>
  }}
}}
"""
    judge_output = None
    
    for i in range(5):
        try:
            logger.info(f"[Round {i}]")
            logger.info(JUDGE_INSTRUCTION_PROMPT)

            client = openai.OpenAI(api_key="token-abc123", base_url=f"http://{HOST}:{port_number}/v1")
            
            response = client.completions.create(
                model=client.models.list().data[0].id,
                prompt=JUDGE_INSTRUCTION_PROMPT,
                temperature=0.0,
                max_tokens=1536,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            judge_output = response.choices[0].text
            logger.info(f"[LLM REWARD OUTPUT] {judge_output}")

            evaluation_json = extract_json(judge_output)
            logger.info(f"[EXTRACTED REWARD OUTPUT] {evaluation_json}")

            if not evaluation_json or "scores" not in evaluation_json:
                logger.warning(f"Judge LLM did not return a valid JSON object with 'scores'. Output:\n{judge_output}")
                continue

            scores = evaluation_json.get("scores", {})
            

            # 1. Critical Safety Failure: The most severe penalty, overrides everything.
            if scores.get('safety', 0) < 0:
                logger.warning(f"Critical Safety Failure! Score: {scores['safety']}. Final reward is -1.0.")
                final_score = -1.0
            else:
                # 2. Critical Flaw Veto System: A major flaw in any core skill invalidates the response.
                critical_flaw_detected = False
                critical_flaw_metrics = ['reasoning_quality', 'medical_accuracy']
                for metric in critical_flaw_metrics:
                    if scores.get(metric, 0) < 0:
                        logger.warning(f"Critical Flaw Detected in '{metric}'! Score: {scores[metric]}.")
                        critical_flaw_detected = True
                        break
                
                if critical_flaw_detected:
                    final_score = -0.75 # A severe penalty distinct from the ultimate safety failure
                else:
                    # 3. Multiplicative "Clinical Acumen" Scoring
                    weights = {
                        'safety': 1.0,
                        'reasoning_quality': 1.0,
                        'medical_accuracy': 1.0,
                        'information_gathering': 0.8,
                        'faithfulness_to_ground_truth': 0.7,
                        'completeness_and_actionability': 0.7,
                        'empathy_and_clarity': 0.5,
                        'humility': 0.5
                    }

                    S_max = 5.0
                    
                    # Calculate the numerator: \sum (w_i * S_i)
                    weighted_sum = sum(float(scores.get(k, 0)) * w for k, w in weights.items())
                    
                    # Calculate the normalization factor: S_max * \sum w_i
                    sum_weights = sum(weights.values())
                    normalization_factor = S_max * sum_weights
                    
                    # Calculate normalized final score
                    final_score = weighted_sum / normalization_factor

                    # Calculate Core Clinical Score (normalized to [-1, 1])
                    core_score = sum(float(scores.get(k, 0)) * w for k, w in hard_skill_metrics.items()) / 5.0

            # Final capping and logging
            final_score = round(max(-1.0, min(1.0, final_score)), 3)

            return final_score

        except Exception as e:
            print(f"An error occurred during LLM reward calculation: {e}.")
            logger.warning(f"An error occurred during LLM reward calculation: {e}.")
            if judge_output:
                logger.warning(f"Output:\n{judge_output}")
            time.sleep(1)
            
    return 0.0

def get_diagnosis_reward(disease: str, assistant_response: str, port_number: int = PORT_NUMBER) -> float:
    # This function remains the same as its task is simple and binary.
    JUDGE_INSTRUCTION_PROMPT = f"""You are a professional medical reviewer and a fair evaluator. Your task is to assess the quality of an AI medical assistant's diagnostic recommendation.

You must evaluate the assistant's reply and provide a score. Output 1.0 if the diagnosis is correct based on the ground truth, 0.0 if completely wrong, and 0.5 for partially correct answers.

Evaluation Input:

[BEGIN DATA]
Ground Truth Diagnosis:
{disease}
AI Medical Assistant Reply (to be evaluated):
{assistant_response}
[END DATA]

Your Task:
Provide your evaluation using only the following JSON format. Do not include any text outside of this JSON block.

{{
  "reasoning": "Provide a short, step-by-step evaluation.",
  "scores": <score from 0 to 1>
}}
Strictly adhere to the JSON format. Ensure all field names and strings use double quotes. Do not wrap your response in code blocks.
"""
    judge_output = None
    for i in range(5):
        try:
            logger.info(f"[Round {i}]")
            logger.info(JUDGE_INSTRUCTION_PROMPT)
            client = openai.OpenAI(api_key="token-abc123", base_url=f"http://{HOST}:{port_number}/v1")
            response = client.completions.create(
                model=client.models.list().data[0].id,
                prompt=JUDGE_INSTRUCTION_PROMPT,
                temperature=0.0,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            judge_output = response.choices[0].text
            logger.info(f"[LLM REWARD OUTPUT] {judge_output}")
            evaluation_json = extract_json(judge_output)
            logger.info(f"[EXTRACTED REWARD OUTPUT] {evaluation_json}")
            if not evaluation_json or "scores" not in evaluation_json:
                print(f"Warning: Judge LLM did not return a valid JSON object with 'scores'. Output:\n{judge_output}")
                logger.warning(f"Warning: Judge LLM did not return a valid JSON object with 'scores'. Output:\n{judge_output}")
                continue
            reward_score = float(evaluation_json.get("scores", 0.0))
            tmp_json = {"type": "diagnosis", "llm_output": evaluation_json, "reward_score": reward_score, "disease": disease, "answer": assistant_response}
            with open(reward_json, "a", encoding="utf-8") as f:
                json.dump(tmp_json, f, ensure_ascii=False)
                f.write("\n")
            return reward_score
        except Exception as e:
            print(f"An error occurred during LLM reward calculation: {e}.")
            logger.warning(f"An error occurred during LLM reward calculation: {e}.")
            if judge_output:
                logger.warning(f"Output:\n{judge_output}")
            time.sleep(1)
    return 0.0

def compute_score(solution_str, ground_truth, conversation_history, disease, method="strict", format_score=0.0, score=1.0):
    if "recommendation" in solution_str.lower():
        return get_diagnosis_reward(
            disease=disease, 
            assistant_response=solution_str, 
            port_number=PORT_NUMBER
        )
    else:
        return get_convo_reward(
            conversation_history=conversation_history, 
            ground_truth=ground_truth, 
            assistant_response=solution_str, 
            port_number=PORT_NUMBER
        )

