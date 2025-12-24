import json
import torch
import numpy as np
from filelock import FileLock
from types import SimpleNamespace
import os
import openai
from transformers import AutoTokenizer
import cohere

# --- 1. Global Configuration and Setup (Unchanged) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_API_BASE_URL = ""
EMBEDDING_API_KEY = ""
EMBEDDING_DIMENSIONS = 512
EMBEDDING_MODEL = ""
RERANKER_API_BASE_URL = ""
RERANKER_API_KEY = ""
RERANKER_MODEL = ""


print("Setting up Embedding API client...")
try:
    embedding_client = openai.OpenAI(base_url=EMBEDDING_API_BASE_URL, api_key=EMBEDDING_API_KEY)
    print(f"Embedding API client configured successfully. Using model: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error setting up Embedding API client: {e}")
    exit()

print("Setting up Reranker API client...")
try:
    reranker_client = cohere.Client(base_url=RERANKER_API_BASE_URL, api_key=RERANKER_API_KEY)
    print(f"Reranker API client configured successfully. Using model: {RERANKER_MODEL}")
except Exception as e:
    print(f"Error setting up Reranker API client: {e}")
    exit()

# --- Helper functions (Unchanged) ---
def get_embeddings_api(texts: list[str], device: torch.device) -> torch.Tensor | None:
    """
    Retrieves embeddings for a list of texts using the configured API.
    Returns a PyTorch tensor on the specified device.
    """
    if not texts: return None
    try:
        responses = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=texts, dimensions=EMBEDDING_DIMENSIONS)
        embeddings_list = [data.embedding for data in responses.data]
        return torch.tensor(np.array(embeddings_list), dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Error calling embedding API: {e}")
        return None

def format_experiences_for_prompt(experiences: list[dict], use_action_suggestion: bool = True) -> str:
    """
    Formats a list of experience dictionaries into a string for the prompt.
    The format depends on whether action suggestion is enabled.
    """
    if not experiences: return "No Relevant Experience."
    formatted_texts = []
    for i, exp in enumerate(experiences):
        state_text = exp.get("state", "N/A")
        if use_action_suggestion:
            action_text = exp.get("action", "N/A")
            formatted_texts.append(f"{i+1}. State: {state_text}\n   Action: {action_text}")
        else:
            formatted_texts.append(f"{i+1}. {state_text}")
    return "\n".join(formatted_texts)

def load_and_prepare_experiences(file_path: str, device: torch.device) -> tuple[list[dict], torch.Tensor | None]:
    """
    Loads experiences from a file and computes their embeddings via API.
    """
    print(f"\nLoading experiences from {file_path} and preparing embeddings...")
    experiences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                experiences.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("Experience file not found. Starting with an empty database.")
        return [], None
    if not experiences: return [], None
    stored_states = [exp["state"] for exp in experiences]
    stored_embeddings = get_embeddings_api(stored_states, device=device)
    if stored_embeddings is not None:
        print(f"Created {len(stored_embeddings)} stored embeddings on device: {stored_embeddings.device}")
    return experiences, stored_embeddings

def save_experience_batch(experiences: list[dict], file_path: str):
    """Safely appends a batch of experiences to a JSONL file (CPU operation)."""
    if not experiences: return
    lock_path = file_path + ".lock"
    lock = FileLock(lock_path)
    try:
        with lock:
            with open(file_path, 'a', encoding='utf-8') as f:
                for exp in experiences:
                    exp["state"] = exp["state"].replace("user\nYou are an experienced doctor tasked with providing a professional diagnosis and treatment plan for a patient through a consultation dialogue. Please carefully listen to the patient's responses, ask targeted questions.\n\nQuick Guide\nObjective:\n1. Gather key information through effective questioning. Each question should be based on the previous round’s information. Avoid repeating questions.\n\nRules:\n1. Complete both action per turn: provide thinking and ask a question.\n2. Repetitive or similar questions are strictly prohibited.\n\nResponse Format:\n<think> [Your reasoning] </think>\n<answer> If information is insufficient, ask one question only, in the following format:\nQuestion: (Your question).\n</answer>\n<answer> If information is sufficient, provide diagnosis and recommendation, in the following format:\nRecommendation: (Your diagnosis and recommendation)\n</answer>.\n\n\nDecide your next action:\nAlways output: <think> [Your reasoning] </think> <answer> [Your reply] </answer> Do not include any additional text. Follow this format strictly.\nuser\nuser\n")[-1]
                    f.write(json.dumps(exp, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving experiences to {file_path}: {e}")

# --- MODIFIED: The Core Logic with 3 Stages ---
def retrieve_rerank_and_filter_batch(
        query_states: list[str],
        all_experiences: list[dict],
        stored_embeddings: torch.Tensor,
        top_k: int = 3,
        reward_coefficient: float = 0.5,
        rerank_top_n: int = 30,
        use_novelty_filter: bool = True,
        novelty_threshold: float = 0.95,
        high_reward_std_factor: float = 1.0
    ) -> list[list[dict]]:
    """
    Retrieves experiences using a three-stage retrieve, rerank, and filter process.
    """
    if not all_experiences or stored_embeddings is None:
        return [[] for _ in range(len(query_states))]

    # --- STAGE 1: Fast Retrieval + Store Similarity Scores ---
    print("--- Stage 1: Retrieving Candidates via Embedding Similarity ---")
    query_embeddings = get_embeddings_api(query_states, device=stored_embeddings.device)
    if query_embeddings is None: return [[] for _ in range(len(query_states))]

    def cos_sim(a: torch.Tensor, b: torch.Tensor):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))
    
    similarities = cos_sim(query_embeddings, stored_embeddings)
    rewards = torch.tensor([exp.get("reward", 0.0) for exp in all_experiences], device=similarities.device)
    combined_scores = similarities + (reward_coefficient * rewards)

    num_to_retrieve = min(rerank_top_n, len(all_experiences))
    if num_to_retrieve <= 0: return [[] for _ in range(len(query_states))]
    
    _, top_indices = torch.topk(combined_scores, k=num_to_retrieve, dim=1)
    top_similarities = torch.gather(similarities, 1, top_indices)
    top_indices_cpu = top_indices.cpu().numpy()
    top_similarities_cpu = top_similarities.cpu().numpy()

    batch_final_experiences = []
    for i, query in enumerate(query_states):
        candidate_indices = top_indices_cpu[i]
        candidate_experiences = []
        for j, idx in enumerate(candidate_indices):
            exp = all_experiences[idx].copy()
            exp['similarity_score'] = top_similarities_cpu[i][j]
            candidate_experiences.append(exp)
        
        candidate_docs = [exp['state'] for exp in candidate_experiences]
        if not candidate_docs:
            batch_final_experiences.append([])
            continue

        # --- STAGE 2: Accurate Reranking ---
        print(f"\n--- Stage 2: Reranking candidates for query: '{query}' ---")
        try:
            rerank_response = reranker_client.rerank(
                model=RERANKER_MODEL,
                query=query,
                documents=candidate_docs,
            )
            
            # Map reranked results back to our rich experience objects
            experience_map = {exp['state']: exp for exp in candidate_experiences}
            reranked_experiences = []
            for result in rerank_response.results:
                original_doc_text = result.document.text
                if original_doc_text in experience_map:
                    exp = experience_map[original_doc_text]
                    exp['relevance_score'] = result.relevance_score
                    reranked_experiences.append(exp)
        
        except Exception as e:
            print(f"  [Error] Reranking failed for query '{query}': {e}. Falling back to similarity ranking.")
            reranked_experiences = sorted(candidate_experiences, key=lambda x: x['similarity_score'], reverse=True)

        # --- STAGE 3: Post-processing with Novelty Filter ---
        print(f"--- Stage 3: Applying Novelty Filter ---")
        if not use_novelty_filter:
            # If filter is off, just take the top_k from the reranked list
            final_exps_for_query = reranked_experiences[:top_k]
        else:
            # If filter is on, apply the logic
            final_exps_for_query = []
            
            # Calculate dynamic reward threshold based on the CANDIDATE set
            candidate_rewards = [exp.get("reward", 0.0) for exp in reranked_experiences]
            if len(candidate_rewards) > 1:
                rewards_tensor = torch.tensor(candidate_rewards, dtype=torch.float32)
                mean_reward = rewards_tensor.mean()
                std_reward = rewards_tensor.std()
                dynamic_high_reward_threshold = (mean_reward + high_reward_std_factor * std_reward).item()
            else:
                dynamic_high_reward_threshold = -1.0 # No meaningful threshold

            print(f"  [Info] Dynamic reward threshold for this query: {dynamic_high_reward_threshold:.4f}")

            # Iterate through the reranked list and apply the filter
            for exp in reranked_experiences:
                similarity_val = exp['similarity_score']
                reward_val = exp.get("reward", 0.0)

                # The filter condition: is the item too similar AND not highly rewarding?
                if similarity_val >= novelty_threshold and reward_val < dynamic_high_reward_threshold:
                    print(f"  [Filter] Filtering out: '{exp['state'][:20]}...' "
                          f"(Sim: {similarity_val:.2f}, Reward: {reward_val:.2f})")
                    continue # Skip this item

                # If it passes the filter, add it to the final list
                final_exps_for_query.append(exp)
                
                # Stop when we have enough results
                if len(final_exps_for_query) >= top_k:
                    break
        
        batch_final_experiences.append(final_exps_for_query)

    return batch_final_experiences

def build_augmented_batch_hyper_optimized(
        original_gen_batch: dict,
        topk_experience_list: list[list[dict]],
        tokenizer,
        config,
        device: torch.device,
        use_action_suggestion: bool = True,
    ) -> dict:
    """
    This function combines retrieved experiences with the original query to create a final prompt.
    """
    template_header = "Experience References：\n"
    template_middle = "\n\nOriginal Question：\n"
    header_tokens = tokenizer(template_header, add_special_tokens=False, return_tensors="pt").input_ids.to(device).squeeze(0)
    middle_tokens = tokenizer(template_middle, add_special_tokens=False, return_tensors="pt").input_ids.to(device).squeeze(0)
    
    batch_experience_strs = [
        format_experiences_for_prompt(exps, use_action_suggestion) if exps else "" 
        for exps in topk_experience_list
    ]

    experience_tokens_batch = tokenizer(
        batch_experience_strs,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=config.data.max_prompt_length // 2,
        return_tensors="pt"
    ).to(device)

    final_input_ids_list = []
    for i in range(len(topk_experience_list)):
        original_tokens = original_gen_batch['input_ids'][i].to(device)
        original_attention_mask = original_gen_batch['attention_mask'][i].to(device)
        actual_original_tokens = original_tokens[original_attention_mask == 1].to(device)

        exp_tokens = experience_tokens_batch['input_ids'][i]
        exp_attention_mask = experience_tokens_batch['attention_mask'][i]
        actual_exp_tokens = exp_tokens[exp_attention_mask == 1]
        
        if len(actual_exp_tokens) > 0:
            combined_ids = torch.cat([
                header_tokens,
                actual_exp_tokens,
                middle_tokens,
                actual_original_tokens
            ], dim=0)
        else:
            combined_ids = torch.cat([
                middle_tokens.squeeze(0),
                actual_original_tokens
            ], dim=0)
        
        final_input_ids_list.append(combined_ids)

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        final_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    max_length = config.data.max_prompt_length
    if padded_input_ids.shape[1] > max_length:
        padded_input_ids = padded_input_ids[:, :max_length]
        
    new_attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()
    seq_len = padded_input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).expand_as(padded_input_ids)
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": new_attention_mask,
        "position_ids": position_ids,
    }