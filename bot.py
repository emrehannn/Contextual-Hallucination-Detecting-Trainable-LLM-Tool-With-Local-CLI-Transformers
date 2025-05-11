import os
from dotenv import load_dotenv
# IMPORTANT: Set these environment variables BEFORE importing torch or transformers
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Adjust if your dGPU ID is different

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import joblib
import time

# --- Helper for colored text ---
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'

    @staticmethod
    def red(text):
        return f"{Colors.RED}{text}{Colors.ENDC}"

    @staticmethod
    def green(text):
        return f"{Colors.GREEN}{text}{Colors.ENDC}"

    @staticmethod
    def yellow(text):
        return f"{Colors.YELLOW}{text}{Colors.ENDC}"

# --- PyTorch/CUDA Diagnostics ---
print(f"--- PyTorch/CUDA Diagnostics ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs available to PyTorch: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        current_gpu_id = torch.cuda.current_device()
        print(f"Current CUDA device ID: {current_gpu_id}")
        print(f"Device name: {torch.cuda.get_device_name(current_gpu_id)}")
    else:
        print("CUDA is available, but no GPUs were found by PyTorch (check CUDA_VISIBLE_DEVICES).")
else:
    print("CUDA is NOT available to PyTorch. Model will load on CPU.")
print(f"--- End Diagnostics ---")

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
CLASSIFIER_PATH = "classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl"
SLIDING_WINDOW_SIZE = 8

print(f"Targeting device for model operations: {DEVICE}")

# --- Load Model and Tokenizer ---
print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_AUTH_TOKEN)
print(f"Loading model {MODEL_ID}...")
model = None
if DEVICE == "cuda":
    print("Attempting to load model on CUDA...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True, token=HF_AUTH_TOKEN, output_attentions=True
        )
        print("Model loaded on CUDA (potentially distributed with CPU via device_map='auto') and attn_implementation='eager'.")
    except Exception as e:
        print(f"Error loading model on CUDA: {e}"); DEVICE = "cpu"
if DEVICE == "cpu" and model is None:
    print("Loading model on CPU...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager",
            trust_remote_code=True, token=HF_AUTH_TOKEN, output_attentions=True
        )
        print("Model loaded on CPU with attn_implementation='eager'.")
    except Exception as e: print(f"FATAL: Error loading model on CPU: {e}"); exit()
if model is None: print("FATAL: Model could not be loaded. Exiting."); exit()
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, 'config') and model.config is not None: model.config.pad_token_id = model.config.eos_token_id
print("Model and tokenizer setup complete.")

# --- Load Lookback Lens Classifier ---
lookback_lens_classifier = None
best_threshold_from_file = None # Initialize to None

try:
    loaded_data = joblib.load(CLASSIFIER_PATH)
    print(f"Successfully loaded data from {CLASSIFIER_PATH}")
    if isinstance(loaded_data, dict):
        classifier_key = 'clf'
        if classifier_key in loaded_data:
            lookback_lens_classifier = loaded_data[classifier_key]
            if hasattr(lookback_lens_classifier, 'predict_proba'):
                print(f"Lookback Lens classifier object extracted from key '{classifier_key}'.")
                if 'best_threshold' in loaded_data:
                    best_threshold_from_file = float(loaded_data['best_threshold']) # Ensure float
                    print(f"Found 'best_threshold' in PKL: {best_threshold_from_file:.4f}.")
            else: print(f"ERROR: Item at key '{classifier_key}' not valid."); lookback_lens_classifier = None
        else: print(f"ERROR: Key '{classifier_key}' not found in PKL.")
    elif hasattr(loaded_data, 'predict_proba'):
        lookback_lens_classifier = loaded_data
        print("Lookback Lens classifier loaded directly.")
    else: print(f"ERROR: Loaded PKL data is not dict or classifier. Type: {type(loaded_data)}")
except ModuleNotFoundError as e:
    if 'sklearn' in str(e): print(f"ERROR: pip install scikit-learn")
    else: print(f"Error loading classifier (ModuleNotFoundError): {e}")
    lookback_lens_classifier = None 
except FileNotFoundError: print(f"ERROR: Classifier not found at {CLASSIFIER_PATH}."); lookback_lens_classifier = None
except Exception as e: print(f"Error loading classifier: {e}"); lookback_lens_classifier = None

# --- ADJUST THIS THRESHOLD ---
LOOKBACK_LENS_THRESHOLD = 0.3 # Try a lower value, e.g., 0.3 or 0.25
if best_threshold_from_file is not None:
    # Optionally use the threshold from the file if it seems more appropriate for general use
    # LOOKBACK_LENS_THRESHOLD = best_threshold_from_file
    # print(f"Note: A 'best_threshold' of {best_threshold_from_file:.4f} was found in the classifier file.")
    pass # For now, stick to the manually set one for experimentation
print(f"Using Lookback Lens threshold: {LOOKBACK_LENS_THRESHOLD:.2f}")

# --- calculate_per_token_lookback_ratios_normalized function ---
def calculate_per_token_lookback_ratios_normalized(
    attentions_for_token_prediction, prompt_token_length, current_total_sequence_length
):
    ratios_all_heads_for_token = []
    if not attentions_for_token_prediction: return np.array([])
    num_layers = len(attentions_for_token_prediction)
    if num_layers == 0: return np.array([])
    key_sequence_length = current_total_sequence_length - 1
    if key_sequence_length <= 0: return np.array([])
    for layer_idx in range(num_layers):
        layer_attentions = attentions_for_token_prediction[layer_idx]
        if layer_attentions is None: continue
        if layer_attentions.dim() == 4 and layer_attentions.shape[2] == 1:
             layer_attentions_squeezed = layer_attentions.squeeze(2)
        elif layer_attentions.dim() == 3: layer_attentions_squeezed = layer_attentions
        else: continue
        actual_key_seq_len_in_tensor = layer_attentions_squeezed.shape[-1]
        if actual_key_seq_len_in_tensor <= 0: continue
        current_prompt_span_len = min(prompt_token_length, actual_key_seq_len_in_tensor)
        attn_to_context_sum = layer_attentions_squeezed[:, :, :current_prompt_span_len].sum(dim=-1)
        prev_gen_start_idx = current_prompt_span_len
        prev_gen_end_idx = actual_key_seq_len_in_tensor
        if prev_gen_start_idx < prev_gen_end_idx:
            attn_to_prev_generated_sum = layer_attentions_squeezed[:, :, prev_gen_start_idx:prev_gen_end_idx].sum(dim=-1)
        else: attn_to_prev_generated_sum = torch.zeros_like(attn_to_context_sum)
        denominator = attn_to_context_sum + attn_to_prev_generated_sum + 1e-9
        normalized_ratio = attn_to_context_sum / denominator
        ratios_all_heads_for_token.extend(
            normalized_ratio.to(torch.float32).cpu().detach().numpy().flatten()
        )
    return np.array(ratios_all_heads_for_token)

# --- Chat Loop ---
print("\nStarting chat (type 'quit' to exit).")
history_chat_format = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit': break
    current_prompt_chat_format = history_chat_format + [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(current_prompt_chat_format, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
    prompt_token_length = inputs.input_ids.shape[1]
    
    # Store CPU versions for extending attention mask correctly
    loop_input_ids_cpu = inputs.input_ids 
    loop_attention_mask_cpu = inputs.attention_mask

    print("Bot: ", end="", flush=True)
    all_generated_ids = []
    all_per_token_ratio_vectors = []
    past_key_values = None
    max_new_tokens = 100
    min_factual_prob_this_response = 1.0 
    hallucination_windows_detected_count = 0
    
    # For better text display
    last_printed_length = 0
    
    for step in range(max_new_tokens):
        input_device = model.device if hasattr(model, 'device') else DEVICE
        
        effective_input_ids = loop_input_ids_cpu.to(input_device)
        effective_attention_mask = loop_attention_mask_cpu.to(input_device)

        with torch.no_grad():
            outputs = model(
                input_ids=effective_input_ids,
                attention_mask=effective_attention_mask,
                past_key_values=past_key_values,
                use_cache=True, output_attentions=True, return_dict=True
            )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id_tensor = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        next_token_id_item = next_token_id_tensor.item()
        all_generated_ids.append(next_token_id_item)

        attentions_for_prediction_list = []
        if outputs.attentions:
            for layer_attention_matrix in outputs.attentions:
                att_from_last_token = layer_attention_matrix[:, :, -1:, :]
                attentions_for_prediction_list.append(att_from_last_token)
        attentions_for_prediction_tuple = tuple(attentions_for_prediction_list)
        current_total_len_incl_new_token = prompt_token_length + step + 1
        per_token_ratio_vec = calculate_per_token_lookback_ratios_normalized(
            attentions_for_prediction_tuple, prompt_token_length, current_total_len_incl_new_token
        )
        all_per_token_ratio_vectors.append(per_token_ratio_vec)

        # ---- IMPROVED STREAMING TEXT DISPLAY ----
        # Decode all tokens generated so far for proper spacing
        current_text = tokenizer.decode(all_generated_ids, skip_special_tokens=True)
        
        # Print only the new part
        new_text = current_text[last_printed_length:]
        print(new_text, end="", flush=True)
        
        # Update the last printed position
        last_printed_length = len(current_text)
        # ---- End improved streaming text display ----

        if next_token_id_item == tokenizer.eos_token_id: break
        
        loop_input_ids_cpu = next_token_id_tensor.cpu() # Next input is just the new token
        loop_attention_mask_cpu = torch.cat(
            [loop_attention_mask_cpu, torch.ones((loop_attention_mask_cpu.shape[0], 1), dtype=torch.long, device='cpu')],
            dim=-1
        )
        past_key_values = outputs.past_key_values

        if lookback_lens_classifier and len(all_per_token_ratio_vectors) >= SLIDING_WINDOW_SIZE:
            window_ratio_vectors = all_per_token_ratio_vectors[-SLIDING_WINDOW_SIZE:]
            valid_ratio_vectors = [v for v in window_ratio_vectors if v is not None and v.size > 0]
            if len(valid_ratio_vectors) == SLIDING_WINDOW_SIZE and \
               len(set(v.shape for v in valid_ratio_vectors)) == 1:
                v_bar_window = np.mean(np.stack(valid_ratio_vectors), axis=0)
                feature_vector_for_classifier = v_bar_window.reshape(1, -1)
                try:
                    prob_factual = lookback_lens_classifier.predict_proba(feature_vector_for_classifier)[0, 1]
                    min_factual_prob_this_response = min(min_factual_prob_this_response, prob_factual)
                    if prob_factual < LOOKBACK_LENS_THRESHOLD:
                        hallucination_windows_detected_count += 1
                except Exception: pass # Ignore classifier errors during live generation for now
    
    print() 
    final_decoded_response = tokenizer.decode(all_generated_ids, skip_special_tokens=True)

    if lookback_lens_classifier and all_per_token_ratio_vectors and len(all_generated_ids) > 0 :
        is_significant_hallucination = (
            hallucination_windows_detected_count > max(1, SLIDING_WINDOW_SIZE // 2) or 
            min_factual_prob_this_response < (LOOKBACK_LENS_THRESHOLD - 0.15) # Made this gap smaller
        ) and hallucination_windows_detected_count > 0

        if is_significant_hallucination:
             print(Colors.red(f"Lookback Lens: Potential significant deviation from context (Min P(Factual): {min_factual_prob_this_response:.2f}, {hallucination_windows_detected_count} flags)."))
        elif hallucination_windows_detected_count > 0:
             print(Colors.yellow(f"Lookback Lens: Some deviation from context observed (Min P(Factual): {min_factual_prob_this_response:.2f}, {hallucination_windows_detected_count} flags)."))
        else:
            # Only print green if min_factual_prob is reasonably high, otherwise stay neutral
            if min_factual_prob_this_response > 0.7 : # Example: only if min prob is high
                print(Colors.green(f"Lookback Lens: Response seems contextually grounded (Min P(Factual) over windows: {min_factual_prob_this_response:.2f})."))
            elif len(all_generated_ids)>0 : # if something was generated but not clearly green
                 print(f"Lookback Lens: (Min P(Factual) over windows: {min_factual_prob_this_response:.2f}, {hallucination_windows_detected_count} flags).")

    history_chat_format.append({"role": "user", "content": user_input})
    history_chat_format.append({"role": "assistant", "content": final_decoded_response})
    if len(history_chat_format) > 10: history_chat_format = history_chat_format[-10:]