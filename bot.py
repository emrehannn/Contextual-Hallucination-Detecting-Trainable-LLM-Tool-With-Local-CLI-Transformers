import os
from dotenv import load_dotenv
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np

# --- Helper for colored text ---
class Colors:
    RED = '\033[91m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; BLUE = '\033[94m'; ENDC = '\033[0m'
    @staticmethod
    def red(text): return f"{Colors.RED}{text}{Colors.ENDC}"
    @staticmethod
    def green(text): return f"{Colors.GREEN}{text}{Colors.ENDC}"
    @staticmethod
    def yellow(text): return f"{Colors.YELLOW}{text}{Colors.ENDC}"
    @staticmethod
    def blue(text): return f"{Colors.BLUE}{text}{Colors.ENDC}"

# --- Configuration ---
load_dotenv()
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf" 
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
CLASSIFIER_PATH = "classifiers/classifier_anno-nq-7b_sliding_window_8.pkl"
INITIAL_MODE_TO_TEST = "attention_ratio"
FEATURE_EXTRACTION_MODE = INITIAL_MODE_TO_TEST
SLIDING_WINDOW_SIZE = 8 
LOOKBACK_LENS_THRESHOLD = 0.4

LLAMA_NUM_LAYERS = 32
LLAMA_NUM_HEADS = 32

# --- PyTorch/CUDA Diagnostics ---
print(Colors.blue("--- PyTorch/CUDA Diagnostics ---"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    if "CUDA_VISIBLE_DEVICES" not in os.environ: print(Colors.yellow("Warning: CUDA_VISIBLE_DEVICES is not set."))
    else: print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.device_count() > 0: print(f"CUDA version: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}, Current: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else: DEVICE = "cpu"; print(Colors.red("No GPUs found by PyTorch. Forcing CPU."))
else: DEVICE = "cpu"; print(Colors.red("CUDA NOT available. Forcing CPU."))
print(f"Targeting device: {Colors.yellow(DEVICE)}")
print(Colors.blue("--- End Diagnostics ---"))

print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_AUTH_TOKEN)

_minimal_chat_for_prefix_eval = [{"role": "user", "content": "Hi"}]
_templated_prompt_before_gen = tokenizer.apply_chat_template(_minimal_chat_for_prefix_eval, tokenize=False, add_generation_prompt=False)
_templated_prompt_with_gen = tokenizer.apply_chat_template(_minimal_chat_for_prefix_eval, tokenize=False, add_generation_prompt=True)
assistant_prefix_str = _templated_prompt_with_gen.replace(_templated_prompt_before_gen, "")
assistant_prefix_tokens = tokenizer.encode(assistant_prefix_str, add_special_tokens=False)
BOT_ASSISTANT_PREFIX_TOKEN_LEN = len(assistant_prefix_tokens)
print(Colors.yellow(f"Determined Assistant Prefix String: '{assistant_prefix_str}' (Length: {BOT_ASSISTANT_PREFIX_TOKEN_LEN} tokens)"))

model_config = AutoConfig.from_pretrained(MODEL_ID, token=HF_AUTH_TOKEN, trust_remote_code=True)
model_config.output_attentions = (INITIAL_MODE_TO_TEST == "attention_ratio")
model_config.output_hidden_states = False 
model_config.return_dict_in_generate = True 
# attn_implementation is set in model_load_args_dict now

print(f"Loading model {MODEL_ID} (config output_attentions={model_config.output_attentions})...")
model = None
model_load_args_dict = {
    "config": model_config, 
    "trust_remote_code": True, 
    "token": HF_AUTH_TOKEN
}
if model_config.output_attentions: # Explicitly set for loading if config needs it
    model_load_args_dict["attn_implementation"] = "eager"

if DEVICE == "cuda":
    try: model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, **model_load_args_dict)
    except Exception as e: print(Colors.red(f"CUDA load error: {e}. Fallback to CPU.")); DEVICE = "cpu"
if DEVICE == "cpu" and model is None:
    model_load_args_dict.pop("torch_dtype", None); model_load_args_dict.pop("attn_implementation", None) # Remove if CPU doesn't use it
    model_load_args_dict["device_map"] = "cpu" 
    try: model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_load_args_dict)
    except Exception as e: print(Colors.red(f"FATAL CPU load error: {e}")); exit()
if model is None: print(Colors.red("FATAL: Model could not be loaded.")); exit()

# Ensure model config reflects the final decision AFTER loading, if any dynamic updates were needed
# (though direct load with attn_implementation="eager" should make this less necessary)
if model.config.output_attentions and getattr(model.config, 'attn_implementation', 'default') != "eager":
    print(Colors.yellow("Warning: Model loaded but attn_implementation might not be 'eager'. Forcing config."))
    model.config.attn_implementation = "eager" # Force it on the loaded model's config too

print(f"Model loaded. Effective config: attentions={model.config.output_attentions}, attn_impl={getattr(model.config, 'attn_implementation', 'N/A')}")

lookback_lens_classifier = None; inferred_num_features = 0
print(f"Attempting to load classifier from: {CLASSIFIER_PATH}")
try:
    with open(CLASSIFIER_PATH, 'rb') as f: loaded_data = pickle.load(f)
    if isinstance(loaded_data, dict) and 'clf' in loaded_data: lookback_lens_classifier = loaded_data['clf']
    elif hasattr(loaded_data, 'predict_proba'): lookback_lens_classifier = loaded_data # Is the classifier itself
    if lookback_lens_classifier:
        if hasattr(lookback_lens_classifier, 'coef_'): inferred_num_features = lookback_lens_classifier.coef_.shape[-1]
        elif hasattr(lookback_lens_classifier, 'n_features_in_'): inferred_num_features = lookback_lens_classifier.n_features_in_
        print(f"Classifier loaded. Inferred features: {inferred_num_features}")
        expected_features = LLAMA_NUM_LAYERS * LLAMA_NUM_HEADS
        if inferred_num_features != expected_features:
            print(Colors.red(f"Warning: Classifier features ({inferred_num_features}) != expected ({expected_features}). Disabling Lookback Lens."))
            lookback_lens_classifier = None 
    else: print(Colors.red("Classifier PKL format not recognized."))
except Exception as e: print(Colors.red(f"Error loading classifier: {e}"))

model.eval()
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None: model.config.pad_token_id = model.config.eos_token_id
print("Model and tokenizer setup complete.")

if lookback_lens_classifier: print(f" {Colors.green('ACTIVE')}, Mode: {Colors.yellow(FEATURE_EXTRACTION_MODE)}")
else: print(f" {Colors.red('INACTIVE')}")
print(f"Threshold: {Colors.yellow(str(LOOKBACK_LENS_THRESHOLD))}, Window Size: {Colors.yellow(str(SLIDING_WINDOW_SIZE))}")

def calculate_feature_vector_for_token(
    model_outputs, prompt_token_len_boundary_N, current_kv_cache_total_len
):
    # (Function definition as in the previous correct version - kept concise here)
    if FEATURE_EXTRACTION_MODE == "attention_ratio":
        if not hasattr(model_outputs, 'attentions') or not model_outputs.attentions: return np.array([])
        all_head_ratios_flat = []
        num_layers_in_output = len(model_outputs.attentions)
        if num_layers_in_output == 0: return np.array([])
        for layer_idx in range(num_layers_in_output):
            layer_att = model_outputs.attentions[layer_idx] 
            if layer_att is None or layer_att.shape[2] != 1: continue 
            squeezed_att_for_layer = layer_att.squeeze(2) 
            current_layer_att_heads = squeezed_att_for_layer[0] 
            num_heads_this_layer = current_layer_att_heads.shape[0]
            kv_len_this_layer = current_layer_att_heads.shape[-1]
            effective_N_boundary = min(prompt_token_len_boundary_N, kv_len_this_layer)
            if effective_N_boundary > 0:
                attn_to_context_mean = current_layer_att_heads[:, :effective_N_boundary].mean(dim=-1)
            else:
                attn_to_context_mean = torch.zeros(num_heads_this_layer, device=current_layer_att_heads.device)
            if effective_N_boundary < kv_len_this_layer:
                attn_to_prev_gen_mean = current_layer_att_heads[:, effective_N_boundary:].mean(dim=-1)
            else: 
                attn_to_prev_gen_mean = torch.zeros(num_heads_this_layer, device=current_layer_att_heads.device)
            denominator = attn_to_context_mean + attn_to_prev_gen_mean + 1e-9
            lr_for_heads = attn_to_context_mean / denominator
            all_head_ratios_flat.extend(lr_for_heads.cpu().to(torch.float32).tolist())
        final_ratios_vector = np.array(all_head_ratios_flat, dtype=np.float32)
        expected_total_features = LLAMA_NUM_LAYERS * LLAMA_NUM_HEADS
        if final_ratios_vector.size != expected_total_features and final_ratios_vector.size > 0 : 
             print(Colors.red(f"[Debug calc_feat] Feature count mismatch! Exp {expected_total_features}, Got {final_ratios_vector.size} (Layers: {num_layers_in_output})"), end=" ")
        return final_ratios_vector
    return np.array([])

print(Colors.blue(f"\n--- Starting Chat (type 'quit' to exit) ---"))
history_chat_format = []
while True:
    # (Chat loop as in the previous correct version - kept concise here)
    try: user_input = input(f"{Colors.GREEN}You: {Colors.ENDC}")
    except KeyboardInterrupt: print("\nExiting..."); break
    if user_input.lower() == 'quit': break
    current_prompt_chat_format = history_chat_format + [{"role": "user", "content": user_input}]
    prompt_text_before_assistant_prefix = tokenizer.apply_chat_template(current_prompt_chat_format, tokenize=False, add_generation_prompt=False)
    context_x_inputs = tokenizer(prompt_text_before_assistant_prefix, return_tensors="pt")
    prompt_token_len_boundary_N = context_x_inputs.input_ids.shape[1]
    prompt_text_for_model_input = tokenizer.apply_chat_template(current_prompt_chat_format, tokenize=False, add_generation_prompt=True)
    inputs_for_model = tokenizer(prompt_text_for_model_input, return_tensors="pt", return_attention_mask=True)
    actual_total_input_len_to_model = inputs_for_model.input_ids.shape[1]
    loop_input_ids_cpu = inputs_for_model.input_ids.cpu()
    loop_attention_mask_cpu = inputs_for_model.attention_mask.cpu()
    print(f"{Colors.BLUE}Bot: {Colors.ENDC}", end="", flush=True)
    all_generated_ids = []; all_per_token_feature_vectors = []
    past_key_values = None; max_new_tokens = 256
    min_factual_prob_this_turn = 1.0; hallu_windows_count_this_turn = 0
    len_last_streamed_text = 0
    current_kv_cache_len = actual_total_input_len_to_model
    for step in range(max_new_tokens):
        input_device = model.device 
        effective_input_ids = loop_input_ids_cpu.to(input_device)
        effective_attention_mask = loop_attention_mask_cpu.to(input_device)
        with torch.no_grad():
            outputs = model(input_ids=effective_input_ids, attention_mask=effective_attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        all_generated_ids.append(next_token_id.item())
        if lookback_lens_classifier:
            feature_vec = calculate_feature_vector_for_token(outputs, prompt_token_len_boundary_N, current_kv_cache_len)
            if feature_vec.size > 0: all_per_token_feature_vectors.append(feature_vec)
        current_full_text = tokenizer.decode(all_generated_ids, skip_special_tokens=True)
        newly_decoded_text = current_full_text[len_last_streamed_text:]
        print(newly_decoded_text, end="", flush=True)
        len_last_streamed_text = len(current_full_text)
        if next_token_id.item() == tokenizer.eos_token_id: break
        loop_input_ids_cpu = next_token_id.cpu()
        loop_attention_mask_cpu = torch.cat([loop_attention_mask_cpu, torch.ones((1,1),dtype=torch.long,device='cpu')],dim=-1)
        past_key_values = outputs.past_key_values
        current_kv_cache_len += 1
        if lookback_lens_classifier and len(all_per_token_feature_vectors) >= SLIDING_WINDOW_SIZE:
            window_vectors = all_per_token_feature_vectors[-SLIDING_WINDOW_SIZE:]
            if inferred_num_features > 0 and len(window_vectors) == SLIDING_WINDOW_SIZE and \
               all(isinstance(v, np.ndarray) and v.shape == (inferred_num_features,) for v in window_vectors):
                try:
                    window_feat_mean = np.mean(np.stack(window_vectors), axis=0).reshape(1, -1)
                    prob_factual = lookback_lens_classifier.predict_proba(window_feat_mean)[0, 1]
                    min_factual_prob_this_turn = min(min_factual_prob_this_turn, prob_factual)
                    if prob_factual < LOOKBACK_LENS_THRESHOLD: hallu_windows_count_this_turn += 1
                except Exception as e: pass
    print() 
    final_response = tokenizer.decode(all_generated_ids, skip_special_tokens=True)
    if lookback_lens_classifier and all_generated_ids:
        is_significant_hallucination = ((hallu_windows_count_this_turn > max(0, SLIDING_WINDOW_SIZE // 3)) or \
                                       (min_factual_prob_this_turn < (LOOKBACK_LENS_THRESHOLD - 0.20))) and \
                                       hallu_windows_count_this_turn > 0
        if is_significant_hallucination: 
            print(Colors.red(f" Deviation (Min P(Fact): {min_factual_prob_this_turn:.2f}, Flags: {hallu_windows_count_this_turn})."))
        elif hallu_windows_count_this_turn > 0: 
            print(Colors.yellow(f" Some deviation (Min P(Fact): {min_factual_prob_this_turn:.2f}, Flags: {hallu_windows_count_this_turn})."))
        else:
            if min_factual_prob_this_turn > (LOOKBACK_LENS_THRESHOLD + 0.2): 
                print(Colors.green(f" Grounded (Min P(Fact): {min_factual_prob_this_turn:.2f})."))
            else: 
                print(f" (Min P(Fact): {min_factual_prob_this_turn:.2f}, Flags: {hallu_windows_count_this_turn}).")
    history_chat_format.append({"role": "user", "content": user_input})
    history_chat_format.append({"role": "assistant", "content": final_response})
    if len(history_chat_format) > 10: history_chat_format = history_chat_format[-10:]
print(Colors.blue("--- Chat Ended ---"))