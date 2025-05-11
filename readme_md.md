# Contextual Hallucination Detecting Trainable LLM Tool With Local CLI Transformers

## Introduction 

This project implements a real-time system for detecting potential hallucinations or deviations from context in LLM responses using attention pattern analysis. The system uses a sliding window approach to monitor the model's attention to context vs. previously generated tokens.

## Acknowledgements / Basis of this Work

This project implements and extends the methods presented in the paper:

*   Chuang, Y.-S., Qiu, L., Hsieh, C.-Y., Krishna, R., Kim, Y., & Glass, J. (2024). *Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps*. arXiv preprint arXiv:2407.07071. Available at: [https://arxiv.org/abs/2407.07071](https://arxiv.org/abs/2407.07071)

Original implementation provided in the following repository:

*   [Yung-Sung Chuang] ([Year of software version]). *[Lookback-Lens]*. Version [e.g., v1.0 or commit hash]. Retrieved from [\[URL of their GitHub repository\]](https://github.com/voidism/Lookback-Lens/)


## Features

- Real-time hallucination detection during text generation
- Colored console output indicating potential issues in responses
- Support for Llama 2 models with Hugging Face integration
- GPU acceleration for faster inference
- Sliding window attention pattern analysis

## Getting Started

### Prerequisites

- Python 3.8+ 
- CUDA-capable GPU (recommended, but will fall back to CPU)
- Hugging Face account with access to Llama 2 models

### Installation

1. Clone this repository:

```bash
git clone https://github.com/emrehannn/hallucination-detecting-llm-tool
cd lookback-lens
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Setting up Hugging Face Access

1. Create a Hugging Face account at [https://huggingface.co/join](https://huggingface.co/join) if you don't have one

2. Request access to Meta's Llama 2 model at [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) by clicking "Access repository"

3. Generate a Hugging Face API token:
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token (read access is sufficient)
   - Copy the token

4. Edit the script to include your token:
   - Open '.env' in the root directory
   - Replace `HF_AUTH_TOKEN = "your token"` with your token

### Setting up the Classifier

1. Create a directory for classifiers:

```bash
mkdir -p classifiers
```

2. Download or train your own classifier. The default configuration expects:
   - Filename: `classifier_anno-cnndm-7b_sliding_window_8.pkl`
   - Path: `classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl`

If using a different classifier, update the `CLASSIFIER_PATH` variable in the script.

## Usage

Run the main script:

```bash
python main.py
```

The application will:
1. Check for CUDA availability
2. Load the Llama 2 model and tokenizer
3. Load the Lookback Lens classifier
4. Start an interactive chat session

I coded this on Windows 11, so sadly no bitsandbytes quantization. But if you are using Linux/OS X, it is only 1 line of change to implement quantization.

During the chat:
- Type your message and press Enter
- The model will respond with real-time token generation
- Lookback Lens will analyze attention patterns and display:
  - 🟢 Green: Response seems contextually grounded
  - 🟡 Yellow: Some deviation from context observed
  - 🔴 Red: Potential significant deviation from context

Type `quit` to exit the chat.

## Configuration

You can adjust these variables in the script:

- `MODEL_ID`: Change the model (default: "meta-llama/Llama-2-7b-chat-hf")
- `LOOKBACK_LENS_THRESHOLD`: Adjust sensitivity (default: 0.3, lower is more sensitive)
- `SLIDING_WINDOW_SIZE`: Change window size for analysis (default: 8)
- `max_new_tokens`: Maximum tokens to generate per response (default: 100)

## Training Your Own Classifier

The included classifier was trained on annotated factual vs. hallucinated text using attention patterns as features. To train your own:

1. Collect a dataset of factual and non-factual model outputs with attention data
2. Extract attention-to-context ratio patterns
3. Train a classifier (such as RandomForest or GradientBoosting)
4. Save using joblib with format: `{'clf': classifier_object, 'best_threshold': threshold_value}`

## OR

1. You can use the "feedback: correct/false" tags to create your own .pk1



## Technical Details

### Attention Analysis

The system calculates the ratio of attention paid to the user's context versus previously generated tokens for each head in each layer. This creates a feature vector that the classifier uses to identify potential hallucinations.

### Sliding Window Approach

Rather than analyzing individual tokens, the system uses a sliding window to examine patterns across multiple tokens, which improves detection reliability.

## License

MIT License


