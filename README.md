This repository contains our bronze medal solution for the AI Mathematical Olympiad Progress Prize 2 competition, featuring an optimized mathematical reasoning model that uses significantly fewer tokens while maintaining high accuracy.
Overview
We fine-tuned the DeepSeek-R1-Distill-Qwen models using the methodology from "Training Language Models to Reason Efficiently" (Arora & Zanette, 2025) to create models that:

Dynamically adapt reasoning length based on problem complexity
Reduce token generation by up to 50% on simpler problems
Maintain competitive accuracy on difficult mathematical olympiad problems

Model Details
Our primary model is a fine-tuned version of DeepSeek-R1-Distill-Qwen-7B optimized specifically for efficient reasoning. The model was trained using reinforcement learning with a modified objective function that rewards shorter correct responses.
Key characteristics:

Base architecture: Qwen2ForCausalLM with 7B parameters
Quantization: AWQ with Marlin kernels for efficient deployment
Context length: Supports up to 32K tokens
Tuning parameter α: Controls the tradeoff between solution length and accuracy

Getting Started
Requirements
vllm==0.7.1
torch
transformers
Usage
pythonfrom vllm import LLM, SamplingParams

# Load the model
model_path = "path/to/deepseek-efficient-reasoning-model"
llm = LLM(
    model_path,
    max_num_seqs=16,
    max_model_len=16384,
    trust_remote_code=True,
    tensor_parallel_size=4,  # Adjust based on your GPU setup
    gpu_memory_utilization=0.95,
    seed=47,
)
tokenizer = llm.get_tokenizer()

# Create prompt with reasoning instruction
prompt = "Please reason step by step, and put your final answer within \\boxed{}. Question: What is the sum of the first 100 positive integers?"

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
sampling_params = SamplingParams(
    temperature=0.7,
    min_p=0.01,
    skip_special_tokens=True,
    max_tokens=16384,
)

outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
Simplified Inference Code
The core components of our solution include:

Model Loading: Initializing the quantized model with appropriate parameters
Answer Extraction: Parsing the model outputs to extract final answers
Batched Inference: Efficient processing of multiple problems simultaneously

```
def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

def predict_for_question(question: str) -> int:
    # Create prompts with different system instructions for ensemble
    list_of_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
            {"role": "user", "content": question},
        ]
        for _ in range(16)  # Generate multiple samples
    ]
    
    # Generate responses
    list_of_messages = batch_message_generate(list_of_messages)
    
    # Extract answers
    extracted_answers = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
    
    # Select final answer based on majority voting
    answer = select_answer(extracted_answers)
    return answer
```

Competition Results
Our model achieved bronze medal performance in the AIMO Progress Prize 2 competition. Performance across different mathematical benchmarks:

GSM8K: ~93% accuracy with up to 50% token reduction
MATH500: ~92% accuracy with ~30% token reduction
AIME2024: ~54% accuracy with ~16% token reduction

Citation
@article{guo2025deepseekr1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={Guo, Damai and Yang, Daniel and Zhang, Hongyi and Song, Jiangtao and Zhang, Renjie and Xu, Ruilong and Zhu, Qi and Ma, Shuai and Wang, Peng and Bi, Xiaohan and others},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}

@article{arora2025training,
  title={Training Language Models to Reason Efficiently},
  author={Arora, Daman and Zanette, Andrea},
  journal={arXiv preprint arXiv:2502.04463v2},
  year={2025}
}
