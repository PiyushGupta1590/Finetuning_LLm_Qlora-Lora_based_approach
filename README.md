# Finetuning_LLm_Qlora-Lora_based_approach
Mental Health Chatbot: Fine-tuning Phi-2 with QLoRA
A parameter-efficient fine-tuning project that adapts Microsoft's Phi-2 language model for empathetic mental health conversation support using Quantized Low-Rank Adaptation (QLoRA).

This project fine-tunes the Phi-2 (2.7B parameters) model on the MentalChat16K dataset to create a supportive conversational AI for mental health contexts. Using QLoRA, we achieve efficient training with 75% less memory while maintaining high-quality outputs.

Parameter-Efficient Training: Uses QLoRA (4-bit quantization + LoRA) for memory-efficient fine-tuning
Mental Health Focus: Trained on 16K mental health conversations for empathetic responses
Safety-Aware: Model learns to suggest professional help and avoid medical diagnoses
Quantitative Evaluation: Includes ROUGE metric evaluation pipeline
Qualitative Testing: Sample test cases for assessing response quality

Requirements
Hardware

GPU with 8GB+ VRAM (recommended: NVIDIA A100, V100, or T4)
20GB+ disk space
16GB+ RAM

Software

Python 3.9+
CUDA 11.8+
conda or virtualenv
Installation
1.Clone the Repo
git clone <your-repo-url>
cd llm-finetuning
2.Create Environment
3. Install Dependencies
bash# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip install transformers peft accelerate bitsandbytes datasets scipy einops evaluate trl rouge_score pandas numpy huggingface_hub
4. Authenticate with Hugging Face
Train in Background 
