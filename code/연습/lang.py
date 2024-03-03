from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

text_generation_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0,  # CPU 사용을 나타냅니다.
)

def generate_text(prompt, max_length=200):
    generated_sequences = text_generation_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_length=max_length,
    )
 
    return generated_sequences[0]["generated_text"].replace(prompt, "")

# 텍스트 생성 테스트
prompt = "사과는"
generated_text = generate_text(prompt, max_length=200)
print(generated_text)