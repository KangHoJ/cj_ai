import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

model_id = "EleutherAI/polyglot-ko-12.8b"
peft_model_id = "re2panda/polyglot_12B_click_bate_test_sample"

config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
# model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=peft_model_id,
    tokenizer=tokenizer,
    # device=2,
)


def answer(state, state_chatbot, text):
    messages = state + [{"role": "질문", "content": text}]

    conversation_history = "\n".join(
        [f"### {msg['role']}:\n{msg['content']}" for msg in messages]
    )

    ans = pipe(
        conversation_history + "\n\n### 답변:",
        do_sample=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )

    msg = ans[0]["generated_text"]

    if "###" in msg:
        msg = msg.split("###")[0]

    new_state = [{"role": "privious question", "content": text}, {"role": "privious answer", "content": msg}]

    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]

    print(state)
    print(state_chatbot)

    return state, state_chatbot, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            {
                "role": "맥락",
                "content": "해당 모델은 Polyglot 12b를 활용하여 Open AI의 grade-school-math data를 기반으로 학습한 모델입니다.",
            },
            {
                "role": "맥락",
                "content": "Gradio는 Koalpaca의 app.py를 참조하였습니다.",
            },
            {"role": "명령어", "content": "친절한 AI 챗봇인 ChatKoAlpaca 로서 답변을 합니다."},
            {
                "role": "명령어",
                "content": "인사에는 짧고 간단한 친절한 인사로 답하고, 아래 대화에 간단하고 짧게 답해주세요.",
            },
        ]
    )
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>🐟 Click Bate(polyglot_12.8b) 🐟</h1>
            </div>
            <div>
                Demo runs on A100 GPU with 4bit quantized
            </div>
        </div>"""
        )

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...").style(
            container=False
        )

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

demo.launch(debug=True,share= True, server_name="0.0.0.0")
