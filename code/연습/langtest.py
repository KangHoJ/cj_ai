from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("re2panda/polyglot_12B_click_bate_test_sample")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-12.8b")
model = PeftModel.from_pretrained(model, "re2panda/polyglot_12B_click_bate_test_sample")