from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_S.gguf")


def get_prompt(instruction: str) -> str:
    system = "You are a gen z ai assistant. You are witty and sarcastic, but still get the job done."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


question = "Tell me about our schedule."

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
print()
