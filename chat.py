from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_S.gguf")


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are a gen z ai assistant. You are witty and sarcastic, but still get the job done."
    prompt = f"###System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    # prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


history = []
question = "Tell me about our schedule."
answer = ""
for word in llm(get_prompt(question), stream=True):
    answer += word
    print(word, end="", flush=True)
print()
history.append(answer)

question = "And of the United States?"
for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
