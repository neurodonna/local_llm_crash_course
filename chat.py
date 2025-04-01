import chainlit as cl
# from typing import List
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are a gen z ai assistant. You are witty and sarcastic, but still get the job done."
    prompt = f"###System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    prompt = get_prompt(message.content)
    response = llm(prompt)
    await cl.Message(response).send()


@cl.on_chat_start
async def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
    await cl.Message("Model initialized. How can I help you?").send()


'''
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
'''
