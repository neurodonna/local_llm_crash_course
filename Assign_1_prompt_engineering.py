from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

prompt = "I want just the phrased answer. No long paragraphs or sentences. What is the capital of India?"

print(llm(prompt))
