from wde import LLM

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
llm = LLM(model="FacebookAI/xlm-roberta-base")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.encode(prompts)
# Print the outputs.
for output in outputs:
    print(output.outputs.shape)
