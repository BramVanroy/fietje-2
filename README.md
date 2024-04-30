<p align="center">
  <img src="img/logo/fietje-2b-banner-rounded.png" alt="Banner image Fietje" style="display: block; max-width: 480px; width: auto; height: auto;">
</p>

<div style="margin:auto; text-align:center" align="center">
<h1 style="margin-bottom: 0">Fietje 2B</h1>
<em>An open and efficient LLM for Dutch.</em>
</div>

## How to use

How to use the final chat model. Also references to the cpt and sft models but emphasize that those are not recommended.

add Modelfile for final GGUF

### Hugging Face interface

You can use Fietje 2B Chat freely in your browser through the [Hugging Face Space](https://huggingface.co/spaces/BramVanroy/fietje-2b).

### Local interface: LM Studio

The easiest way to get started with Fietje locally, is by using it through a chat interface like [LM Studio](https://lmstudio.ai/), which is a beginner-friendly program to run LLMs on your own device.

1. Download and install [LM Studio](https://lmstudio.ai/).
2. In the sidebar, click on the magnify glass and then search for `bramvanroy` (no space) in the search bar.
3. Click on the model that you want to download. For the chat model, this is `fietje-2b-chat-gguf`. Then select an appropriate quantization method under "Available Files". For starters you can select the most efficient version, which is `q5_k_m`.

<p align="center">
  <img src="img/lm-studio/download.png"  style="display: block; max-width: 1570px; width: 100%; height: auto;">
</p>

4. When the download has finished, click on the chat icon in the sidebar. 
5. At the top center, select `fietje chat`  and wait for it to load.
6. Start using the interface by talking to the model!

<p align="center">
  <img src="img/lm-studio/usage.png"  style="display: block; max-width: 1570px; width: 100%; height: auto;">
</p>

### Command-line: ollama



### Python

Rather than using an interface or ollama via the command-line, you can also plainly use the model from good ol' `transformers`.

One-off generation

```python
from transformers import pipeline, Conversation

# load_in_8bit: lower precision but saves a lot of GPU memory
# attn_implementation: uses flash attention, if your device supports it - otherwise remove it
# device_map=auto: loads the model across multiple GPUs
chatbot = pipeline(
    "conversational",
    model="BramVanroy/fietje-2b-chat",
    model_kwargs={"load_in_8bit": True, "attn_implementation": "flash_attention_2"},
    device_map="auto"
)

start_messages = [
    {"role": "system", "content": "Je bent een grappige chatbot die Bert heet. Je maakt vaak mopjes."},
    {"role": "user", "content": "Hallo, ik ben Bram. Ik wil vanavond graag een film kijken. Heb je enkele suggesties?"}
]
conversation = Conversation(start_messages)
conversation = chatbot(conversation)
response = conversation.messages[-1]["content"]
print(response)
```

Interactive conversation:

```python
from transformers import pipeline, Conversation

# load_in_8bit: lower precision but saves a lot of memory
# attn_implementation: uses flash attention, if your device supports it - otherwise remove it
# device_map=auto: loads the model across multiple GPUs
chatbot = pipeline(
    "conversational",
    model="BramVanroy/fietje-2b-dpo-lr2.0e-6-beta0.2-gradaccum2-v6",
    model_kwargs={"load_in_8bit": True, "attn_implementation": "flash_attention_2"},
    device_map="auto"
)

while (system_message := input("System message ('q' to quit): ")) != "q":
    start_messages = [
        {"role": "system", "content": system_message},
    ]
    conversation = Conversation(start_messages)
    while (user_input := input("User ('r' to reset): ")) != "r":
        conversation.add_user_input(user_input)
        conversation = chatbot(conversation)
        response = conversation.messages[-1]["content"]
        print("Assistant:", response)
```

If that's still not fast enough, you can use [vLLM](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py) for optimized inference, which is much faster.

TODO: add chat template
```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="BramVanroy/fietje-2b-dpo-lr2.0e-6-beta0.2-gradaccum2-v6")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Performance

Benchmark results



### Training data

28B tokens from wikipedia and culturax, extra filtered

### Training methodology

Alignment handbook; add recipes to github repo

## Thanks

- Edwin Rijgersberg
- David Berenstein, Argilla
- Hugging Face, esp. `trl` and `alignment-handbook` teams
- Dutch NLP Community
- Michiel Buisman (UWV), Maarten Lenz-Fitzgerald (SVB) - project leesplank