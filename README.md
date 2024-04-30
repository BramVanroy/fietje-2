<p align="center">
  <img src="img/logo/fietje-2b-banner-rounded.png" alt="Banner image Fietje" style="display: block; max-width: 480px; width: auto; height: auto;">
</p>

<div style="margin:auto; text-align:center" align="center">
<h1 style="margin-bottom: 0">Fietje 2B</h1>
<em>An open and efficient LLM for Dutch.</em>
</div>

## Performance

Despite its small size, Fietje keeps up with other state-of-the-art models adapted for Dutch that are more than twice its size.

The results in this table have been calculated with [ScandEval](https://github.com/ScandEval/ScandEval) (v12.6.1), which automatically runs each benchmark ten times to give a more precise indication of system's performance. Confidence intervals are not reported here so the table below gives a limited view of nuances between models. Furthermore, benchmarks for generative models are inherently flawed. It is hard to capture "quality" of a model - it will always depend on your task and your prompt. Don't trust leaderboards - try out your task with a limited dataset to get a better idea which model works for you!

The important takeaway is that **Fietje punches above its weight class** when it comes to these benchmarks. And that is its goal: to be powerful but efficient!

Full results, including confidence interval and other metrics, will be added to the [ScandEval leaderboard](https://scandeval.com/dutch-nlg/) soon. For now, you can find the raw results (including other models not reported in the table) in [evaluation/scandeval_benchmark_results.jsonl](evaluation/scandeval_benchmark_results.jsonl).


|       model         	        | dutch-social<br>(macro f1) 	 | conll-nl<br>(micro f1) 	 | scala-nl<br>(macro f1) 	 | squad-nl<br>(f1) 	 | wiki-lingua-nl<br>(bertscore) 	 | mmlu-nl<br>(accuracy) 	 | hellaswag-nl<br>(accuracy) 	 | **average** 	|
|:----------------------------:|:----------------------------:|:------------------------:|:------------------------:|:------------------:|:-------------------------------:|:-----------------------:|:----------------------------:|:-----------:	|
|   GEITje-7B-ultra        	   |   **42.30**             	    |   26.26             	    |   50.33             	    |   66.47       	    |   **68.32**                	    |   44.52            	    |   **43.78**             	    | **48.85**   	|
|   GEITje-7B-chat-v2      	   |   40.13                 	    |   31.16             	    |   49.59             	    |   70.19       	    |   65.57                    	    |   44.92            	    |   36.76                 	    | 48.33       	|
| **fietje-2b-chat**         	 |   39.92                 	    |   31.81             	    |   50.99             	    |   71.03       	    |   65.37                    	    |   44.86            	    |   32.71                 	    | 48.10       	|
|   GEITje-7B              	   |   28.11                 	    |   30.04             	    |   **63.76**         	    |   67.54       	    |   66.17                    	    |   44.44            	    |   31.80                 	    | 47.41       	|
| **fietje-2b-instruct**     	 |   40.77                 	    |   28.73             	    |   43.19             	    |   **71.62**   	    |   66.01                    	    |   **44.94**        	    |   34.12                 	    | 47.05       	|
|   GEITje-7B-chat         	   |   27.53                 	    |   36.05             	    |   58.93             	    |   66.72       	    |   66.86                    	    |   42.25            	    |   30.85                 	    | 47.03       	|
|   GEITje-7B-ultra-sft    	   |   35.05                 	    |   30.71             	    |   52.32             	    |   65.67       	    |   67.71                    	    |   42.92            	    |   33.09                 	    | 46.78       	|
|   GEITje-7B-ultra-v2     	   |   31.46                 	    |   32.70             	    |   49.80             	    |   65.49       	    |   67.61                    	    |   42.46            	    |   31.03                 	    | 45.79       	|
| **fietje-2b**              	 |   41.03                 	    |   28.63             	    |   41.28             	    |   69.39       	    |   61.49                    	    |   42.68            	    |   27.19                 	    | 44.53       	|
|   Phi-3-mini-4k-instruct 	   |   29.23                 	    |   **42.76**         	    |   50.26             	    |   48.39       	    |   57.17                    	    |   40.28            	    |   34.69                 	    | 43.26       	|
|   phi-2                  	   |   29.30                 	    |   31.52             	    |   38.18             	    |   36.54       	    |   59.26                    	    |   31.98            	    |   25.71                 	    | 36.07       	|
|   gpt2-medium-dutch      	   |   10.30                 	    |   0.33              	    |   45.08             	    |   1.69        	    |   34.01                    	    |   24.76            	    |   23.61                 	    | 19.97       	|




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

All models can be used through the command-line interface [`ollama`](https://github.com/ollama/ollama). This is a more advanced way to interact with the model, but it is also more efficient.

Once you have installed `ollama`, you can easily run the base, instruct, or chat model. You have the option to use a larger `f16` model or a quantized, smaller `q5_k_m` version. The first is larger and likely yields better quality, but the second is much smaller and faster. `q5_k_m` is the default.

All models are available on [my ollama overview page](https://ollama.com/bramvanroy).

```shell
# First make sure that the ollama service is running in the background (e.g., via `screen` or a different terminal)
ollama serve

# Then start the chat model. Download will happen automatically
ollama run bramvanroy/fietje-2b-chat:q5_k_m
```

Example:

<p align="center">
  <img src="img/ollama.png">
</p>

### Python

Rather than using an interface or ollama via the command-line, you can also plainly use the model from good ol' `transformers`.

Here is an example of how to use the model in Python for a one-off generation to create a DnD character. Note that the `pipeline` automatically handles the correct formatting of the conversation according to the required chat template.

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
    {"role": "user", "content": "Maak een nieuw DnD personage met de naam 'Bram'. Geef een beschrijving, de skills, en de extra 'traits'' met het aantal punten per vaardigheid. Gebruik JSON. Geef geen extra uitleg."}
]
conversation = Conversation(start_messages)
conversation = chatbot(conversation)
response = conversation.messages[-1]["content"]
print(response)
"""
{
  "naam": "Bram",
  "beschrijving": "Een onverschrokken avonturier met een hart van goud, Bram is een man van weinig woorden, maar met een onuitputtelijke moed. Zijn leven is gevuld met verhalen van overwinningen en verliezen, maar hij draagt ze met een glimlach, wetende dat elke ervaring hem sterker heeft gemaakt."
  "vaardigheden": {
    "wapens": {
      "dolk": 10,
      "zwaard": 8,
      "boog": 6
    },
    "magie": {
      "schaduw": 5,
      "vuur": 4,
      "water": 3
    }
  },
  "extra_traits": {
    "moed": 10,
    "vindingrijkheid": 8,
    "doorzettingsvermogen": 7
  }
}
"""
```

A more elaborate approach is to create a minimalistic chat environment in Python. This is similar to what `ollama` offers, but slower.

```python
from transformers import pipeline, Conversation

# load_in_8bit: lower precision but saves a lot of memory
# attn_implementation: uses flash attention, if your device supports it - otherwise remove it
# device_map=auto: loads the model across multiple GPUs
chatbot = pipeline(
    "conversational",
    model="BramVanroy/fietje-2b-chat",
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