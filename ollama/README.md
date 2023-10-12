# Steps to getting started

## Installation of Ollama

### For Mac

1. Install [Ollama](https://github.com/jmorganca/ollama) via this link: [download](https://github.com/jmorganca/ollama).
2. Run the installation script.

### For Linux

1. Install [Ollama](https://github.com/jmorganca/ollama#linux--wsl2).

    ```sh
    curl https://ollama.ai/install.sh | sh
    ```

### For Windows

1. Use WSL. Ollama is not yet compiled for Windows.

## Model Sizes

To run these models, you'll minimally need memory at least of these sizes, if not more.
After verifying that your model can run locally, proceed with the Python instructions.

| Model	Parameters | Size | Download |
|--|--|--|
| Mistral	7B | 4.1GB | `ollama run mistral`
| Llama 2	7B | 3.8GB | `ollama run llama2`
| Code Llama	7B | 3.8GB | `ollama run codellama`
| Llama 2 Uncensored	7B | 3.8GB | `ollama run llama2-uncensored`
| Llama 2 13B	13B | 7.3GB | `ollama run llama2:13b`
| Llama 2 70B	70B | 39GB | `ollama run llama2:70b`
| Orca Mini	3B | 1.9GB | `ollama run orca-mini`
| Vicuna	7B | 3.8GB | `ollama run vicuna`

## Python setup

To set up, follow these steps:

1. Set up your virtual environment:

    ```sh
    python -m venv <VENV NAME>
    ```
  
2. Activate it.

    ```sh
    source <VENV NAME>/bin/activate
    ```

3. Run the installation of requirements.

    ```sh
    python3 -m pip install -r requirements.txt
    ```

4. Inspect `main.py`. At the top, you'll find a base model config.
  `llama2` should work for most base machines. If you wish to add more models,
  simply add them to the config, keeping in mind:

    a. The model name. It has to be supported by ollama, and the model has to fit 
      within your machine's RAM.
    b. The prompt chain method. It has to be available via Langchain Hub, and tuned
      for the model. For instance, for Mistral 7B, it is 'rlm/rag-prompt-mistral'.

    c. The embeddings method. For my machine, it can only run the GGML MiniLM embeddings method within reasonable time, and Ollama llama 7b takes considerably
    longer. Your milage may vary.

5. Run `main.py` in your virtual environment. You may wish to export it to a notebook to keep the loading libraries in memory, speeding up subsequent inference.
