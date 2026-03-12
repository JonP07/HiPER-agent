<h3 align="center">
<b>HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents
</b>
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2602.16165">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://jonp07.notion.site/HiPER-Hierarchical-Plan-Execute-Reinforcement-Learning-for-Multi-turn-LLM-Agents-314911747a7e801b86e7eee6187d7cd5?source=copy_link">
    <img src="https://img.shields.io/badge/Notion-Blog-blue?logo=notion" alt="Notion Blog"></a>
</p>

HiPER is a hierarchical reinforcement learning framework for training large language model agents in long-horizon environments. Instead of treating agent behavior as a flat sequence of actions, HiPER explicitly separates high-level planning from low-level execution, and introduces Hierarchical Advantage Estimation (HAE) for more effective credit assignment across multiple time scales. This repository builds on <a href="https://github.com/langfengQ/verl-agent">verl-agent</a>, with extensions to both the agent interface and the training algorithm. [Our webpage is still under construction, check back for updates!]

## Results Summary
<p align="center">
    <img src="header_plot.png">
</p>

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Task</th>
      <th>Model</th>
      <th>Success Rate</th>
      <th>Training Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>HiPER</b></td>
      <td>ALFWorld</td>
      <td>Qwen2.5-1.5B-Instruct</td>
      <td>95.3%</td>
      <td><a href="https://api.wandb.ai/links/mhong-university-of-minnesota/6sbbne68"><img src="https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb" alt="wandb"></a></td>
    </tr>
    <tr>
      <td><b>HiPER</b></td>
      <td>ALFWorld</td>
      <td>Qwen2.5-7B-Instruct</td>
      <td>97.4%</td>
      <td><a href="https://api.wandb.ai/links/mhong-university-of-minnesota/6sbbne68"><img src="https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb" alt="wandb"></a></td>
    </tr>
    <tr>
      <td><b>HiPER</b></td>
      <td>WebShop</td>
      <td>Qwen2.5-1.5B-Instruct</td>
      <td>71.4%</td>
      <td><a href="https://api.wandb.ai/links/mhong-university-of-minnesota/uwifhx6v"><img src="https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb" alt="wandb"></a></td>
    </tr>
    <tr>
      <td><b>HiPER</b></td>
      <td>WebShop</td>
      <td>Qwen2.5-7B-Instruct</td>
      <td>83.3%</td>
      <td><a href="https://api.wandb.ai/links/mhong-university-of-minnesota/uwifhx6v"><img src="https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb" alt="wandb"></a></td>
    </tr>
  </tbody>
</table>



## Installation
### veRL
```bash
conda create -n verl python==3.12 -y
conda activate verl

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip3 install -e .
pip3 install vllm==0.8.5
pip3 install peft==0.17.1
```

### ALFWorld
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
alfworld-download -f
```

### WebShop
To avoid conflict, it is recommended to install ALFWorld and WebShop separately in two conda environments (e.g. verl-alfworld and verl-webshop). Note that WebShop requires Python version <=3.10

```bash
conda create -n verl-webshop python==3.10 -y
conda activate verl-webshop
```

```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

```bash
cd repo_root/ # replace with your repository root path
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
pip3 install peft==0.17.1
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# The above warnings can be ignored.
```

## HiPER Example Scripts
```bash
bash example_scripts/HiPER_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash example_scripts/HiPER_trainer/run_webshop.sh # WebShop
```

# Citation
If you find HiPER helpful, please cite our paper below:
```
@article{peng2026hiper,
  title={HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents},
  author={Peng, Jiangweizhi and Liu, Yuanxin and Zhou, Ruida and Fleming, Charles and Wang, Zhaoran and Garcia, Alfredo and Hong, Mingyi},
  journal={arXiv preprint arXiv:2602.16165},
  year={2026}
}
```


## Acknowledgement
Our codebase is built upon <a href="https://github.com/langfengQ/verl-agent">verl-agent</a> and <a href="https://github.com/volcengine/verl">veRL</a>. The environments are adapted from [ALFWorld](https://github.com/alfworld/alfworld) and 
[WebShop](https://github.com/princeton-nlp/WebShop). We sincerely thank the authors and contributors of these projects for making their valuable work publicly available.