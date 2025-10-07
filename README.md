# Doctor-R1: An AI Doctor Agent with Strategic Inquiry and Decision-Making

[![Paper](https://img.shields.io/badge/Paper-arXiv:2510.04284-b31b1b.svg)](https://arxiv.org/abs/2510.04284) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Coming_Soon-yellow.svg)](https://huggingface.co/YourOrg/Doctor-R1) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **Note:** The official code and model weights are currently being prepared for public release. Please **star ⭐ and watch 👀 this repository** to be notified when they are available!

**Doctor-R1** is an AI doctor agent trained to conduct strategic, multi-turn patient inquiries to guide its diagnostic decision-making. Unlike traditional models that excel at static medical QA, Doctor-R1 is designed to master the complete, dynamic consultation process, unifying the two core skills of a human physician: communication and clinical reasoning.

![](assets/methodology.jpg)
## 📖 Abstract

The proficiency of a human physician is defined by two core abilities: making accurate medical decisions and conducting strategic, empathetic patient inquiries. While Large Language Models (LLMs) have achieved remarkable accuracy on static medical benchmarks, they often lack the crucial inquiry skills essential for real-world clinical scenarios. To address this gap, we propose Doctor-R1, an AI doctor agent trained to master both capabilities. Our framework, based on **Experiential Agentic Reinforcement Learning**, introduces a multi-agent interactive environment, a two-tiered reward architecture that separately optimizes clinical and communicative skills, and an experience repository to ground policy learning. Evaluations on HealthBench and MAQuE show that Doctor-R1 surpasses state-of-the-art open-source models and outperforms several proprietary models, with human evaluations confirming a strong preference for its ability to generate human-preferred clinical dialogue.

## ✨ Key Features

* **Unified Clinical Skills:** The first agent framework to holistically integrate two core clinical skills, **strategic patient inquiry** and **accurate medical decision-making** within a single model.
* **Experiential Reinforcement Learning:** A novel closed-loop framework where the agent learns and improves from an accumulating repository of its own high-quality experiences.
* **Dual-Competency Reward System:** A sophisticated two-tiered reward architecture that separately optimizes for both conversational quality (soft skills) and diagnostic accuracy (hard skills), featuring a "safety-first" veto system.
* **State-of-the-Art Performance:** Outperforms leading open-source models on challenging dynamic benchmarks like HealthBench and MAQuE with high parameter efficiency.

## 🚀 Code and Model Release (Coming Soon!)

We are in the process of cleaning up and packaging the code and model weights for a public release. We plan to make the following assets available in this repository shortly:

-   [ ] The full source code for our **Experiential Agentic Reinforcement Learning** framework.
-   [ ] Training configurations and evaluation scripts to reproduce our key results.
-   [ ] The final **Doctor-R1** model weights, which will be hosted on the Hugging Face Hub.

Thank you for your interest and patience!

## 📊 Evaluation

Doctor-R1 establishes a new state-of-the-art for open-source medical consultation agents. It demonstrates superior performance on dynamic benchmarks and strong foundational knowledge on static QA tasks.

| Benchmark          | Key Metric | Doctor-R1 | Best Open-Source (>=32B) |
| :----------------- | :--------- | :-------: | :----------------------: |
| **HealthBench**    | Avg. Score | **36.29** |          33.16           |
| **MAQuE**          | Accuracy   | **60.00** |          57.00           |
| **MedQA**          | Accuracy   | **83.50** |          81.50           |
| **MMLU (Medical)** | Accuracy   | **85.00** |          84.00           |

For a detailed breakdown of results, including ablations and human evaluation, please see our paper.

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@misc{lai2025doctorr1masteringclinicalinquiry,
      title={Doctor-R1: Mastering Clinical Inquiry with Experiential Agentic Reinforcement Learning}, 
      author={Yunghwei Lai and Kaiming Liu and Ziyue Wang and Weizhi Ma and Yang Liu},
      year={2025},
      eprint={2510.04284},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.04284}, 
}
