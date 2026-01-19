# Automatic Speech Recognition (ASR) Hub

![ASR](https://img.shields.io/badge/Task-Automatic%20Speech%20Recognition-blue)
![Models](https://img.shields.io/badge/Focus-Models%20%7C%20Datasets%20%7C%20Metrics-green)
![Notebooks](https://img.shields.io/badge/Includes-Fine--tuning%20Notebooks-orange)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-lightgrey)

A curated repository for **Automatic Speech Recognition (ASR)** that aggregates **models, datasets, and evaluation metrics**, along with **practical Jupyter notebooks** demonstrating fine-tuning workflows and a **minimal end-to-end ASR project structure**.

This repository is intended as a **reference and starting point** for learning, experimenting, and building ASR systems, rather than a unified framework or benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Datasets](#datasets)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Notebooks & Project Structure](#notebooks--project-structure)
- [References](#references)

---

## Overview

Automatic Speech Recognition (ASR) has evolved rapidly with the introduction of end-to-end architectures and large-scale self-supervised pretraining. As a result, the ASR ecosystem now consists of a wide range of models, datasets, training recipes, and evaluation practices scattered across papers and repositories.

This repository aims to:
- **Summarize and organize** commonly used ASR models and datasets
- **Provide a clear entry point** to important repositories and papers
- **Demonstrate practical fine-tuning workflows** using runnable notebooks
- **Show a minimal, clean project structure** for end-to-end ASR training and experimentation

The focus is on **clarity, practicality, and discoverability**.

---

## Models

This section lists representative ASR models grouped by modeling approach. Each entry typically includes:
- the original paper
- the official or commonly used implementation repository

### End-to-End ASR Models

- **DeepSpeech / DeepSpeech2**  
  Paper: Deep Speech: Scaling up end-to-end speech recognition  
  Repository: https://github.com/mozilla/DeepSpeech

- **Listen, Attend and Spell (LAS)**  
  Paper: Listen, Attend and Spell  
  Repository: https://github.com/tensorflow/models

### CTC-based Models

- **wav2vec 2.0**  
  Paper: wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations  
  Repository: https://github.com/facebookresearch/fairseq

- **HuBERT**  
  Paper: HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units  
  Repository: https://github.com/facebookresearch/fairseq

### Transducer-based Models

- **RNN-T**  
  Paper: Sequence Transduction with Recurrent Neural Networks  
  Repository: https://github.com/kensho-technologies/pyctcdecode

- **Emformer**  
  Paper: Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition  
  Repository: https://github.com/pytorch/audio

> This list is not exhaustive and will be expanded over time.

---

## Datasets

This section summarizes widely used datasets for ASR research and development.

Each dataset entry may include:
- language(s)
- approximate size
- primary use case

### Common ASR Datasets

- **LibriSpeech**  
  Language: English  
  Size: ~1000 hours  
  Link: https://www.openslr.org/12

- **Common Voice**  
  Language: Multilingual  
  Size: Varies by language  
  Link: https://commonvoice.mozilla.org

- **TED-LIUM**  
  Language: English  
  Domain: Public talks  
  Link: https://www.openslr.org/19

- **AISHELL**  
  Language: Mandarin Chinese  
  Link: https://www.openslr.org/33

---

## Evaluation & Benchmarks

This section provides an overview of evaluation practices commonly used in ASR.

### Evaluation Metrics

- **WER (Word Error Rate)**
- **CER (Character Error Rate)**
- **SER (Sentence Error Rate)**

### Benchmarks

- LibriSpeech test-clean / test-other
- Common Voice test sets
- TED-LIUM test set

Evaluation scripts and decoding strategies may differ across repositories and frameworks.

---

## Notebooks & Project Structure

In addition to curated resources, this repository includes **practical code examples**:

### Jupyter Notebooks
- Fine-tuning pretrained ASR models
- Dataset preprocessing examples
- Inference and decoding demonstrations

These notebooks are designed to be:
- runnable
- minimal
- educational

### Minimal ASR Project Structure

The repository also provides a **lightweight project structure**, inspired by *cookiecutter data science*, for end-to-end ASR training and experimentation. It includes:
- data handling
- training scripts
- evaluation scripts
- configuration files

This structure serves as a **reference template**, not a framework.

---

## References

- Graves et al., *Connectionist Temporal Classification*
- Graves et al., *Sequence Transduction with Recurrent Neural Networks*
- Baevski et al., *wav2vec 2.0*
- Hsu et al., *HuBERT*
- Chan et al., *Listen, Attend and Spell*

Additional references and repositories will be added incrementally.
