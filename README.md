<div align="center">

# Online Handwritten Text Recognition with PyTorch

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/) <br>
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license) <br>
[![Paper](http://img.shields.io/badge/paper-Carbune_et_al._(2020)-B31B1B.svg)](https://doi.org/10.1007/s10032-020-00350-4)

A clean PyTorch implementation of ["Fast multi-language LSTM-based online handwriting recognition"](https://doi.org/10.1007/s10032-020-00350-4) by Carbune *et al.* (2020) 🚀⚡🔥.<br>

_Contributions are always welcome!_

</div>

<br>

## 📌 Introduction

In this repository I provide a [PyTorch](https://pytorch.org/) implementation of the paper "Fast multi-language [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)-based online handwriting recognition" by Victor Carbune *et al.* (see paper here: [10.1007/s10032-020-00350-4](http://doi.org/10.1007/s10032-020-00350-4)), hence the name of this repository.

This work is part of my attempt to build a handwriting recognition system for [Xournal++](https://github.com/xournalpp/xournalpp), a fabulous open-source handwriting notetaking software. Most of this aforementioned attempt is captured in the [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr) repository where I publish working solutions.

## 📺 Project Demo

<div align="center">

<a href="https://youtu.be/H62bjwNkMvc?utm_source=github&utm_medium=readme&utm_campaign=github_readme">
    <img src="docs/static/demo.gif" width="700">
</a>

*([Click on GIF or here to get to video](https://youtu.be/H62bjwNkMvc?utm_source=github&utm_medium=readme&utm_campaign=github_readme).)*

</div>

## 🚀 Quickstart

The following explanation sets you up to use both `src/draw_and_predict_sample.py` and `src/draw_and_store_sample.py` to both predict your own handwritten text and to store it.

1. Install the project according to [the installation section](#installation) in this README and activate the corresponding environment.
2. Download the model `....` (see all available models in section ...) and place it in `...`.
3. Invoke the following command from the root of this repository: `python src/draw_and_predict_sample.py TODO`.

If you want to store your own handwriting sample in a CSV file, then execute `python src/draw_and_store_sample.py`.

## Installation

This repository uses a conda environment in which packages are installed using pip.

Follow these steps to install this package:

1. `conda create --prefix <path> python=3.10.11`
2. `conda activate <path>`
3. `pip3 install torch torchvision torchaudio`
4. `pip install -r requirements.txt`
5. `pip install -e .` (do not forget the dot, `.`)
6. `make test` to confirm that installation was successful

## 🏋️ Training from scratch

- TODO Activate your environment and place training data according to below section
- TODO To train: `python src/train.py` - use `scripts/train ... .sh` instead!

1. Follow installation procedure provided in [Installation](#installation).
2. Download the [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) dataset to `{data_dir}/datasets/IAM-OnDB` where `{data_dir}` is defined in the [paths config](configs/paths/default.yaml). This download process is as follows:
   1. Download the following files that are listed on the above stated dataset website: `data/original-xml-part.tar.gz`, `data/writers.xml`, `data/lineStrokes-all.tar.gz`, `data/lineImages-all.tar.gz`, `data/original-xml-all.tar.gz`, `data/forms.txt` & `ascii-all.tar.gz`.
   2. Extract the content of each of those files into the `{data_dir}/datasets/IAM-OnDB/<file_base_name>` folder where `<file_base_name>` denote the basenames of all downloaded files.

## Training data

TODO: explain how to install data; also provide quick video

[IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) is used as training and validation data. Register on their website to obtain the dataset for free. Afterwards, place the following folders and files from their dataset in this repository's subfolder `data/datasets/IAM-OnDB`. This is how it should look like:

```
├── data/datasets/IAM-OnDB
│   ├── ascii-all/
│   ├── forms.txt
│   ├── lineImages-all
│   ├── lineStrokes-all
│   ├── original-xml-all
│   ├── original-xml-part
│   └── writers.xml
```

## Available models & model cards

- [x] `featuresRaw_decoderGreedy_...`
  - TODO: describe how it's trained
  - TODO: download here (link to blog article w/ bitly)
  - TODO: replicate this in `scripts/`

## ⌛ Open tasks

*All contributions are welcome! :-)*

- [x] Allow inference on own handwriting.
- [ ] Implement CTC beam decoding with language model.
- [ ] Implement Bezier curve fitting algorithm as data preprocessor.
- [ ] Publish trained models on [🤗 Hugging Face](https://huggingface.co/) for easy access.

## 👩‍💻 Contributing

I would love you to contribute! Let's make it a great project that people can benefit from :-).

## 🙏🏼 Acknowledgments

Thanks [Leonard Salewski](https://twitter.com/l_salewski) and [Jonathan Prexl](https://scholar.google.de/citations?user=pqep1wkAAAAJ&hl=de) for super useful discussions on training PyTorch models! Thanks [Harald Scheidl](https://githubharald.github.io/) for providing both great content and code around handwritten text recognition.

I thank the department where I do my PhD, the [School of Physics and Astronomy](https://www.ph.ed.ac.uk/) of [The University of Edinburgh](https://www.ed.ac.uk/), for providing computational resources to train the models.

The scaffold of this code is based on [the awesome lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by [ashleve](https://github.com/ashleve) - thank you for providing this!

## License

This repository is licensed under the [MIT License](LICENSE.md).
