# Perspectrum 
[![Build Status](https://semaphoreci.com/api/v1/projects/13a8c4da-13ae-4934-a9b9-37611f91528f/2266937/badge.svg)](https://semaphoreci.com/danyaljj/perspective)

This repository contains codes and scripts relevant to the dataset introduced in [this paper](http://cogcomp.org/page/publication_view/870).

## Code structure: 

 - `website/`: contains the webapp used for crowdsourcing experiments, as well as the search engine for claim/perspectives. 
 - `data/`: contains the dataset at each stages of our annotations. 
 - `experiments`: Jupyter notebooks containing different small experiments for dataset construction. 
 
## Model

You can download the BERT-based baseline models for all four tasks from the [google drive](https://drive.google.com/drive/folders/1L3WlAtf9DrEhEgE46QIeceRimqLK8cMk?usp=sharing).

## Demo

We host a demo of our BERT-based baseline systems [here](http://orwell.seas.upenn.edu:4002/)
The server code for the demo is available at this [repo](https://github.com/CogComp/perspectroscope)

## Citation 
Work is published under **the Creative Commons license**. 
Please cite the following work if you want to refer to this work: 
```
@inproceedings{chen2018perspectives,
  title={Seeing Things from a Different Angle: Discovering Diverse Perspectives about Claims},
  author={Chen, Sihao and Khashabi, Daniel and Yin, Wenpeng and Callison-Burch, Chris and Roth, Dan},
  book={Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year={2019}
}
```
