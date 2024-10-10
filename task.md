## Task

Reading/ Coding

Building a Large Language Model (From Scratch). Book by Sebastian Raschka
Axiomatic Attribution for Deep Networks. This paper was the precursor to XAI methods like Grad-CAM and Grad-CAM++ and variants.
Explaining and Harnessing Adversarial Examples(2014). This was the precursor to many adversarial attacks papers and robustness methods in DL and grown into a massive research area with implications in AI security.
code Compute gradients of a Deep Network w.r.t to inputs and apply them in XAI and Adversatial Attacks. Video tutorial here
Task:

Create Midterm-Bonus branch of your private AI-839 repo.
Verify that you can pre-train 150M GPT-2 small model based on code of Building LLMs from Scratch. Highly recommend reading this book by Sebastian Raschka.
Verify that you can fine-tune with GPT-2 small model on a binary classification task. Consider the spam/ham example dataset provided in there.
Verify that you can compute gradients w.r.t to inputs and the weights of the model.
Due by
11.59PM IST, Tuesday, 15th Oct, 2024.

## First message

I can give one bonus problem, which is open to all, and is optional, for 25pts (1/4th midterm) grade. It will be an implementation exercise of a small LLM (125M parameter) model. Reference code is available. You need a good understanding of PyTorch, a bit of compute, and some time, and a desire to improve the grade. If anyone one of you wants to solve this bonus problem, I will give details. Deadline will be October 15th (hard).

Why that is exciting - will explain in a bit.

## Second message

While we have focussed on tab data to get thought the concepts (both ML Engineering and ML Science) for Full Stack ML in Production,  given the buzz/hype/excitement, we need to cover Gen AI (LLMs). 

But, there are two challenges I see
1. Taming big models is out of question in class room setting (due to memory, compute and time, skill gap issues)
2. ⁠I realized that most things we are talking about (in Tab data setting) are also applicable to the LLM setting - like UQ, XAI, Robustness. One common ingredients to all of these are “gradients” w.r.t to parameters and the inputs (dy/dw, dy/dw, dL/dx, dl/dx) where L is loss, y is output, x is input, w is weight.


So, with a small LLM (in the order of 125M or less), we will be able address both. 

The bonus problem is actually to get ready for the rest of the semester.

## Third message

Decoder-only architecture. Gpt-2-small

Model: 125M (we can go for even smaller model)
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/gpt_train.py
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb

Need three kinds of datasets, one for pre-training, another for instruction fine-tuning, and third for regular (LoRA) fine-tuning on a atask.

That is not difficult. 

For the bonus problem:

Make sure you are able to train classifier (gpt2-small) on the example spam classification using codes from this repo. 
Demonstrate that you are able to calculate/access all the gradients and logits.


This will be pathbreaking for the course.

