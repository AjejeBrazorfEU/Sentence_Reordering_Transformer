# Sentence Reordering Transformer

This repo contains my project submission for the exam of Deep Learning in the AI Master at UNIBO.

## Project Description

In this project I implemented a Transformer model for the task of Sentence Reordering. I followed the [Tensorflow implementation](https://www.tensorflow.org/text/tutorials/transformer) of the Transformer model, that follows the original paper ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762).

<picture>
  <img alt="Transformer architecture" src="https://www.researchgate.net/publication/344310523/figure/fig1/AS:11431281098991710@1669171990308/Transformer-architecture-figure-sourced-from-original-paper-26.png">
</picture>

The Transformer model is a deep learning model that is based on self-attention mechanism. It is particularly useful for sequence-to-sequence tasks, such as translation, summarization, and sentence reordering.

The task of Sentence Reordering consists in taking a sentence and shuffling the words in it. The model is then trained to predict the original order of the words.

## Dataset

The dataset is composed by sentences taken from the generics_kb dataset of hugging face. We restricted the vocabolary to the 10K most frequent words, and only took sentences making use of this vocabulary. The dataset is composed by 240k sentences, that has been splitted in 217k training, 3k validation, and 20k test.

## Final Parameters

The final model has the following parameters:
For the model, the following hyperparameter have been used:

- `num_layers` -> **$4$** (number of Encoder and Decoder layers)
- `d_model` -> **$172$** (dimension of the token embedding inside the model)
- `num_heads` -> **$10$** (number of attention heads in each attention block)
- `dff` -> **$512$** (neurons inside a feed forward layer)
- `max_length` -> **$32$** (max lenght of each input phrase)
- `input_vocab_size` -> **$10\,000$** (size of the input vocabolary)
- `target_vocab_size` -> **$10\,000$** (size of the target vocabolary)
- `output_vocab_size` -> **$28$** (size of the output vocabolary)
- `train_size` -> **$220\,000$** (dimension of the train set)
- `validation_size` -> **$3000$** (dimension of the validation set)
- `epochs` -> **$64$** (epochs used to train the model)
- `learning_rate` -> **$10^{-4}$** (for some learning rate scheduler it varies)
- `lr_scheduler` -> **"constant"** (learning rate scheduler)
- `warmup_steps` -> **Not used** (warmup steps used in some learning rate scheduler)
- `weight_decay` -> **$10^{-3}$** (weight decay used in loss function)
- `optimizer` -> **"adamw"** (optimizer used in training)
- `loss` -> **"categorical_crossentropy"** (loss used in training)

## Evaluation

Let $s$ be the source string and $p$ the prediction of the model. The quality of the results is measured according to the following metric:

1.  Look for the longest substring $w$ between $s$ and $p$.
2.  Compute $\frac{|w|}{max(|s|,|p|)}$

If the match is exact, the score is 1.

## Results

After the training, the model average score on the test is $\large0.6503$.

## How to run the code

You can run the code on Colab by clicking on the following link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AjejeBrazorfEU/Sentence_Reordering_Transformer/blob/main/Domeniconi_Luca_DeepLearning_12_06_2024.ipynb)

## References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Tensorflow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)

## Contacts

For any question, you can contact me at luca.domeniconi5@studio.unibo.it
