## LT2316 H20 Assignment A2 : Ner Classification

Mohamed Zarzoura

## Notes on Part 1

1. I used PyTorch-nlp package `pip install pytorch-nlp --user` to use GLove pre-trained word vectors.

2. The architecture of the model is as follows:

   1. An embedding layer that embeds each token in the sequence.
   2. A Bi-LSTM layer that process the embeddings (one per time-step). The forward and backward hidden states from the final layer of the LSTM are then concatenated.
   3. A linear fully connected layer takes the concatenated final hidden states from the LSTM to predict tag for each token in the sequence.

## Notes on Part 2

No comments

## Notes on Part 3

The set of hyperparameters that I tried are:

1. learning rate
2. number layers
3. batch size
4. Size of token embeddings

I choose these parameters to examine which one of them may affect the model generalization ability. Each parameter has two different values. I build eight models using all possible combinations of the hyperparameters. The best model that has a higher F1-score is then saved. The training loop stops when the metric score does not change for five successive epochs, early stopping. The implementation of stopping is from [here](https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d).

From the parallel coordination plot, we can notice the following:

1. There is no big difference in F1-score. All models achieve F1-score around 0.98 (Average F1-score = 0.9863 ±0.00114)
2. F1-score can be grouped based on the size of embeddings:
   1. the higher scores are achieved by the 300 vector embeddings.
   2. models with 300 vector embeddings reach to the stopping condition faster, need fewer epochs to generalize.
3. All other parameters do not show any effect on the models' training.

Generally speaking, the performance of all models are the same. However, models with a larger size of embeddings need less training epochs to generalize. In the next step, I choose to fix the above mentioned four parameters and see the effect of learning rate and optimizer on the models training.

From the second parallel coordination plot, we can notice that Adam optimizer has a higher F1-score.

## Notes on Part 4

No comment
