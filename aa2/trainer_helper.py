import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def calculate_time(start_time, end_time):
    time = end_time - start_time
    mins = int(time // 60)
    secs = int(time - (mins * 60))
    return mins, secs

def buid_model(paramaters, model_class):
    embed_layer_in_size = paramaters[0]
    embed_size = paramaters[1]
    lstm_hidden_size = paramaters[2]
    num_layers = paramaters[3]
    num_dirs = paramaters[4]
    dropout_ = paramaters[5]
    out_size = paramaters[6]
    

    model = model_class(input_size=embed_layer_in_size,
                        embedding_size=embed_size,
                        hidden_size=lstm_hidden_size,
                        output_size=out_size,
                        num_layers=num_layers,
                        num_dirs=num_dirs,
                        dropout_=dropout_
                        )

    return model


def F_measure_batch(y_true, y_pred, pad_token_id):
    # predictedner labels id
    labels_id_pred = torch.max(y_pred, dim=1)[1]

    # filter pad id
    labels_nopad_idx = (y_true != pad_token_id).nonzero()
    labels_id_pred = labels_id_pred[labels_nopad_idx]
    y_true = y_true[labels_nopad_idx]

    # calc F1, sklearn convert tensor to numpy and need them in cpu
    y_true_cpu = y_true.squeeze(1).to("cpu")
    labels_id_pred_cpu = labels_id_pred.squeeze(1).to("cpu")
    F_score = f1_score(y_true_cpu, labels_id_pred_cpu, average="micro")
    F_score = torch.tensor(F_score, dtype=torch.float32, device=y_pred.device)

    return F_score


def train_epoch(model, iterator, optimizer, criterion, pad_token_id):
    # Define Loss, score
    running_loss = 0
    running_f1 = 0

    model.train()

    for _i, batch in enumerate(iterator):

        x = batch[0]
        y = batch[1]
        # resets the gradients after every batch
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = model(x)

        # Reshape output dim:
        #   FROM:   (num_batch        , seq_len    , output_size )
        #   TO:     (Seq_len*num_batch, output_size              )
        # Reshape y dim:
        #   FROM:   (num_batch        , seq_len)
        #   TO:     (Seq_len*num_batch         )
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y = y.view(-1)

        loss = criterion(y_pred, y)
        score_f1 = F_measure_batch(y, y_pred, pad_token_id)

        loss.backward()
        optimizer.step()

        # loss
        running_loss += loss.item()
        running_f1 += score_f1.item()

        # if _i%300 == 299 or (_i%300 != 299 and _i == len(iterator)-1):
        #         print("\tBatch-{: <5d}: loss: {:.3f},  F1_score: {:.3f}".format(
        #             (_i+1)*8, running_loss/(_i+1), running_f1/(_i+1)))

    return running_loss/len(iterator), running_f1/len(iterator)


def evaluate_epoch(model, iterator, criterion, pad_token_id):
    # Define Loss, score
    running_loss = 0
    running_f1 = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:

            x = batch[0]
            y = batch[1]
            y_pred = model(x)

            # Reshape output dim:
            #   FROM:   (num_batch        , seq_len    , output_size )
            #   TO:     (Seq_len*num_batch, output_size              )
            # Reshape y dim:
            #   FROM:   (num_batch        , seq_len)
            #   TO:     (Seq_len*num_batch         )
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)

            loss = criterion(y_pred, y)
            score_f1 = F_measure_batch(y, y_pred, pad_token_id)

            running_loss += loss.item()
            running_f1 += score_f1.item()

    return loss.to("cpu"), running_loss/len(iterator), running_f1/len(iterator)
