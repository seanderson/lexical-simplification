import torch as tt
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm.auto import tqdm

from src.graphalign.dataset import AlignDatasetTrain
from src.graphalign.models import GGNN, AlignLSTM
from src.graphalign.losses import AlignmentLoss


def run_epoch(model, optimizer, criterion, dataset, backprop=True, keep_results=False, cuda=True):
    """
    Run a given model for one epoch on the given dataset and return the loss and the accuracy
    :param model:           the model to train\evaluate
    :param optimizer:       optimizer to use
    :param criterion:       loss function to use
    :param dataset:         a DataLoader object
    :param backprop:        whether to backpropagate or not
    :param keep_results:    whether to store the results for later alignment testing
    :param cuda:            whether to use cuda
    :return:                mean loss, mean accuracy, results (if these are kept)
    """
    model.double()
    if cuda:
        model.cuda()
    if backprop:
        model.train()
    else:
        model.eval()

    total_loss, total_acc, step = 0, 0, 0
    batches = tqdm(dataset)
    results = []

    for matrix, features, mask in batches:

        if cuda:
            matrix = matrix.cuda()
            features = features.cuda()
            mask = mask.cuda()

        step += 1
        optimizer.zero_grad()
        model.zero_grad()
        distances = model(matrix, features)

        loss, acc = criterion(distances, mask)
        total_acc += acc.cpu().data.numpy().item()
        total_loss += loss.cpu().data.numpy().item()
        batches.set_description("Acc: step=%.2f, m.=%.3f. M. loss=%.3f" % (acc, total_acc / step, total_loss / step))

        if backprop:
            loss.backward(retain_graph=True)
            tt.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if keep_results:
            results.append(distances.detach.cpu())

    return total_loss/step, total_acc/step, results


def train(model, optimizer, criterion, max_epochs, patience, train_data, val_data, best_model_file,
          cuda=True):
    """
    Train a given model for a given number of epochs. Use early stopping with given patience.
    :param model:           a model to train
    :param optimizer:       optimizer to use
    :param criterion:       loss function to use
    :param max_epochs:      maximum number of epochs to run the model for
    :param patience:        patience value to use for early stopping
    :param train_data:      training dataset
    :param val_data:        validation dataset
    :param best_model_file: file to save the best model to
    :param cuda:            whether to use cuda (or cpu)
    """
    best_val_loss, best_val_loss_epoch = float("inf"), 0
    for epoch in range(max_epochs):
        run_epoch(model, optimizer, criterion, train_data, True, False, cuda)
        val_loss, _, _ = run_epoch(model, optimizer, criterion, val_data, False, False, cuda)

        if val_loss < best_val_loss:
            tt.save(model.state_dict(), best_model_file)
            print("Best epoch so far. Saving to '%s')" % best_model_file)
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
        elif epoch - best_val_loss_epoch >= patience:  # early stopping
            print("Early Stopping after %i epochs." % epoch)
            break


def test_model(model, optimizer, criterion, test_data, cuda=True):
    """
    Test a given model on a given dataset. This does not test the alignments per se, only the
    prediction of whether a entence belongs to an article or not
    :param model:           a model to train
    :param optimizer:       optimizer to use
    :param criterion:       loss function to use
    :param test_data:       testing dataset
    :param cuda:            whether to use cuda (or cpu)
    :return:                mean loss, mean accuracy
    """
    loss, acc, _ = run_epoch(model, optimizer, criterion, test_data, False, False, cuda)
    print("testing loss: %f, acc: %f." % (loss, acc))
    return loss, acc


if __name__ == "__main__":

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tt.manual_seed(seed)

    # """
    train_data = AlignDatasetTrain(filename="/home/nlp/wpred/newsela/articles/parsedTrain.json",
                                    levels=(0, 2), max_sents=100, max_words=50, embedding_size=8,
                                    edge_types=2)
    train_data = DataLoader(train_data, shuffle=True, batch_size=20, pin_memory=True)
    # """

    val_data = AlignDatasetTrain(filename="/home/nlp/wpred/newsela/articles/parsedValid.json",
                                    levels=(0, 2), max_sents=100, max_words=50, embedding_size=8,
                                    edge_types=2)
    val_data = DataLoader(val_data, shuffle=False, batch_size=20, pin_memory=True)

    test_data = AlignDatasetTrain(filename="/home/nlp/wpred/newsela/articles/parsedTest.json",
                                    levels=(0, 2), max_sents=100, max_words=50, embedding_size=8,
                                    edge_types=2)
    test_data = DataLoader(test_data, shuffle=False, batch_size=20, pin_memory=True)

    """
    model = GGNN(state_dim=8, n_edge_types=2, n_nodes=50, n_steps=3, annotation_dim=8)
    """
    model = AlignLSTM(state_dim=8,
                      n_nodes=50,
                      n_hidden=20,
                      n_output=20,
                      n_layers=2)
    # """

    criterion = AlignmentLoss()
    optimizer = tt.optim.Adam(model.parameters(), lr=1e-3)

    best_filename = "best"+str(8)+"-" + str(1e-3) + "_" + str(3) + ".model"

    train(model, optimizer=optimizer, criterion=criterion, max_epochs=20, patience=3,
          train_data=train_data, val_data=val_data, best_model_file=best_filename)

    model.load_state_dict(tt.load(best_filename))
    loss, acc, _ = run_epoch(model, optimizer, criterion, test_data, False, False)
    print(loss, acc)
