"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/Tongji',
                    help="Directory containing the dataset")
parser.add_argument('--dataset', choices=['IIT Delhi V1','IIT Delhi V1 Segmented','Tongji','Tongji Segmented','REST'], help="Choose between datasets")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, test_dataloader,train_dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        test_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches eval data
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches enrolled users data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    
    batch_enrolled_embeds = []
    batch_enrolled_labels = []
    # compute enrolled users embedding
    for _ in range(params.num_enroll_iters):
        data_batch, labels_batch = next(iter(train_dataloader))
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        batch_enrolled_embeds.append(model(data_batch).data.cpu())
        batch_enrolled_labels.append(labels_batch.data.cpu())
    enrolled_embeds = torch.cat([batch_enrolled_embeds[i] for i in range(params.num_enroll_iters)], 0)
    enrolled_labels = torch.cat([batch_enrolled_labels[i] for i in range(params.num_enroll_iters)], 0)
    # compute metrics over the dataset
    for data_batch, labels_batch in test_dataloader:
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu
        output_batch = output_batch.data.cpu()
        labels_batch = labels_batch.data.cpu()

        # compute all metrics on this batch
        # summary_batch = {metric: metrics[metric](output_batch, labels_batch)
        #                  for metric in metrics}
        EER, threshold = metrics["EER"](enrolled_embeds, enrolled_labels, output_batch, labels_batch, "cpu")
        summary_batch = {"EER": EER
                         , "Average Non Zero": metrics["Average Non Zero"](output_batch, labels_batch)}
        summary_batch['threshold'] = threshold
        summary_batch['loss'] = loss[0].item()
        summary_batch['length triplets'] = loss[1]
        summ.append(summary_batch)


    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def main():
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_online_dataloader(['train','test'], args.data_dir, args.dataset, params)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, train_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)


if __name__ == '__main__':
    main()
