from flautim.pytorch.common import run_federated, get_argparser
from flautim.pytorch import Model, Dataset
from flautim.pytorch.federated import Experiment
import MNISTDataset, MNISTModel, MNISTExperiment

def generate_client_fn(context, measures, logger):
    
    def create_client_fn(id):
    
        model = MNISTModel.MNISTModel(context, suffix = id)

        if str(id) != 'FL-Global':
            file = '0'
        else:
            file = str(id)
        
        dataset = MNISTDataset.MNISTDataset(context.path +"data/{}.npz".format(file), batch_size = 10, shuffle = False, num_workers = 0)
        
        return MNISTExperiment.MNISTExperiment(model, dataset, measures, logger, context)
        
    return create_client_fn
    

def evaluate_fn(context, measures, logger):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = MNISTModel.MNISTModel(context)
        model.set_parameters(parameters)
        
        dataset = MNISTDataset.MNISTDataset(context.path +"data/1.npz", batch_size = 10, shuffle = False, num_workers = 0)
        
        experiment = MNISTExperiment.MNISTExperiment(model, dataset, measures, logger, context)
        
        loss, accuracy = experiment.validation_loop(dataset.dataloader(validation=True)) 

        return loss, {"accuracy": accuracy}

    return fn

if __name__ == '__main__':


    parser, context, backend, logger, measures = get_argparser()
    
    
    client_fn_callback = generate_client_fn(context, measures, logger)
    evaluate_fn_callback = evaluate_fn(context, measures, logger)

    run_federated(client_fn_callback, evaluate_fn_callback)