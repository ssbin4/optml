import torch
import torchvision
from stanford_cars import StanfordCars
import os
import time
import logging
#import orchestrate.io
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
#from retrying import retry
import shutil
import wandb

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

#fileHandler = logging.FileHandler("{0}/{1}.log".format('./', 'resnet_training_'+str(int(time.time()))))
#fileHandler.setFormatter(logFormatter)
#rootLogger.addHandler(fileHandler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_type_mapping = {'ResNet18': torchvision.models.resnet18, 'ResNet50': torchvision.models.resnet50}


#@retry(wait_fixed=2000, stop_max_attempt_number=5, retry_on_exception=lambda e: isinstance(e, RuntimeError))
def get_pretrained_resnet(is_freeze_weights, number_of_labels, model_type):
    logging.info("loading pretrained resnet model with %d number of labels", number_of_labels)

    try:
        if model_type not in model_type_mapping.keys():
            raise Exception("Please pick either ResNet18 or ResNet50")
        else:
            resnet_pretrained = model_type_mapping[model_type](pretrained=True)
    except RuntimeError:
        # if download fails, delete created download directory
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
        logging.debug("Pytorch pretrained model download failed, deleting model directory for retry.")
        shutil.rmtree(model_dir)
        raise RuntimeError("PyTorch pretrained model download failed.")

    if is_freeze_weights:
        logging.info("tuning fc layer only")
        for parameter in resnet_pretrained.parameters():
            parameter.requires_grad = False
    else:
        logging.info("tuning whole network")
        for parameter in resnet_pretrained.parameters():
            parameter.requires_grad = True
    num_features_fc = resnet_pretrained.fc.in_features
    # add new fully connected layer
    resnet_pretrained.fc = torch.nn.Linear(num_features_fc, number_of_labels)
    return resnet_pretrained


class PalmNet(object):

    def __init__(self, validation_frequency, model,  model_checkpointing, torch_checkpoint_location,
                 epochs, gd_optimizer, loss_function, learning_rate_scheduler, lr_scheduler, init_lr, 
                 gamma, t_0, t_mult, eta_min, annealing_factor,scheduler_rate,freeze_weights, diff_lr, fc_init_lr):
        self.validation_frequency = validation_frequency
        self.model = model
        model.to(device)
        self.model_checkpointing = model_checkpointing
        self.torch_checkpoint_location = torch_checkpoint_location
        self.epochs = epochs
        self.gd_optimizer = gd_optimizer
        self.loss_function = loss_function
        self.learning_rate_scheduler = learning_rate_scheduler
        self.model_directory = None
        self.checkpoint_directory = None
        self.confusion_matrix_directory = None
        self.lr_scheduler = lr_scheduler
        self.init_lr = init_lr
        self.gamma = gamma
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eta_min = eta_min
        self.annealing_factor = annealing_factor
        self.scheduler_rate = scheduler_rate
        self.freeze_weights = freeze_weights
        self.diff_lr = diff_lr
        self.fc_init_lr = fc_init_lr
        if self.model_checkpointing is not None:
            self.generate_directory()

    def generate_directory(self):
        if self.freeze_weights:
            path_name = "freeze_" + self.lr_scheduler
        else:
            path_name = self.lr_scheduler
        if self.diff_lr:
            path_name += '_diffinit_base_%0.4f' % self.init_lr + '_fc_%0.4f' % self.fc_init_lr
        else:
            path_name += '_initlr_%0.4f' % self.init_lr
        if self.lr_scheduler == 'step' or self.lr_scheduler == 'reduce':
            path_name += '_factor%0.3f_step%d' % (self.annealing_factor, self.scheduler_rate)
        elif self.lr_scheduler == 'exp':
            path_name += '_gamma%0.3f' % self.gamma
        elif self.lr_scheduler == 'cos':
            path_name += '_t0_%d_tmult%d_eta%0.3f' % (self.t_0, self.t_mult, self.eta_min)

        self.model_directory = os.path.join(self.torch_checkpoint_location,path_name)
        logging.info("generating training directory in %s", self.model_directory)

        self.checkpoint_directory = os.path.join(self.model_directory, 'model_checkpoints')

        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)
        if not os.path.exists(self.checkpoint_directory):
            os.mkdir(self.checkpoint_directory)
        
        project_name=path_name
        
        wandb.init(project="lr_scheduling", name=path_name, entity="optml")
        """
        wandb.config = {
            "init_lr": self.init_lr,
            "learning rate scheulder": self.lr_scheduler,
            "annealing_factor": self.annealing_factor,
            "step": self.scheduler_rate,
            "gamma": self.gamma,
            "t_max": self.t_max,
            "t_0": self.t_0,
            "t_mult": self.t_mult,
            "eta_min": self.eta_min
        }
        """

    def forward_pass(self, inputs):
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        return outputs, preds

    def backward_pass(self, outputs, labels):
        return self.loss_function(outputs, labels)

    def training_pass(self, inputs, labels, enable_gradients):
        #logging.debug(" and backward pass")

        # zero the parameter gradients
        self.gd_optimizer.zero_grad()

        # forward + backward
        with torch.set_grad_enabled(enable_gradients):
            outputs, preds = self.forward_pass(inputs)
            loss = self.backward_pass(outputs, labels)
            if enable_gradients:
                loss.backward()
                self.gd_optimizer.step()
        return loss, preds

    def train_model(self, training_data, validation_data, number_of_labels):
        """Defines training for tuning of pretrained model.
        Training_data and validation_data are both objects of type DataLoader."""

        logging.info("starting training process")
        logging.info("device being used: %s", device)
        logging.info("training data size: %d", len(training_data.dataset))
        logging.info("validation data size: %d", len(validation_data.dataset))

        logging.info("training data label, unique count: %s", training_data.dataset.get_label_unique_count())
        logging.info("training data label, percentage: %s", training_data.dataset.get_class_distribution())

        logging.info("validation data label, unique count: %s", validation_data.dataset.get_label_unique_count())
        logging.info("validation data label, percentage: %s", validation_data.dataset.get_class_distribution())

        validation_accuracy = 0.0

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            logging.info("epoch number: %d", epoch)
            running_training_loss = 0.0
            running_training_correct_count = 0

            # used for model checkpointing
            # training_loss = None

            all_training_labels = []
            all_training_predictions = []

            self.model.train()

            for i, data in enumerate(training_data):
                inputs = data[StanfordCars.TRANSFORMED_IMAGE]
                labels = data[StanfordCars.LABEL]
                inputs = inputs.to(device)
                labels = labels.to(device)
                training_loss, training_preds = self.training_pass(inputs, labels, True)
                all_training_predictions.extend(training_preds.tolist())
                all_training_labels.extend(labels.tolist())

                correct_count = torch.sum(training_preds == labels.data)
                running_training_loss += training_loss.item()
                running_training_correct_count += correct_count
                """
                logging.debug("fraction of training data processed: %f", (float(i)/len(training_data))*100)
                logging.debug("batch running training loss: %f", running_training_loss)
                logging.debug("batch running training accuracy: %f", running_training_correct_count.item())
                """

            # calculating loss and accuracy over an epoch
            logging.info(
                'Epoch: {} Weigthed F1-Score: {:.4f}, Loss: {:.4f} Acc: {:.4f} '.format("training",
                                                                                       f1_score(y_true=all_training_labels, y_pred=all_training_predictions, average='weighted'),
                                                                                       running_training_loss / len(training_data.dataset),
                                                                                       (running_training_correct_count.double() / len(training_data.dataset)).item()))

            #print("learning rate!!!!:",self.learning_rate_scheduler.get_last_lr())

            if self.lr_scheduler == 'reduce':
                self.learning_rate_scheduler.step(running_training_loss / len(training_data.dataset))
            else:
                self.learning_rate_scheduler.step()

            for param_group in self.gd_optimizer.param_groups:
                logging.debug("current learning rate: %f", param_group['lr'])

            if self.model_checkpointing is not None:
                if epoch % self.model_checkpointing == 0 or epoch == self.epochs -1:
                    self.checkpoint_model(epoch, running_training_loss / len(training_data.dataset), epithet='')

            if epoch % self.validation_frequency == 0 or epoch == self.epochs-1:

                logging.info("validating model")

                self.model.eval()

                running_validation_loss = 0.0
                running_validation_correct_count = 0

                all_validation_labels = []
                all_validation_predictions = []

                # run forward pass on validation dataset
                for i, data in enumerate(validation_data):
                    validation_input = data[StanfordCars.TRANSFORMED_IMAGE]
                    validation_input = validation_input.to(device)
                    validation_labels = data[StanfordCars.LABEL]
                    validation_labels = validation_labels.to(device)

                    validation_loss, validation_predictions = self.training_pass(validation_input, validation_labels, False)

                    all_validation_predictions.extend(validation_predictions.tolist())
                    all_validation_labels.extend(validation_labels.tolist())

                    validation_correct_counts = torch.sum(validation_predictions == validation_labels.data)
                    running_validation_loss += validation_loss.item()
                    running_validation_correct_count += validation_correct_counts
                    """
                    logging.debug("fraction of validation data processed: %f", (float(i)/len(validation_data))*100)
                    logging.debug("batch running validation loss: %f", running_validation_loss)
                    logging.debug("batch running validation accuracy: %f", running_validation_correct_count.item())
                    """

                cm = confusion_matrix(y_true=all_validation_labels, y_pred=all_validation_predictions, labels=list(range(number_of_labels)))
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                #logging.info("confusion matrix:\n %s", cm)

                # Calculating loss over 1 epoch (all data)
                validation_f1_score = f1_score(y_true=all_validation_labels, y_pred=all_validation_predictions, average='weighted')
                validation_accuracy = (running_validation_correct_count.double() / len(validation_data.dataset)).item()
                logging.info('Epoch: {} F1-Score: {:.4f}, Loss: {:.4f} Acc: {:.4f}'.format("validation",
                                                                                          validation_f1_score,
                                                                                          running_validation_loss / len(validation_data.dataset),
                                                                                          validation_accuracy))
                
                if self.diff_lr:
                    wandb.log({"train": {"loss": running_training_loss / len(training_data.dataset), "acc": (running_training_correct_count.double() / len(training_data.dataset))},
                            "val":{"loss": running_validation_loss / len(validation_data.dataset), "acc": validation_accuracy},
                            "learning_rate": self.gd_optimizer.param_groups[0]['lr'], "fc_learning_rate": self.gd_optimizer.param_groups[1]['lr']})
                else:
                    wandb.log({"train": {"loss": running_training_loss / len(training_data.dataset), "acc": (running_training_correct_count.double() / len(training_data.dataset))},
                            "val":{"loss": running_validation_loss / len(validation_data.dataset), "acc": validation_accuracy},
                            "learning_rate": self.gd_optimizer.param_groups[0]['lr']})
            
            else:
                if self.diff_lr:
                    wandb.log({"train": {"loss": running_training_loss / len(training_data.dataset), "acc": (running_training_correct_count.double() / len(training_data.dataset))},
                            "learning_rate": self.gd_optimizer.param_groups[0]['lr'], "fc_learning_rate": self.gd_optimizer.param_groups[1]['lr']})
                else:
                    wandb.log({"train": {"loss": running_training_loss / len(training_data.dataset), "acc": (running_training_correct_count.double() / len(training_data.dataset))},
                            "learning_rate": self.gd_optimizer.param_groups[0]['lr']})
        # orchestrate hook to keep track of metric
        #orchestrate.io.log_metric('accuracy', validation_accuracy)

        logging.info('Finished Training')

        return self.model, validation_accuracy

    def checkpoint_model(self, epoch, training_loss, epithet):
        path_name = self.lr_scheduler + 'init_lr_%0.2f' % self.init_lr
        if self.lr_scheduler == 'step' or self.lr_scheduler == 'reduce':
            path_name += 'factor%0.2fstep%d' % (self.annealing_factor, self.scheduler_rate)
        elif self.lr_scheduler == 'exp':
            path_name += 'gamma%0.2f' % self.gamma
        elif self.lr_scheduler == 'cos':
            path_name += 't0_%d_tmult%d_eta_min%0.3f' % (self.t_0, self.t_mult, self.eta_min)
        model_checkpoint_path = self.model_directory
        logging.info("saving model at %s", model_checkpoint_path)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.gd_optimizer.state_dict(),
                    'learning_rate_scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
                    'loss': training_loss},
                   os.path.join(model_checkpoint_path, 'epoch_%d.pth' % epoch))
