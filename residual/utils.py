import time
from pathlib import Path
import torch
from .residual_net import ResNet


class Trainer:
    def __init__(
            self,
            model_config,
            n_classes,
            optimizer,
            criterion,
            trainset,
            testset,
            learning_rate=0.01,
            average_pool_kernel_size=1,
            batch_size=4,
            patience=0,
            min_delta=0,
            checkpoint_path='.',
            verbosity=0):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.trainloader  = self.build_dataloader(trainset)
        self.testloader  = self.build_dataloader(testset)
        self.model = ResNet(
                in_size = self.get_input_dims(self.trainloader),
                    out_size = n_classes,
                    configs = model_config,
                    average_pool_kernel_size = average_pool_kernel_size,
                    device = self.device
                )
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.curr_train_loss = None
        self.wait = 0
        self.patience = patience
        self.best = float('inf')
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbosity = verbosity
        self.history = {}
        self.history['train'] = {}
        self.history['train']['loss'] = []
        self.history['train']['acc'] = []
        self.history['val'] = {}
        self.history['val']['loss'] = []
        self.history['val']['acc'] = []


    def reset_history(self):
        self.history = {}
        self.history['train'] = {}
        self.history['train']['loss'] = []
        self.history['train']['acc'] = []
        self.history['val'] = {}
        self.history['val']['loss'] = []
        self.history['val']['acc'] = []

    def build_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2)

    def get_input_dims(self, dataset):
        images, labels = next(iter(dataset))
        return (1, *images[0].shape)


    def accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def train_step(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train() 
        
        for (x, y) in self.trainloader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)
            acc = self.accuracy(y_pred, y)

            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.trainloader), epoch_acc / len(self.trainloader)


    def eval_step(self):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for (x, y) in self.testloader:

                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)
                acc = self.accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.testloader), epoch_acc / len(self.testloader)


    def early_stop(self, epoch):
        self.wait += 1
        if self.curr_train_loss - self.min_delta < self.best:
            self.best = self.curr_train_loss
            torch.save(self.model, Path(checkpoint_path, f'best_model_epoch_{epoch}.pt'))
            self.wait = 0

        if self.wait >= self.patience and epoch > 0:
            return True
        return False


    def train(self, epochs):
        for epoch in range(epochs):
            start = time.time()
            train_loss, train_acc = self.train_step()
            self.history['train']['loss'].append(train_loss)
            self.history['train']['acc'].append(train_acc)
            self.curr_train_loss = train_loss

            val_loss, val_acc = self.eval_step()
            self.history['val']['loss'].append(val_loss)
            self.history['val']['acc'].append(val_acc)
            elapsed = time.time() - start
            elapsed_formatted = time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed))
            if self.verbosity > 0:
                print(f'train loss: {train_loss}, ' + \
                        f'train accuracy: {train_acc}, ' + \
                        f'validation loss: {val_loss}, ' + \
                        f'validation accuracy: {val_acc}, ' + \
                        f'elapsed: {elapsed_time}')

            # if patience is 0, assume user doesn't want to early stop
            if self.patience > 0 and self.early_stop(epoch):
                break
            else:
                # if not using early stopping, just save the best model
                self.best = self.curr_train_loss
                torch.save(self.model, Path(checkpoint_path, f'best_model_epoch_{epoch}.pt'))


        torch.save(self.model, Path(checkpoint_path, f'final_model.pt'))
        return self.history


