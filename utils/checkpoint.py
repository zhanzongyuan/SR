import torch
import os
class Checkpoint:
    def __init__(self, checkpoint_path='', filename=''):
        self.contextual = {}
        self.contextual['b_epoch'] = 0
        self.contextual['b_batch'] = 0
        self.contextual['prec'] = 0
        self.contextual['loss'] = 0
        self.checkpoint_path = checkpoint_path
        self.filename=filename
        self.best_prec1 = 0
        self.best=False
    
    def record_contextual(self, contextual):
        self.contextual = contextual
        if self.contextual['prec'] > self.best_prec1:
            self.best = True
            self.best_prec1 = self.contextual['prec']
        else:
            self.best = False


    def save_checkpoint(self, model):
        path = os.path.join(self.checkpoint_path, self.filename)

        torch.save(self.contextual, path+'_contextual.pth')
		print('...Contextual saved')

        torch.save(model.state_dict(), path+'.pth')
		print('...Model saved')

        if (self.best):
            torch.save(self.contextual, path+'_contextual_best.pth')
            torch.save(model.state_dict(), path+'_best.pth')
		    print('...Best model and contextual saved')

    def load_checkpoint(self, model):
        path = os.path.join(self.checkpoint_path, self.filename)

        if path and os.path.isfile(path+'_contextual.pth'):
            self.contextual = torch.load(path+'_contextual.pth')

            # Update best prec.
            if self.contextual['prec'] > self.best_prec1:
                self.best = True
                self.best_prec1 = self.contextual['prec']
            else:
                self.best = False
        else:
            print("====> no checkpoint contextual at '{}'".format(path+'_contextual.pth'))

        if path and os.path.isfile(path+'.pth'):
            model.load_state_dict(torch.load(path+'.pth'))
        else:
            print("====> no pretrain model at '{}'".format(path+'.pth'))