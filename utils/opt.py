import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Facial expression recognition with resnet')
        parser.add_argument('-b', type=int, default=128, help='batch size')
        parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('-e', '--epoch', type=int, default=100, help='No. of epochs')
        parser.add_argument('-d', '--data_dir', type=str, default='./data', help='Data path')

        parser.add_argument('--type', type=str, required=True, help='Type of model (dt/regr/nn): decision tree, logictics regression, resnet(neural networks), respectively')
        
        self.parser = parser
    
    def parse(self):
        args = self.parser.parse_args()
        return args


