import os

class Monitor(object):
    
    def __init__(self, layers, load_path=None, save_path=None):
        self.layers = layers
        self.load_path = load_path
        self.save_path = save_path
        self.min_cost = None
    
    def load(self):
        assert not self.load_path == None, self.load_path
        assert os.path.exists(self.load_path), self.load_path
        for layer in self.layers:
            layer.load_data(self.load_path)

    def save(self, cost):
        assert not self.save_path == None, self.save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # latest
        temp_name = os.path.join(self.save_path, 'latest')
        for layer in self.layers:
            layer.save_data(temp_name)

        # min_cost
        if self.min_cost == None or self.min_cost > cost:
            self.min_cost = cost
            temp_name = os.path.join(self.save_path, 'min_cost')
            for layer in self.layers:
                layer.save_data(temp_name)
