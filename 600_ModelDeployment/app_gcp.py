import json
import requests
import torch
import torch.nn as nn

def predict(request):
    class MultiClassNet(nn.Module):
        def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
            super().__init__()
            self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
            self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
            self.log_softmax = nn.LogSoftmax(dim=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.lin1(x)
            x = self.relu(x)
            x = self.lin2(x)
            x = self.log_softmax(x)
            return x
    # model instance
    model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
    
    # model weights
    URL = 'https://storage.googleapis.com/deploy_iris_model/model_iris_state.pt'
    r = requests.get(URL)
    local_temp_file = "/tmp/model.pt" 
    file = open(local_temp_file, "wb")
    file.write(r.content)
    file.close()
    model.load_state_dict(torch.load(local_temp_file))

    dict_data = request.get_json()
    X = torch.tensor([dict_data['data']])
            
    y_test_hat_softmax = model(X)
    y_test_hat = torch.max(y_test_hat_softmax.data, 1)
    y_test_cls = y_test_hat.indices.detach().numpy()[0]
    cls_dict = {
                0: 'setosa', 
                1: 'versicolor', 
                2: 'virginica'
            }
    
    result = f"Your flower belongs to class {cls_dict[y_test_cls]}."
    return result
