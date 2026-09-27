import torch
import torch.optim as optim
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, dataloader, test_dataloader, device="cpu") -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
        # The long-term optimizer updates the Linear weights across all batches
        self.weight_opt = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=1):
        lr = 0.1
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, label) in enumerate(self.dataloader):
                loss = self.train_step(data, label, lr)
                total_loss += loss
                lr *= .999
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Energy: {loss:.4f} | lr: {lr}")
            self.evaluate()

    def train_step(self, x_data, y_label, lr):
        batch_size = x_data.size(0)
        x_data = x_data.to(self.device)
        
        # One-hot encode the labels (e.g. integer 3 becomes [0,0,0,1,0...])
        y_label = F.one_hot(y_label, num_classes=10).float().to(self.device)

        # 1. Setup the batch states
        self.model.init_hidden(x_data, batch_size, self.device)
        
        # Optimizer for short-term inference (only tracks the hidden state)
        state_opt = optim.SGD([self.model.hidden_layer.state], lr=lr)

        # 2. Inference Phase (Settling)
        for _ in range(3): 
            state_opt.zero_grad()
            energy = self.model.get_energy(y_label)
            energy.backward()
            state_opt.step()

        # 3. Learning Phase (Weight Update)
        self.weight_opt.zero_grad()
        
        # Detach the settled state so gradients don't flow infinitely backwards through time
        self.model.hidden_layer.state = self.model.hidden_layer.state.detach()
        
        final_energy = self.model.get_energy(y_label)
        final_energy.backward()
        self.weight_opt.step()

        return final_energy.item()

    def evaluate(self):
            # Good practice for standard layers (like Dropout), even if we don't have them yet
            test_dataloader = self.test_dataloader
            self.model.eval() 
            
            correct = 0
            total = 0
            
            # Notice the absence of 'with torch.no_grad():'
            for data, true_labels in test_dataloader:
                true_labels = true_labels.to(self.device)
                
                # 1. Run the iterative settling phase
                predictions, y_state = self.infer_step(data)
                
                # 2. Tally up the correct guesses
                correct += (predictions == true_labels).sum().item()
                total += true_labels.size(0)
                
            accuracy = 100 * correct / total
            print(f"--> Epoch Validation Accuracy: {accuracy:.2f}%")
            
            # Set back to training mode
            self.model.train()
            return accuracy

    def infer_step(self, x_data):
            batch_size = x_data.size(0)
            x_data = x_data.to(self.device)

            y_state = torch.zeros(batch_size, 10, device=self.device, requires_grad=True)

            self.model.init_hidden(x_data, batch_size, self.device)

            infer_opt = optim.SGD([self.model.hidden_layer.state, y_state], lr=0.001)

            for _ in range(50): # Often requires more iterations than training
                infer_opt.zero_grad()
                
                energy = self.model.get_energy(y_state)
                
                energy.backward()
                infer_opt.step()

            predictions = torch.argmax(y_state, dim=1)
            
            return predictions, y_state
class StatePredictorTrainer:
    def __init__(self, model, state_predictor, dataloader, test_dataloader, device="cpu") -> None:
        self.model = model.to(device)
        self.state_predictor = state_predictor.to(device)
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.state_pred_opt = optim.Adam(self.state_predictor.parameters(), lr=0.001) 
        # The long-term optimizer updates the Linear weights across all batches
        self.weight_opt = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=1):
        lr = 0.1
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, label) in enumerate(self.dataloader):
                loss = self.train_step(data, label, lr)
                total_loss += loss
                lr *= 0.999
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Energy: {loss:.4f} | lr: {lr}")
            self.evaluate()

    def train_step(self, x_data, y_label, lr):
        batch_size = x_data.size(0)
        x_data = x_data.to(self.device)
        
        # One-hot encode the labels (e.g. integer 3 becomes [0,0,0,1,0...])
        y_label = F.one_hot(y_label, num_classes=10).float().to(self.device)

        # 1. Setup the batch states
        self.model.init_hidden(x_data, batch_size, self.device)
        self.state_pred_opt.zero_grad()
        pred_h, pred_d = self.model(y_label)
        pred_state = self.state_predictor(pred_h, x_data)
        self.model.hidden_layer.state = pred_state.detach().clone().requires_grad_(True)

        state_opt = optim.SGD([self.model.hidden_layer.state], lr=lr)

        # 2. Inference Phase (Settling)
        for _ in range(3): 
            state_opt.zero_grad()
            energy = self.model.get_energy(y_label)
            energy.backward()
            state_opt.step()


        # 3. Learning Phase (Weight Update)
        self.weight_opt.zero_grad()
        
        # Detach the settled state so gradients don't flow infinitely backwards through time
        self.model.hidden_layer.state = self.model.hidden_layer.state.detach()
        
        final_energy = self.model.get_energy(y_label)
        final_energy.backward()
        self.weight_opt.step()

        self.state_pred_opt.zero_grad()
        predictor_loss = F.mse_loss(pred_state, self.model.hidden_layer.state)
        predictor_loss.backward()
        self.state_pred_opt.step()
        return final_energy.item()

    def evaluate(self):
            # Good practice for standard layers (like Dropout), even if we don't have them yet
            test_dataloader = self.test_dataloader
            self.model.eval() 
            
            correct = 0
            total = 0
            
            # Notice the absence of 'with torch.no_grad():'
            for data, true_labels in test_dataloader:
                true_labels = true_labels.to(self.device)
                
                # 1. Run the iterative settling phase
                predictions, y_state = self.infer_step(data)
                
                # 2. Tally up the correct guesses
                correct += (predictions == true_labels).sum().item()
                total += true_labels.size(0)
                
            accuracy = 100 * correct / total
            print(f"--> Epoch Validation Accuracy: {accuracy:.2f}%")
            
            # Set back to training mode
            self.model.train()
            return accuracy

    def infer_step(self, x_data):
            batch_size = x_data.size(0)
            x_data = x_data.to(self.device)

            y_state = torch.zeros(batch_size, 10, device=self.device, requires_grad=True)

            self.model.init_hidden(x_data, batch_size, self.device)

            with torch.no_grad():
                pred_h, pred_d = self.model(y_state)
                pred_state = self.state_predictor(pred_h, x_data)

            self.model.hidden_layer.state = pred_state.clone().requires_grad_(True)
            infer_opt = optim.SGD([self.model.hidden_layer.state, y_state], lr=0.01)
            for _ in range(10): # Often requires more iterations than training
                infer_opt.zero_grad()
                
                energy = self.model.get_energy(y_state)
                
                energy.backward()
                infer_opt.step()

            predictions = torch.argmax(y_state, dim=1)
            
            return predictions, y_state
 
