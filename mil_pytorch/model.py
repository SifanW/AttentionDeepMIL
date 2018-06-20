import torch
import torch.nn as nn
import torch.nn.functional as F




class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
#         self.L = 500
#         self.D = 128
#         self.K = 1
        self.L = 256
        self.D = 64
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
            nn.Conv2d(1, 20, kernel_size=5,stride=1), #128 -> 124
            nn.Conv2d(20, 40, kernel_size=5,stride=1),#124 -> 120
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),#120 -> 60
            nn.Conv2d(40, 50, kernel_size=3),#60->58
            nn.Conv2d(50, 50, kernel_size=3),#58->56
            nn.MaxPool2d(2, stride=2),#56->28
            nn.ReLU(),
            nn.Conv2d(50, 80, kernel_size=3),#28->26
            nn.Conv2d(80, 80, kernel_size=3),#26->24
            nn.MaxPool2d(2, stride=2),#24->12
            nn.Conv2d(80, 128, kernel_size=3),#12->10
            nn.Conv2d(128, 128, kernel_size=3),#10->8
            nn.MaxPool2d(2, stride=2)#8->4
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(128 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 128 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        #error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
def make_dot(var, params=None):  
    """ Produces Graphviz representation of PyTorch autograd graph 
    Blue nodes are the Variables that require grad, orange are Tensors 
    saved for backward in torch.autograd.Function 
    Args: 
        var: output Variable 
        params: dict of (name, Variable) to add names to node that 
            require grad (TODO: make optional) 
    """  
    if params is not None:  
        assert isinstance(params.values()[0], Variable)  
        param_map = {id(v): k for k, v in params.items()}  
  
    node_attr = dict(style='filled',  
                     shape='box',  
                     align='left',  
                     fontsize='12',  
                     ranksep='0.1',  
                     height='0.2')  
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))  
    seen = set()  
  
    def size_to_str(size):  
        return '('+(', ').join(['%d' % v for v in size])+')'  
  
    def add_nodes(var):  
        if var not in seen:  
            if torch.is_tensor(var):  
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')  
            elif hasattr(var, 'variable'):  
                u = var.variable  
                name = param_map[id(u)] if params is not None else ''  
                node_name = '%s\n %s' % (name, size_to_str(u.size()))  
                dot.node(str(id(var)), node_name, fillcolor='lightblue')  
            else:  
                dot.node(str(id(var)), str(type(var).__name__))  
            seen.add(var)  
            if hasattr(var, 'next_functions'):  
                for u in var.next_functions:  
                    if u[0] is not None:  
                        dot.edge(str(id(u[0])), str(id(var)))  
                        add_nodes(u[0])  
            if hasattr(var, 'saved_tensors'):  
                for t in var.saved_tensors:  
                    dot.edge(str(id(t)), str(id(var)))  
                    add_nodes(t)  
    add_nodes(var.grad_fn)  
    return dot  
