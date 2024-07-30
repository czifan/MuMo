'''
This file implements the Feed forward network
Adapted from OpenNMT-Py
'''

from torch import nn

class FFN(nn.Module):
    def __init__(self, input_dim, scale, output_dim=None, p=0.1, expansion=False):
        super(FFN, self).__init__()
        output_dim = input_dim if output_dim is None else output_dim

        proj_features = input_dim * scale if expansion else input_dim // scale
        self.w_1 = nn.Linear(input_dim, proj_features)
        self.w_2 = nn.Linear(proj_features, output_dim)

        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p)
        self.residual = True if input_dim == output_dim else False

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x if self.residual else output