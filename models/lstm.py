import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Simple LSTM model for time series forecasting.
    Compatible with Informer's interface.
    """

    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=512,
        dropout=0.1,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        device=torch.device("cuda:0"),
    ):
        super(LSTM, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.output_attention = output_attention
        
        self.encoder_lstm = nn.LSTM(
            input_size=enc_in,
            hidden_size=d_model,
            num_layers=e_layers,
            dropout=dropout if e_layers > 1 else 0,
            batch_first=True,
        )
        
        self.projection = nn.Linear(d_model, c_out)
        
        self.output_layer = nn.Linear(seq_len, out_len)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # x_enc: [Batch, seq_len, enc_in]
        
        lstm_out, (hidden, cell) = self.encoder_lstm(x_enc)  # [B, seq_len, d_model]
        
        projected = self.projection(lstm_out)  # [B, seq_len, c_out]
        
        # [B, c_out, seq_len]
        projected_t = projected.transpose(1, 2)
        
        output = self.output_layer(projected_t)  # [B, c_out, pred_len]
        
        # [B, pred_len, c_out]
        output = output.transpose(1, 2)
        
        if self.output_attention:
            return output, None
        else:
            return output

