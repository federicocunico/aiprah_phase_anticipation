import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import SwinTransformer, Swin_T_Weights
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder
import torch.nn.functional as F
import math


class AdaptiveSwinTransformer(nn.Module):
    """SwinTransformer backbone adapted for variable channel input (RGB or RGB-D)"""
    def __init__(self, in_channels=3, pretrained=True, model_size="tiny"):
        super().__init__()
        self.in_channels = in_channels
        self.model_size = model_size
        
        # Load different Swin model sizes
        if model_size == "tiny":
            if pretrained:
                self.swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            else:
                self.swin = models.swin_t(weights=None)
        elif model_size == "small":
            if pretrained:
                self.swin = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            else:
                self.swin = models.swin_s(weights=None)
        elif model_size == "base":
            if pretrained:
                self.swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            else:
                self.swin = models.swin_b(weights=None)
        else:
            raise ValueError(f"Unknown model_size: {model_size}")
            
        # Adapt initial conv if needed for different input channels
        if in_channels != 3:
            orig_conv = self.swin.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=(orig_conv.bias is not None)
            )
            with torch.no_grad():
                if in_channels < 3:
                    new_conv.weight.copy_(orig_conv.weight[:, :in_channels])
                else:
                    new_conv.weight[:, :3] = orig_conv.weight
                    # Initialize additional channels (e.g., depth) as average of RGB
                    extra = orig_conv.weight.mean(dim=1, keepdim=True)
                    for i in range(3, in_channels):
                        new_conv.weight[:, i:i+1] = extra
                if orig_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
            self.swin.features[0][0] = new_conv
            
        # Remove classification head
        self.swin.head = nn.Identity()
        
        # Get output dimensions based on model size
        if model_size == "tiny":
            self.output_dim = 768
        elif model_size == "small":
            self.output_dim = 768
        elif model_size == "base":
            self.output_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        return self.swin(x)  # [B, output_dim]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0), :]


class TemporalEncoderDecoder(nn.Module):
    """Transformer encoder-decoder for temporal sequence processing"""
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # seq_first format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # seq_first format
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Learnable query tokens for decoder
        self.query_tokens = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, src: torch.Tensor, tgt_len: int = 1) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            tgt_len: number of output tokens (usually 1 for regression)
        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        batch_size, seq_len, _ = src.shape
        
        # Convert to seq_first format: [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Encode
        memory = self.encoder(src)  # [seq_len, batch_size, d_model]
        
        # Prepare decoder input
        tgt = self.query_tokens.expand(tgt_len, batch_size, -1)  # [tgt_len, batch_size, d_model]
        
        # Decode
        output = self.decoder(tgt, memory)  # [tgt_len, batch_size, d_model]
        
        # Convert back to batch_first: [batch_size, tgt_len, d_model]
        return output.transpose(0, 1)


class BERTTemporalProcessor(nn.Module):
    """Alternative BERT-based temporal processor"""
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_seq_len: int = 50,
        use_pretrained: bool = True,
        pretrained_model_name: str = "bert-base-uncased"
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            try:
                # Load pre-trained BERT model
                self.bert = BertModel.from_pretrained(pretrained_model_name)
                bert_hidden_size = self.bert.config.hidden_size
                
                # Add projection layer if dimensions don't match
                if bert_hidden_size != d_model:
                    self.input_projection = nn.Linear(d_model, bert_hidden_size)
                    self.output_projection = nn.Linear(bert_hidden_size, d_model)
                else:
                    self.input_projection = nn.Identity()
                    self.output_projection = nn.Identity()
                    
                print(f"Loaded pre-trained BERT: {pretrained_model_name}")
                
            except Exception as e:
                print(f"Could not load pre-trained BERT ({e}), using random initialization")
                use_pretrained = False
        
        if not use_pretrained:
            # Create BERT configuration for random initialization
            config = BertConfig(
                hidden_size=d_model,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_seq_len,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            
            # Initialize BERT model without embeddings (we provide our own)
            self.bert = BertModel(config, add_pooling_layer=True)
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()
        
    def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_embeddings: [batch_size, seq_len, d_model]
        Returns:
            pooled_output: [batch_size, d_model]
        """
        # Project to BERT's expected dimension if needed
        projected_embeddings = self.input_projection(sequence_embeddings)
        
        # BERT expects inputs_embeds directly
        outputs = self.bert(inputs_embeds=projected_embeddings)
        
        # Use pooler output if available, otherwise mean pool
        if outputs.pooler_output is not None:
            bert_output = outputs.pooler_output  # [batch_size, bert_hidden_size]
        else:
            bert_output = outputs.last_hidden_state.mean(dim=1)  # [batch_size, bert_hidden_size]
        
        # Project back to expected dimension
        return self.output_projection(bert_output)  # [batch_size, d_model]


class SwinTemporalRegressor(nn.Module):
    """
    Complete model: Swin Transformer spatial features -> Temporal Encoder-Decoder -> Regression
    """
    def __init__(
        self,
        sequence_length: int,
        num_classes: int = 7,
        time_horizon: int = 5,
        in_channels: int = 4,
        swin_model_size: str = "base",  # "tiny", "small", "base"
        temporal_processor: str = "transformer",  # "transformer" or "bert"
        d_model: int = None,  # Auto-inferred if None
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_pretrained_bert: bool = True,  # New parameter
        bert_model_name: str = "bert-base-uncased"  # New parameter
    ):
        super().__init__()
        
        # Spatial feature extractor
        self.backbone = AdaptiveSwinTransformer(
            in_channels=in_channels, 
            pretrained=True, 
            model_size=swin_model_size
        )
        
        # Determine d_model based on Swin output if not specified
        if d_model is None:
            d_model = self.backbone.output_dim
        
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Project Swin features to transformer dimension if needed
        if self.backbone.output_dim != d_model:
            self.feature_projection = nn.Linear(self.backbone.output_dim, d_model)
        else:
            self.feature_projection = nn.Identity()
        
        # Temporal processor
        if temporal_processor == "transformer":
            self.temporal_processor = TemporalEncoderDecoder(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=sequence_length + 10  # Some buffer
            )
        elif temporal_processor == "bert":
            self.temporal_processor = BERTTemporalProcessor(
                d_model=d_model,
                num_layers=num_encoder_layers + num_decoder_layers,
                num_heads=nhead,
                intermediate_size=dim_feedforward,
                max_seq_len=sequence_length + 10,
                use_pretrained=use_pretrained_bert,  # Use the parameter
                pretrained_model_name=bert_model_name  # Use the parameter
            )
        else:
            raise ValueError(f"Unknown temporal_processor: {temporal_processor}")
        
        self.temporal_processor_type = temporal_processor
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.time_horizon = time_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, channels, height, width]
        Returns:
            predictions: [batch_size, num_classes]
        """
        B, T, C, H, W = x.shape
        
        # Extract spatial features for each frame independently
        x_flat = x.view(B * T, C, H, W)
        spatial_features = self.backbone(x_flat)  # [B*T, spatial_dim]
        
        # Project to transformer dimension
        spatial_features = self.feature_projection(spatial_features)  # [B*T, d_model]
        
        # Reshape to sequence format
        sequence_embeddings = spatial_features.view(B, T, self.d_model)  # [B, T, d_model]
        
        # Process temporal sequence
        if self.temporal_processor_type == "transformer":
            # Encoder-decoder transformer
            temporal_output = self.temporal_processor(sequence_embeddings, tgt_len=1)  # [B, 1, d_model]
            final_features = temporal_output.squeeze(1)  # [B, d_model]
        else:  # BERT
            final_features = self.temporal_processor(sequence_embeddings)  # [B, d_model]
        
        # Final regression
        logits = self.regressor(final_features)  # [B, num_classes]
        
        return torch.sigmoid(logits) * self.time_horizon


def create_model(
    sequence_length: int = 8,
    num_classes: int = 7,
    time_horizon: int = 5,
    in_channels: int = 4,
    swin_model_size: str = "base",  # "tiny", "small", "base" 
    temporal_processor: str = "transformer",  # "transformer" or "bert"
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    nhead: int = 8,
    use_pretrained_bert: bool = True,  # New parameter
    bert_model_name: str = "bert-base-uncased"  # New parameter
) -> SwinTemporalRegressor:
    """
    Factory function to create the complete model
    """
    return SwinTemporalRegressor(
        sequence_length=sequence_length,
        num_classes=num_classes,
        time_horizon=time_horizon,
        in_channels=in_channels,
        swin_model_size=swin_model_size,
        temporal_processor=temporal_processor,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_pretrained_bert=use_pretrained_bert,  # Pass the parameter
        bert_model_name=bert_model_name  # Pass the parameter
    )


if __name__ == "__main__":
    # Test the model
    B, T, C, H, W = 4, 8, 4, 224, 224
    
    print("Testing Swin-Base + Transformer Encoder-Decoder...")
    model_transformer = create_model(
        sequence_length=T,
        in_channels=C,
        swin_model_size="base",
        temporal_processor="transformer",
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_transformer = model_transformer.to(device)
    x = torch.randn(B, T, C, H, W).to(device)
    
    with torch.no_grad():
        y_transformer = model_transformer(x)
    print(f"Transformer output shape: {y_transformer.shape}")
    if torch.cuda.is_available():
        print(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Test BERT version
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\nTesting Swin-Base + BERT Temporal Processor...")
    model_bert = create_model(
        sequence_length=T,
        in_channels=C,
        swin_model_size="base",
        temporal_processor="bert",
        num_encoder_layers=12,  # Total layers for BERT,
        bert_model_name="bert-base-uncased"  # Use a specific BERT model
    )
    
    model_bert = model_bert.to(device)
    with torch.no_grad():
        y_bert = model_bert(x)
    print(f"BERT output shape: {y_bert.shape}")
    if torch.cuda.is_available():
        print(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Test with different Swin sizes
    for swin_size in ["tiny", "small"]:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"\nTesting Swin-{swin_size.title()} + Transformer...")
        model_test = create_model(
            sequence_length=T,
            in_channels=C,
            swin_model_size=swin_size,
            temporal_processor="transformer",
            num_encoder_layers=4,
            num_decoder_layers=4
        )
        model_test = model_test.to(device)
        with torch.no_grad():
            y_test = model_test(x)
        print(f"Swin-{swin_size.title()} output shape: {y_test.shape}")
        if torch.cuda.is_available():
            print(f"GPU Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    hs = [1, 3, 5]
    for horizon in hs:
    # Test operativo
        m3d = create_model(
            sequence_length=4,              # Your T=4 frames
            num_classes=7,                  # Adjust based on your workflow phases
            time_horizon=horizon,                 # Keep as is for your regression target
            in_channels=4,                  # Your C=4 (RGB-D)
            swin_model_size="base",         # Best balance for surgical detail recognition
            temporal_processor="bert",       # Better for short sequences
            num_encoder_layers=8,           # Increased for surgical complexity
            num_decoder_layers=4,           # Lighter decoder for efficiency
            nhead=12,                       # More attention heads for fine details
            use_pretrained_bert=True,       # Leverage pre-trained knowledge
            bert_model_name="bert-base-uncased"  # Better performance than BERT
        )
        m2d = create_model(
            sequence_length=4,              # Your T=4 frames
            num_classes=7,                  # Adjust based on your workflow phases
            time_horizon=horizon,                 # Keep as is for your regression target
            in_channels=3,                  # Your C=4 (RGB-D)
            swin_model_size="base",         # Best balance for surgical detail recognition
            temporal_processor="bert",       # Better for short sequences
            num_encoder_layers=8,           # Increased for surgical complexity
            num_decoder_layers=4,           # Lighter decoder for efficiency
            nhead=12,                       # More attention heads for fine details
            use_pretrained_bert=True,       # Leverage pre-trained knowledge
            bert_model_name="bert-base-uncased"  # Better performance than BERT
        )

        fake_input = torch.randn(2, 4, 4, 224, 224)  # Batch size of 2, T=4 frames, C=4 channels
        m3d = m3d.to(device)
        m2d = m2d.to(device)
        optim = torch.optim.Adam(m3d.parameters(), lr=1e-4)
        output_3d = m3d(fake_input.to(device))
        output_2d = m2d(fake_input[:, :, :3, :, :].to(device))
        loss = F.mse_loss(output_3d, torch.zeros_like(output_3d).to(device))  # Dummy target
        optim.zero_grad()
        loss.backward()
        optim.step()


        print(f"Output shape for 3D model with horizon {horizon}: {output_3d.shape}")
        print(f"Output shape for 2D model with horizon {horizon}: {output_2d.shape}")   