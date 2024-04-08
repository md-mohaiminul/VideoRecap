import torch.nn as nn

from .qformer import BertConfig, BertLMHeadModel
from .coca import CrossAttention, LayerNorm

class PercieverMapper(nn.Module):
    def __init__(self, query_len, query_dim, context_dim, finetune_mapper):
        super().__init__()
        
        self.attn_pool = CrossAttention(
            dim=query_dim, context_dim=context_dim,
            dim_head=64, heads=8,
            norm_context=True
        )
        self.attn_pool_norm = LayerNorm(query_dim)
        
    def forward(self, query, context, context_mask = None):
        query = self.attn_pool(query, context)
        query = self.attn_pool_norm(query)
        return query
        
class QformerMapper(nn.Module):
    def __init__(self, query_len, query_dim, context_dim, finetune_mapper):
        super().__init__()
        
        encoder_config = BertConfig.from_pretrained("distilbert-base-uncased")
        encoder_config.encoder_width = context_dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = query_len
        self.mapper = BertLMHeadModel.from_pretrained(
            "distilbert-base-uncased", config=encoder_config
        )
        if not finetune_mapper:
            self.mapper.freeze_lm_weights()
        
    def forward(self, query, context, context_mask = None):
        query_output = self.mapper.bert(
            query_embeds=query,
            encoder_hidden_states=context,
            encoder_attention_mask=context_mask,
            return_dict=True,
        )
        return query_output.last_hidden_state
    
class TransformerMapper(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, query, context, context_mask = None):
        return self.encoder(context)
    
def get_mapper(mapper_type, num_queries, query_dim, context_dim, finetune_mapper=False):
    if mapper_type=='perciever':
        return PercieverMapper(num_queries, query_dim, context_dim, finetune_mapper=finetune_mapper)
    elif mapper_type=='qformer':
        return QformerMapper(num_queries, query_dim, context_dim, finetune_mapper=finetune_mapper)
    elif mapper_type=='transformer':
        return TransformerMapper(d_model=context_dim)
    else:
        raise NotImplementedError
    
        