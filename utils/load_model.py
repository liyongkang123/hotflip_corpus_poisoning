import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.amp import autocast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import torch
import logging
from transformers import AutoModel,AutoTokenizer
logger = logging.getLogger(__name__)

from .utils import model_code_to_qmodel_name,model_code_to_cmodel_name,get_model_prompts_tasks

class HFtoSF(torch.nn.Module):
    def __init__(self, hf_model, hf_tokenizer, prompt="", normalize= False, pooling='last', max_seq_length=8192 ,device='cuda'):
        super().__init__()  # <--- This line is required
        try:
            self.hf_model = hf_model.to(device)
            self.hf_model.eval()
        except: # for BGE M3, no attribute 'eval' and 'to'
            self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.max_seq_length = max_seq_length  # Adjust according to specific model
        self.pooling = pooling
        self.normalize = normalize
        self.prompt = str(prompt)+" " # Define query prompt or context prompt at initialization, add a space at the end
                    # 1. Convert prompt to token ids (without adding special tokens, as we will insert in the middle)
        self.prompt_tokens = self.hf_tokenizer(self.prompt, padding=False, add_special_tokens=False, return_tensors='pt')

        if pooling == 'mask_prompt_mean': # Pooling method designed specifically for ReasonIR
            self.prompt_tokens = self.hf_tokenizer(self.prompt, padding=False, add_special_tokens=True, return_tensors='pt')

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['mask_prompt_mean']:
            attention_mask[:, :len(self.prompt_tokens['input_ids'][0])] = 0
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
             
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def tokenize_texts(self, texts):
        batch_dict = self.hf_tokenizer(texts, max_length=self.max_seq_length, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, pad_to_multiple_of=8)
        batch_dict = { k: v.to(self.device) for k, v in batch_dict.items() }
        return batch_dict

    def encode(self, texts, convert_to_numpy=False, show_progress_bar=False):
        # Input is a text list that has already been batched
        # Output is the corresponding embeddings
        if isinstance(texts, str):
            texts = [texts]
        # Prepend prompt to texts
        if self.prompt != '':
            texts = [self.prompt + text for text in texts]
        batch_inputs = self.tokenize_texts(texts)
        with torch.no_grad():
            outputs = self.hf_model(**batch_inputs)
            embeddings = self._pooling(outputs.last_hidden_state, batch_inputs['attention_mask'])
        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings

    def encode_tokenized(self, tokenized_inputs, convert_to_numpy=False, show_progress_bar=False):
        # Safety check for Llama/Qwen models: remove position_ids so the model can auto-generate based on new length
        if 'position_ids' in tokenized_inputs:
            del tokenized_inputs['position_ids']

        # How to add prompt to tokenized_inputs
        if self.prompt.strip(): # If prompt is not empty
                         # Ensure device consistency
            device = tokenized_inputs['input_ids'].device
            prompt_ids = self.prompt_tokens['input_ids'].to(device)
            
            batch_size = tokenized_inputs['input_ids'].shape[0]
            prompt_len = prompt_ids.shape[1]
            
            # 2. Expand prompt to the current batch size
            prompt_ids = prompt_ids.repeat(batch_size, 1)
            prompt_mask = torch.ones((batch_size, prompt_len), device=device, dtype=tokenized_inputs['attention_mask'].dtype)
            
            # 3. Concatenation operation: [CLS] + Prompt + Text ...
            # Assuming input_ids[:, 0] is [CLS]
            tokenized_inputs['input_ids'] = torch.cat([
                tokenized_inputs['input_ids'][:, :1], 
                prompt_ids, 
                tokenized_inputs['input_ids'][:, 1:]
            ], dim=1)
            
            tokenized_inputs['attention_mask'] = torch.cat([
                tokenized_inputs['attention_mask'][:, :1], 
                prompt_mask, 
                tokenized_inputs['attention_mask'][:, 1:]
            ], dim=1)
            
            # If token_type_ids exists (BERT and similar models), usually Prompt also belongs to the first sentence (Type 0)
            if 'token_type_ids' in tokenized_inputs:
                prompt_type_ids = torch.zeros((batch_size, prompt_len), device=device, dtype=tokenized_inputs['token_type_ids'].dtype)
                tokenized_inputs['token_type_ids'] = torch.cat([
                    tokenized_inputs['token_type_ids'][:, :1], 
                    prompt_type_ids, 
                    tokenized_inputs['token_type_ids'][:, 1:]
                ], dim=1)
        
        # with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = self.hf_model(**tokenized_inputs)
        embeddings = self._pooling(outputs.last_hidden_state, tokenized_inputs['attention_mask'])
        embeddings = embeddings.float()
        
        # Clean up intermediate variables promptly
        del outputs
        torch.cuda.empty_cache()  # Optional: use when memory is tight

        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings
        


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":  # average pooling
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def bi_encoder_senctence_transformer_get_emb(model, input): #  All models provided by SentenceTransformer
    # here tas-b ， ance, dragon, condenser are from SentenceTransformer
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def llm_get_emb(model, input):
    # Here the input is tokenized input
    return model.encode_tokenized(input)

 
        

def load_models(model_code, datasets_name=""):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' == model_code:
        q_model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = q_model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code],use_fast=True)
        get_emb = contriever_get_emb
    elif 'dpr' in model_code:
        q_model = DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    elif any(sub_model_code in model_code for sub_model_code in ['ance', 'tas', 'condenser']):
        q_model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = q_model
        tokenizer = q_model.tokenizer
        get_emb = bi_encoder_senctence_transformer_get_emb
    elif 'dragon' in model_code:
        q_model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        c_model = SentenceTransformer(model_code_to_cmodel_name[model_code])
        tokenizer = q_model.tokenizer
        get_emb = bi_encoder_senctence_transformer_get_emb
    
    elif 'contriever-msmarco' == model_code:
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco",cache_dir=os.getenv('HF_HOME')) # this is the latest version
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']

        q_model_base = AutoModel.from_pretrained("facebook/contriever-msmarco", torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME')) #torch_dtype=torch.bfloat16
        q_model = HFtoSF(q_model_base, tokenizer,  prompt=q_prompt, normalize=False , pooling='mean', max_seq_length = 512, device='cuda') # use dot 

        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=False , pooling='mean', max_seq_length = 512, device='cuda')
        get_emb = llm_get_emb


    elif 'reasonir' == model_code:
        # This is the most special case, requires using SentenceTransformer, will handle separately later
        
        model_kwargs = {
        "torch_dtype": torch.bfloat16,  # Use bfloat16, more stable and better performance
        # "attn_implementation": "flash_attention_2",  # Use Flash Attention 2 for efficiency
        }
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        # q_model = SentenceTransformer("reasonir/ReasonIR-8B", trust_remote_code=True, model_kwargs=model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained("reasonir/ReasonIR-8B",cache_dir=os.getenv('HF_HOME'),trust_remote_code=True,)
        q_model_base = AutoModel.from_pretrained("reasonir/ReasonIR-8B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='mask_prompt_mean', device='cuda')
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='mask_prompt_mean', device='cuda')
        get_emb = llm_get_emb


    elif 'bge_reasoner' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928",cache_dir=os.getenv('HF_HOME')) # this is the latest version
        q_model_base = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb

    elif 'diver' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("AQ-MedAI/Diver-Retriever-4B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-4B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-4B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb
        
    elif 'diver_1.7B' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("AQ-MedAI/Diver-Retriever-1.7B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-1.7B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-4B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb
    elif 'diver_0.6B' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("AQ-MedAI/Diver-Retriever-0.6B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-0.6B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-4B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb

    elif 'qwen3' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb
    
    elif 'qwen3_4B' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb

    elif 'qwen3_0.6B' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb


    elif 'gte' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb
    
    elif 'linq' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb

    elif 'bge_m3' == model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("BAAI/bge-m3",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='cls', device='cuda')
        # c_model = AutoModel.from_pretrained("BAAI/bge-m3", )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='cls', device='cuda')
        get_emb = llm_get_emb
        
    else:
        raise NotImplementedError

    return q_model, c_model, tokenizer, get_emb