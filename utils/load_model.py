import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.amp import autocast
from transformers import AutoTokenizer
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
        super().__init__()  # <--- 必须加上这一行
        try:
            self.hf_model = hf_model.to(device)
            self.hf_model.eval()
        except: # for BGE M3, no attribute 'eval' and 'to'
            self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.max_seq_length = max_seq_length  # 根据具体模型调整
        self.pooling = pooling
        self.normalize = normalize
        self.prompt = str(prompt)+" " # 初始化的时候就定义好query prompt或者context prompt 后面加一个空格
                    # 1. 将 prompt 转换为 token ids (不添加 special tokens，因为我们要插在中间)
        self.prompt_tokens = self.hf_tokenizer(self.prompt, padding=False, add_special_tokens=False, return_tensors='pt')

        if pooling == 'mask_prompt_mean': # 专为 ReasonIR 设计的 pooling 方法
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
        # 输入的是已经经过了batch size 之后的文本list
        # 输出的是对应的embeddings
        if isinstance(texts, str):
            texts = [texts]
        # 把prompt 拼接到texts 前面
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
        # 针对 Llama/Qwen 等模型的安全检查：移除 position_ids 以便模型根据新长度自动生成
        if 'position_ids' in tokenized_inputs:
            del tokenized_inputs['position_ids']

        # 怎么把tokenized_inputs 加上prompt
        if self.prompt.strip(): # 如果prompt不为空
                         # 确保设备一致
            device = tokenized_inputs['input_ids'].device
            prompt_ids = self.prompt_tokens['input_ids'].to(device)
            
            batch_size = tokenized_inputs['input_ids'].shape[0]
            prompt_len = prompt_ids.shape[1]
            
            # 2. 扩展 prompt 到当前 batch 的大小
            prompt_ids = prompt_ids.repeat(batch_size, 1)
            prompt_mask = torch.ones((batch_size, prompt_len), device=device, dtype=tokenized_inputs['attention_mask'].dtype)
            
            # 3. 拼接操作： [CLS] + Prompt + Text ...
            # 假设 input_ids[:, 0] 是 [CLS]
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
            
            # 如果有 token_type_ids (BERT等模型)，通常 Prompt 也属于第一句 (Type 0)
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
        
        # 及时清理中间变量
        del outputs
        torch.cuda.empty_cache()  # 可选：在内存紧张时使用

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
    # 这里的input 是 tokenized input
    return model.encode_tokenized(input)

 
        

def load_models(model_code, datasets_name=""):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    # if 'contriever' in model_code:
    #     q_model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
    #     assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
    #     c_model = q_model
    #     tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code],use_fast=True)
    #     get_emb = contriever_get_emb
    if 'dpr' in model_code:
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


    elif 'reasonir' in model_code:
        # 这个是最为特殊的，需要使用SentenceTransformer， 稍后我单独处理
        
        model_kwargs = {
        "torch_dtype": torch.bfloat16,  # 使用bfloat16，更稳定且性能好
        # "attn_implementation": "flash_attention_2",  # 使用Flash Attention 2提高效率
        }
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        # q_model = SentenceTransformer("reasonir/ReasonIR-8B", trust_remote_code=True, model_kwargs=model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained("reasonir/ReasonIR-8B",cache_dir=os.getenv('HF_HOME'))
        # q_model.max_seq_length = 8192
        #彻底改为 AutoModel + HFtoSF 的形式
        q_model_base = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype=torch.bfloat16, trust_remote_code=True,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='mask_prompt_mean', device='cuda')
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='mask_prompt_mean', device='cuda')
        get_emb = llm_get_emb


    elif 'bge_reasoner' in model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928",cache_dir=os.getenv('HF_HOME')) # this is the latest version
        q_model_base = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb

    elif 'diver' in model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("AQ-MedAI/Diver-Retriever-4B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-4B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("AQ-MedAI/Diver-Retriever-4B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb
        
    elif 'qwen3' in model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb


    elif 'gte' in model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb
    
    elif 'linq' in model_code:
        prompts = get_model_prompts_tasks(model_name=model_code,dataset_name=datasets_name)
        q_prompt = prompts['query']
        c_prompt = prompts['passage']
        tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral",cache_dir=os.getenv('HF_HOME'))
        q_model_base = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral",trust_remote_code=True , torch_dtype=torch.bfloat16,cache_dir=os.getenv('HF_HOME'))
        q_model = HFtoSF(q_model_base, tokenizer, prompt=q_prompt, normalize=True , pooling='last', device='cuda')
        # c_model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral" )
        c_model = HFtoSF(q_model_base, tokenizer, prompt=c_prompt, normalize=True , pooling='last', device='cuda')
        get_emb = llm_get_emb

    elif 'bge_m3' in model_code:
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