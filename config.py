import argparse

def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--method', type=str, default="hotflip", help='hotflip  hotflip_raw or aggd')
    parser.add_argument('--attack_dataset', type=str, default="arguana", help='BEIR dataset to evaluate,  nq, nq-train, msmarco  arguana fiqa')
    parser.add_argument('--split', type=str, default='train',help='train, test') # Use the query of the training set to generate clusters, and then use the generated documents to attack
    parser.add_argument('--attack_model_code', type=str, default='contriever',help='contriever, contriever-msmarco, dpr-single, dpr-multi,ance')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_query_length', type=int, default=32)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument('--attack_query', default=True, help="whether attack the query or documents")

    parser.add_argument("--num_adv_passage_tokens", default=50, type=int)
    parser.add_argument("--num_cand", default=100, type=int) #  top-100 tokens as candidates
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--num_iter", default=5000, type=int)
    parser.add_argument("--num_grad_iter", default=1, type=int)

    parser.add_argument("--output_file", default=None, type=str)

    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--kmeans_split", default=0, type=int, help="which split in kmeans clustering, use s to denote")
    parser.add_argument("--do_kmeans", default=True, action="store_true")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument("--init_gold", dest='init_gold', action='store_true', help="If specified, init with gold passages") #Dual switch setting
    parser.add_argument("--no_init_gold", dest='init_gold', action='store_false', help="If specified, do not init with gold passages")
    
    parser.add_argument("--dont_init_gold", action="store_true", help="if ture, do not init with gold passages") # 默认值是 False

    # embedding_index
    parser.add_argument('--result_output', default="results/beir_result", type=str)
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])

    # evaluate_attack
    parser.add_argument("--beir_results_path", type=str, default='beir_result', help='Eval results path of eval_model on the beir eval_dataset')
    parser.add_argument("--eval_res_path", type=str, default="eval_result")
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance", 'tas-b','dragon','condenser'])
    parser.add_argument('--eval_dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')

    #corpus attack
    parser.add_argument("--attack_rate",type=float,default=0.0001, help="the rate to attack the corpus")

    args = parser.parse_args()

    return args