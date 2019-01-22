import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from allennlp.nn.util import remove_sentence_boundaries
from . import layers

class FlowQA(nn.Module):
    """Network for the FlowQA Module."""
    def __init__(self, opt, embedding=None, padding_idx=0):
        super(FlowQA, self).__init__()

        # Input size to RNN: word emb + char emb + question emb + manual features
        doc_input_size = 0
        que_input_size = 0

        layers.set_my_dropout_prob(opt['my_dropout_p'])
        layers.set_seq_dropout(opt['do_seq_dropout'])

        if opt['use_wemb']:
            # Word embeddings
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
            if embedding is not None:
                self.embedding.weight.data = embedding
                if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                    opt['fix_embeddings'] = True
                    opt['tune_partial'] = 0
                    for p in self.embedding.parameters():
                        p.requires_grad = False
                else:
                    assert opt['tune_partial'] < embedding.size(0)
                    fixed_embedding = embedding[opt['tune_partial']:]
                    # a persistent buffer for the nn.Module
                    self.register_buffer('fixed_embedding', fixed_embedding)
                    self.fixed_embedding = fixed_embedding
            embedding_dim = opt['embedding_dim']
            doc_input_size += embedding_dim
            que_input_size += embedding_dim
        else:
            opt['fix_embeddings'] = True
            opt['tune_partial'] = 0

        if opt['CoVe_opt'] > 0:
            self.CoVe = layers.MTLSTM(opt, embedding)
            CoVe_size = self.CoVe.output_size
            doc_input_size += CoVe_size
            que_input_size += CoVe_size

        if opt['use_elmo']:
            options_file = "glove/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            weight_file = "glove/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            doc_input_size += 1024
            que_input_size += 1024
        if opt['use_pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            doc_input_size += opt['pos_dim']
        if opt['use_ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            doc_input_size += opt['ner_dim']

        if opt['do_prealign']:
            self.pre_align = layers.GetAttentionHiddens(embedding_dim, opt['prealign_hidden'], similarity_attention=True)
            doc_input_size += embedding_dim
        if opt['no_em']:
            doc_input_size += opt['num_features'] - 3
        else:
            doc_input_size += opt['num_features']

        # Setup the vector size for [doc, question]
        # they will be modified in the following code
        doc_hidden_size, question_hidden_size = doc_input_size, que_input_size
        print('Initially, the vector_sizes [doc, query] are', doc_hidden_size, question_hidden_size)

        flow_size = opt['hidden_size']

        # RNN document encoder
        self.doc_rnn1 = layers.StackedBRNN(doc_hidden_size, opt['hidden_size'], num_layers=1)
        self.dialog_flow1 = layers.StackedBRNN(opt['hidden_size'] * 2, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        self.doc_rnn2 = layers.StackedBRNN(opt['hidden_size'] * 2 + flow_size + CoVe_size, opt['hidden_size'], num_layers=1)
        self.dialog_flow2 = layers.StackedBRNN(opt['hidden_size'] * 2, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        doc_hidden_size = opt['hidden_size'] * 2

        # RNN question encoder
        self.question_rnn, question_hidden_size = layers.RNN_from_opt(question_hidden_size, opt['hidden_size'], opt,
        num_layers=2, concat_rnn=opt['concat_rnn'], add_feat=CoVe_size)

        # Output sizes of rnn encoders
        print('After Input LSTM, the vector_sizes [doc, query] are [', doc_hidden_size, question_hidden_size, '] * 2')

        # Deep inter-attention
        self.deep_attention = layers.DeepAttention(opt, abstr_list_cnt=2, deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], do_similarity=opt['deep_inter_att_do_similar'], word_hidden_size=embedding_dim + CoVe_size, no_rnn=True)

        self.deep_attn_rnn, doc_hidden_size = layers.RNN_from_opt(self.deep_attention.att_final_size + flow_size, opt['hidden_size'], opt, num_layers=1)
        self.dialog_flow3 = layers.StackedBRNN(doc_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)

        # Question understanding and compression
        self.high_lvl_question_rnn, question_hidden_size = layers.RNN_from_opt(question_hidden_size * 2, opt['hidden_size'], opt, num_layers = 1, concat_rnn = True)

        # Self attention on context
        att_size = doc_hidden_size + 2 * opt['hidden_size'] * 2

        if opt['self_attention_opt'] > 0:
            self.high_lvl_self_attention = layers.GetAttentionHiddens(att_size, opt['deep_att_hidden_size_per_abstr'])
            self.high_lvl_doc_rnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size * 2 + flow_size, opt['hidden_size'], opt, num_layers = 1, concat_rnn = False)
            print('Self deep-attention {} rays in {}-dim space'.format(opt['deep_att_hidden_size_per_abstr'], att_size))
        elif opt['self_attention_opt'] == 0:
            self.high_lvl_doc_rnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size + flow_size, opt['hidden_size'], opt, num_layers = 1, concat_rnn = False)

        print('Before answer span finding, hidden size are', doc_hidden_size, question_hidden_size)

        # Question merging
        self.self_attention = layers.LinearSelfAttn(question_hidden_size)
        if opt['do_hierarchical_query']:
            self.hier_query_rnn = layers.StackedBRNN(question_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
            question_hidden_size = opt['hidden_size']

        # Attention for span start/end
        self.get_answer = layers.GetSpanStartEnd(doc_hidden_size, question_hidden_size, opt,
        opt['ptr_net_indep_attn'], opt["ptr_net_attn_type"], opt['do_ptr_update'])

        self.ans_type_prediction = layers.BilinearLayer(doc_hidden_size * 2, question_hidden_size, opt['answer_type_num'])

        # Store config
        self.opt = opt

    def forward(self, doc, doc_char, doc_feature, doc_pos, doc_ner, doc_mask,
                n_question, n_question_char, n_question_mask):
        """Inputs:
        doc = document word indices             [batch , len_d]
        doc_char = document char indices           [batch , len_d , len_w] or [1]
        doc_feature = document word features indices  [batch , q_num , len_d , nfeat]
        doc_pos = document POS tags             [batch , len_d]
        doc_ner = document entity tags          [batch , len_d]
        doc_mask = document padding mask        [batch , len_d]
        n_question = question word indices        [batch , q_num , len_q]
        n_question_char = question char indices           [(batch , q_num) , len_q , len_w]
        n_question_mask = question padding mask   [batch , q_num , len_q]
        """

        # precomputing ELMo is only for context (to speedup computation)
        if self.opt['use_elmo'] and self.opt['elmo_batch_size'] > self.opt['batch_size']: # precomputing ELMo is used
            if doc_char.dim() != 1: # precomputation is needed
                precomputed_bilm_output = self.elmo._elmo_lstm(doc_char)
                self.precomputed_layer_activations = [t.detach().cpu() for t in precomputed_bilm_output['activations']]
                self.precomputed_mask_with_bos_eos = precomputed_bilm_output['mask'].detach().cpu()
                self.precomputed_cnt = 0

            # get precomputed ELMo
            layer_activations = [t[doc.size(0) * self.precomputed_cnt: doc.size(0) * (self.precomputed_cnt + 1), :, :] for t in self.precomputed_layer_activations]
            mask_with_bos_eos = self.precomputed_mask_with_bos_eos[doc.size(0) * self.precomputed_cnt: doc.size(0) * (self.precomputed_cnt + 1), :]
            if doc.is_cuda:
                layer_activations = [t.cuda() for t in layer_activations]
                mask_with_bos_eos = mask_with_bos_eos.cuda()

            representations = []
            for i in range(len(self.elmo._scalar_mixes)):
                scalar_mix = getattr(self.elmo, 'scalar_mix_{}'.format(i))
                representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                        representation_with_bos_eos, mask_with_bos_eos
                )
                representations.append(self.elmo._dropout(representation_without_bos_eos))

            doc_elmo = representations[0][:, :doc.size(1), :]
            self.precomputed_cnt += 1

            precomputed_elmo = True
        else:
            precomputed_elmo = False

        """
        n_doc = document word indices        [batch * question_num * len_doc]
        n_doc_mask = document padding mask   [batch * question_num * len_doc]
        """
        n_doc = doc.unsqueeze(1).expand(n_question.size(0), n_question.size(1), doc.size(1)).contiguous()
        n_doc_mask = doc_mask.unsqueeze(1).expand(n_question.size(0), n_question.size(1), doc.size(1)).contiguous()

        doc_rnn_input_list, question_rnn_input_list = [], []

        n_question = n_question.view(-1, n_question.size(-1)) # 去掉1的batch
        n_question_mask = n_question_mask.view(-1, n_question.size(-1)) # 去掉1的batch

        if self.opt['use_wemb']:
            # Word embedding for both document and question
            emb = self.embedding if self.training else self.eval_embed
            doc_emb = emb(doc)
            n_question_emb = emb(n_question)
            # Dropout on embeddings
            if self.opt['dropout_emb'] > 0:
                doc_emb = layers.dropout(doc_emb, p=self.opt['dropout_emb'], training=self.training)
                n_question_emb = layers.dropout(n_question_emb, p=self.opt['dropout_emb'], training=self.training)

            doc_rnn_input_list.append(doc_emb)
            question_rnn_input_list.append(n_question_emb)

        if self.opt['CoVe_opt'] > 0:
            doc_cove_mid, doc_cove_high = self.CoVe(doc, doc_mask)
            n_question_cove_mid, n_question_cove_high = self.CoVe(n_question, n_question_mask)
            # Dropout on contexualized embeddings
            if self.opt['dropout_emb'] > 0:
                doc_cove_mid = layers.dropout(doc_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                doc_cove_high = layers.dropout(doc_cove_high, p=self.opt['dropout_emb'], training=self.training)
                n_question_cove_mid = layers.dropout(n_question_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                n_question_cove_high = layers.dropout(n_question_cove_high, p=self.opt['dropout_emb'], training=self.training)

            doc_rnn_input_list.append(doc_cove_mid)
            question_rnn_input_list.append(n_question_cove_mid)

        if self.opt['use_elmo']:
            if not precomputed_elmo:
                doc_elmo = self.elmo(doc_char)['elmo_representations'][0]#torch.zeros(x1_emb.size(0), x1_emb.size(1), 1024, dtype=x1_emb.dtype, layout=x1_emb.layout, device=x1_emb.device)
            n_question_elmo = self.elmo(n_question_char)['elmo_representations'][0]#torch.zeros(x2_emb.size(0), x2_emb.size(1), 1024, dtype=x2_emb.dtype, layout=x2_emb.layout, device=x2_emb.device)
            # Dropout on contexualized embeddings
            if self.opt['dropout_emb'] > 0:
                doc_elmo = layers.dropout(doc_elmo, p=self.opt['dropout_emb'], training=self.training)
                n_question_elmo = layers.dropout(n_question_elmo, p=self.opt['dropout_emb'], training=self.training)

            doc_rnn_input_list.append(doc_elmo)
            question_rnn_input_list.append(n_question_elmo)

        if self.opt['use_pos']:
            doc_pos_emb = self.pos_embedding(doc_pos)
            doc_rnn_input_list.append(doc_pos_emb)

        if self.opt['use_ner']:
            doc_ner_emb = self.ner_embedding(doc_ner)
            doc_rnn_input_list.append(doc_ner_emb)

        doc_input = torch.cat(doc_rnn_input_list, dim=2)
        n_question_input = torch.cat(question_rnn_input_list, dim=2)

        def expansion_for_doc(z):
            return z.unsqueeze(1).expand(z.size(0), n_question.size(0), z.size(1), z.size(2)).contiguous().view(-1, z.size(1), z.size(2))

        n_doc_emb = expansion_for_doc(doc_emb) # [1,384,300] -> [10,384,300]
        n_doc_cove_high = expansion_for_doc(doc_cove_high)
        #x1_elmo_expand = expansion_for_doc(x1_elmo)
        if self.opt['no_em']:
            doc_feature = doc_feature[:, :, :, 3:]

        n_doc_input = torch.cat([expansion_for_doc(doc_input), doc_feature.view(-1, doc_feature.size(-2), doc_feature.size(-1))], dim=2) # [10,384,1951]
        n_doc_mask = n_doc_mask.view(-1, n_doc_mask.size(-1)) # [10,384]

        if self.opt['do_prealign']:
            doc_atten = self.pre_align(n_doc_emb, n_question_emb, n_question_mask) # [10,384,300],[10,20,300],[10,20]
            n_doc_input = torch.cat([n_doc_input, doc_atten], dim=2)

        # === Start processing the dialog ===
        # cur_h: [batch_size * max_qa_pair, context_length, hidden_state]
        # flow : fn (rnn)
        # x1_full: [batch_size, max_qa_pair, context_length]
        def flow_operation(cur_h, flow):
            flow_in = cur_h.transpose(0, 1).view(n_doc.size(2), n_doc.size(0), n_doc.size(1), -1)
            flow_in = flow_in.transpose(0, 2).contiguous().view(n_doc.size(1), n_doc.size(0) * n_doc.size(2), -1).transpose(0, 1)
            # [batch * context_length, max_qa_pair, hidden_state]
            flow_out = flow(flow_in)
            # [batch * context_length, max_qa_pair, flow_hidden_state_dim (hidden_state/2)]
            if self.opt['no_dialog_flow']:
                flow_out = flow_out * 0

            flow_out = flow_out.transpose(0, 1).view(n_doc.size(1), n_doc.size(0), n_doc.size(2), -1).transpose(0, 2).contiguous()
            flow_out = flow_out.view(n_doc.size(2), n_doc.size(0) * n_doc.size(1), -1).transpose(0, 1)
            # [batch * max_qa_pair, context_length, flow_hidden_state_dim]
            return flow_out

        """
            Encode document with RNN
        """
        n_doc_rnn_list = []

        n_doc_hidden = self.doc_rnn1(n_doc_input, n_doc_mask)
        n_doc_hidden_flow = flow_operation(n_doc_hidden, self.dialog_flow1)

        n_doc_rnn_list.append(n_doc_hidden)

        n_doc_hidden = self.doc_rnn2(torch.cat((n_doc_hidden, n_doc_hidden_flow, n_doc_cove_high), dim=2), n_doc_mask)
        n_doc_hidden_flow = flow_operation(n_doc_hidden, self.dialog_flow2)
        n_doc_rnn_list.append(n_doc_hidden)

        #with open('flow_bef_att.pkl', 'wb') as output:
        #    pickle.dump(doc_hiddens_flow, output, pickle.HIGHEST_PROTOCOL)
        #while(1):
        #    pass
        """
            Encode question with RNN
        """
        _, n_question_rnn_list = self.question_rnn(n_question_input, n_question_mask,
                                                   return_list=True, additional_x=n_question_cove_high)

        # Final question layer
        n_question_hidden = self.high_lvl_question_rnn(torch.cat(n_question_rnn_list, 2), n_question_mask) # [10,20,250]
        n_question_rnn_list += [n_question_hidden]
        """
            Main Attention Fusion Layer On Doc + Question
        """
        doc_info = self.deep_attention([torch.cat([n_doc_emb, n_doc_cove_high], 2)], n_doc_rnn_list,
                                       [torch.cat([n_question_emb, n_question_cove_high], 2)], n_question_rnn_list,
                                       n_doc_mask, n_question_mask)  #[10,384,1250]

        n_doc_hidden = self.deep_attn_rnn(torch.cat((doc_info, n_doc_hidden_flow), dim=2), n_doc_mask) # [10,384,1250],[10,384,125]->[10,384,250]
        n_doc_hidden_flow = flow_operation(n_doc_hidden, self.dialog_flow3) # [10,384,125]

        n_doc_rnn_list += [n_doc_hidden]
        """
            Self Attention Fusion Layer On Doc-Attention-Question
        """
        n_doc_attn = torch.cat(n_doc_rnn_list, 2) # [10,384,750]

        if self.opt['self_attention_opt'] > 0:
            highlvl_self_attn_hiddens = self.high_lvl_self_attention(n_doc_attn, n_doc_attn, n_doc_mask, x3=n_doc_hidden, drop_diagonal=True)
            n_doc_hidden = self.high_lvl_doc_rnn(torch.cat([n_doc_hidden, highlvl_self_attn_hiddens, n_doc_hidden_flow], dim=2), n_doc_mask)
        elif self.opt['self_attention_opt'] == 0:
            n_doc_hidden = self.high_lvl_doc_rnn(torch.cat([n_doc_hidden, n_doc_hidden_flow], dim=2), n_doc_mask)

        n_doc_rnn_list += [n_doc_hidden]

        # Merge the question hidden vectors
        """
            Self Attention On N Questions
        """
        question_merge_weights = self.self_attention(n_question_hidden, n_question_mask) # [10,20,250]->[10,20]
        question_avg_hidden = layers.weighted_avg(n_question_hidden, question_merge_weights) # [10,20,250],[10,20]->[10,250]
        if self.opt['do_hierarchical_query']:
            question_avg_hidden = self.hier_query_rnn(question_avg_hidden.view(n_doc.size(0), n_doc.size(1), -1))
            question_avg_hidden = question_avg_hidden.contiguous().view(-1, question_avg_hidden.size(-1)) # [10,125]

        # Get Start, End span
        start_scores, end_scores = self.get_answer(n_doc_hidden, question_avg_hidden, n_doc_mask) # [10,384,250],[10,125]->[10,384]
        all_start_scores = start_scores.view_as(n_doc)     # batch x q_num x len_d
        all_end_scores = end_scores.view_as(n_doc)         # batch x q_num x len_d

        # Get whether there is an answer
        doc_avg_hidden = torch.cat((torch.max(n_doc_hidden, dim=1)[0], torch.mean(n_doc_hidden, dim=1)), dim=1)
        class_scores = self.ans_type_prediction(doc_avg_hidden, question_avg_hidden)
        all_class_scores = class_scores.view(n_doc.size(0), n_doc.size(1), -1)      # batch x q_num x class_num
        all_class_scores = all_class_scores.squeeze(-1) # when class_num = 1

        return all_start_scores, all_end_scores, all_class_scores
