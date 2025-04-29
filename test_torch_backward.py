import unittest
import random
import torch
from neural_net_model import NeuralNetworkModel

class TestTorchBackward(unittest.TestCase):

    def test_atomic_manual_grad(self):
        ## create model
        block_size = 3
        embed_size = 10
        hidden_size = 64
        vocab_size = 27
        model = NeuralNetworkModel("test",
                                   [vocab_size, embed_size, embed_size*block_size, hidden_size, vocab_size],
                                   "xavier", "random",
                                   ["embedding", "linear", "batchnorm", "tanh", "linear", "softmax"],
                                   None)
        batch_size = model.training_buffer_size

        ## Add enough data to meet the training buffer size as batch size
        sample_input = torch.randint(low=0, high=vocab_size, size=(batch_size, block_size))
        target = [[random.randint(0, vocab_size - 1)] for _ in range(batch_size)]

        ## Forward pass
        for p in model.params:
            p.requires_grad_()
        activations, cost = model._forward(sample_input, target)

        ## torch backward
        for p in model.params:
            p.grad = None
        for a in activations:
            a.retain_grad()
        cost.backward()

        ## Unpack tensors
        emb, h_pre_bn, h_pre_act, h, logits, probs = tuple(activations)
        c, w1, b1, bn_gain, bn_bias, w2, b2 = tuple(model.params)
        y_b = torch.tensor([tgt[0] for tgt in target], dtype=torch.int64)

        # manual cross entropy
        logit_maxes = logits.max(1, keepdim=True).values
        norm_logits = logits - logit_maxes
        counts = norm_logits.exp()
        counts_sum = counts.sum(1, keepdim=True)
        counts_sum_inv = counts_sum ** -1

        # manual batchnorm
        bn_mean_i = 1.0 / batch_size * h_pre_bn.sum(0, keepdim=True)
        bn_diff = h_pre_bn - bn_mean_i
        bn_diff_2 = bn_diff ** 2
        bn_var = 1 / (batch_size - 1) * bn_diff_2.sum(0, keepdim=True)
        bn_var_inv = (bn_var + 1e-5) ** -0.5
        bn_raw = bn_diff * bn_var_inv

        # manual embedding
        emb_cat = emb.view(emb.shape[0], -1)

        ## manual backward
        ## cross entropy
        d_log_probs = torch.zeros_like(probs) # -logprobs[range(batch_size), y_b].mean()
        d_log_probs[range(batch_size), y_b] = -1.0 / batch_size
        d_probs = (1.0 / probs) * d_log_probs # probs.log()
        d_counts_sum_inv = (counts * d_probs).sum(1, keepdim=True) # counts * counts_sum_inv
        d_counts = counts_sum_inv * d_probs # counts * counts_sum_inv
        d_counts_sum = (-counts_sum ** -2) * d_counts_sum_inv # counts_sum ** -1
        d_counts += torch.ones_like(counts_sum) * d_counts_sum # counts.sum(1, keepdim=True)
        d_norm_logits = norm_logits.exp() * d_counts # norm_logits.exp()
        d_logit_maxes = (-d_norm_logits).sum(1, keepdim=True) # logits - logit_maxes
        ## linear layer 2
        d_logits = 1.0 * d_norm_logits # logits - logit_maxes
        # logits.max(1, keepdim=True).values
        d_logits += torch.nn.functional.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * d_logit_maxes
        d_w2 = h.T @ d_logits # h @ w2 + b2
        d_b2 = (1.0 * d_logits).sum(0) # h @ w2 + b2
        ## non-linear layer 1
        d_h = d_logits @ w2.T # h @ w2 + b2
        ## batchnorm layer 1
        d_h_pre_act = (1.0 - h ** 2) * d_h # torch.tanh(h_pre_act)
        d_bn_gain = (bn_raw * d_h_pre_act).sum(0) # bngain * bnraw + bnbias
        d_bn_bias = d_h_pre_act.sum(0) # bngain * bnraw + bnbias
        d_bn_raw = bn_gain * d_h_pre_act # bngain * bnraw + bnbias
        d_bn_var_inv = (bn_diff * d_bn_raw).sum(0) # bn_diff * bn_var_inv
        d_bn_var = (-0.5 * (bn_var + 1e-5) ** -1.5) * d_bn_var_inv # (bn_var + 1e-5) ** -0.5
        # 1 / (batch_size - 1) * bn_diff_2.sum(0, keepdim=True)
        d_bn_diff_2 = (1 / (batch_size - 1) * torch.ones_like(bn_diff_2)) * d_bn_var
        d_bn_diff = bn_var_inv * d_bn_raw # bn_diff * bn_var_inv
        d_bn_diff += (2 * bn_diff) * d_bn_diff_2 # bn_diff ** 2
        d_bn_mean_i = (-1.0 * d_bn_diff).sum(0) # h_pre_bn - bn_mean_i
        ## linear layer 1
        d_h_pre_bn = 1.0 * d_bn_diff # h_pre_bn - bn_mean_i
        # 1 / batch_size * h_pre_bn.sum(0, keepdim=True)
        d_h_pre_bn += (1 / batch_size * torch.ones_like(h_pre_bn)) * d_bn_mean_i
        d_w1 = emb_cat.T @ d_h_pre_bn # emb_cat @ w1 + b1
        d_b1 = (1.0 * d_h_pre_bn).sum(0) # emb_cat @ w1 + b1
        d_emb_cat = d_h_pre_bn @ w1.T # emb_cat @ w1 + b1
        ## embedding layer
        d_emb = d_emb_cat.view(emb.shape) # emb.view(emb.shape[0], -1)
        d_c = torch.zeros_like(c) # c[sample_input]
        for i in range(batch_size):
            for j in range(block_size):
                d_c[sample_input[i,j]] += d_emb[i,j]

        # verify
        ## linear layer 2
        torch.testing.assert_close(d_logits, logits.grad, rtol=1e-15, atol=1e-17)
        torch.testing.assert_close(d_w2, w2.grad, rtol=1e-14, atol=1e-16)
        torch.testing.assert_close(d_b2, b2.grad, rtol=1e-14, atol=1e-16)
        # non-linear layer 1
        torch.testing.assert_close(d_h, h.grad, rtol=1e-15, atol=1e-17)
        ## batchnorm layer 1
        torch.testing.assert_close(d_h_pre_act, h_pre_act.grad, rtol=1e-17, atol=1e-19)
        torch.testing.assert_close(d_bn_gain, bn_gain.grad, rtol=1e-14, atol=1e-16)
        torch.testing.assert_close(d_bn_bias, bn_bias.grad, rtol=1e-15, atol=1e-17)
        ## linear layer 1
        torch.testing.assert_close(d_h_pre_bn, h_pre_bn.grad, rtol=1e-15, atol=1e-17)
        torch.testing.assert_close(d_w1, w1.grad, rtol=1e-14, atol=1e-16)
        torch.testing.assert_close(d_b1, b1.grad, rtol=1e-14, atol=1e-16)
        ## embedding layer
        torch.testing.assert_close(d_emb, emb.grad, rtol=1e-14, atol=1e-16)
        torch.testing.assert_close(d_c, c.grad, rtol=1e-14, atol=1e-16)

if __name__ == '__main__':
    unittest.main()
