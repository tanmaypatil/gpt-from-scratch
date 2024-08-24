from self_attention import *
d_in = 3
d_out = 2
inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
batch = torch.stack((inputs, inputs), dim=0)

def test_attn():
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    z = (sa_v1(inputs))
    assert z.shape == (6,2)
    print(z)
    
def test_attention2():
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    z = (sa_v2(inputs))
    assert z.shape == (6,2)
    print(z)
    
def test_casual_Attn():
    torch.manual_seed(123)
    context_length = batch.shape[1]
    assert context_length == 6
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    z = (ca(batch))
    print(z)
    
def test_multihead():
    context_length = batch.shape[1] # This is the number of tokens
    torch.manual_seed(123)
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    

