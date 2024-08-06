r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
student_name_1 = 'Roee Lapushin' # string
student_ID_1 = '318875366' # string
student_name_2 = 'Yair Nadler' # string
student_ID_2 = '316387927' # string

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 100
    hypers['seq_len'] = 64
    hypers['h_dim'] = 512
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.1
    hypers['learn_rate'] = 1e-2
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "I must attend to"
    temperature = 0.6
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences to manage VRAM usage more efficiently. Training on the entire text would provide 
excessive context, leading the model to find connections between distant sentences instead of focusing on closer, 
more relevant relationships. This approach also helps avoid the need for a deeper model, which could increase the risk 
of issues like exploding and vanishing gradients.
"""

part1_q2 = r"""
The model maintains a hidden state that carries information from previous batches, allowing it to remember and 
incorporate context from earlier sentences, thus enabling longer memory.
"""

part1_q3 = r"""
We keep the order of batches intact to preserve the logical sequence of the text. This helps the model learn 
higher-level contextual relationships, ensuring it generates meaningful sentences. Maintaining the order also preserves 
the hidden state, which is crucial for retaining context across batches.
"""

part1_q4 = r"""
1. Lowering the temperature during generation reduces the risk of producing nonsensical text. It favors more probable 
    words, leading to more coherent results, even if they are less diverse.
2. With a very high temperature, the output becomes nearly uniform, meaning the probability distribution flattens. 
    This randomness can make the generated text less coherent, as the model is essentially picking words at random.
3. At a very low temperature, the model becomes highly confident in its word choices, often selecting the most probable word. 
    While this increases coherence, it can reduce diversity and lead to repetitive or looping text if the temperature is too low.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
We can't answer the question as it relates to the cancelled part of the assignment (Notebook 3).
"""

part4_q2 = r"""
If you freeze the last layers of the model and fine-tune internal layers, the 
model might still be able to fine-tune to the task, but the results could be 
worse compared to freezing internal layers and fine-tuning the last layers.
This is because the last layers of the model are typically responsible for
learning task-specific features and the internal layers are primarily responsible
for learning general representations from the input data.
"""


# ==============
