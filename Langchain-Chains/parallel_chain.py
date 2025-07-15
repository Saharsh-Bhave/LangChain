from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

#using two models for making notes and making a quiz parallely.
model1 = ChatOpenAI(model='o3-mini-2025-01-31') 
model2 = ChatOpenAI(model='gpt-4.1-nano-2025-04-14')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables = ['text']
)
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes ->{notes} and quiz->{quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Transformers were first developed to solve the problem of sequence transduction, or neural machine translation, which means they are meant to solve any task that transforms an input sequence to an output sequence. This is why they are called “Transformers”.

But let’s start from the beginning.

What Are Transformer Models?
A transformer model is a neural network that learns the context of sequential data and generates new data out of it.

To put it simply:

A transformer is a type of artificial intelligence model that learns to understand and generate human-like text by analyzing patterns in large amounts of text data.

Transformers are a current state-of-the-art NLP model and are considered the evolution of the encoder-decoder architecture. However, while the encoder-decoder architecture relies mainly on Recurrent Neural Networks (RNNs) to extract sequential information, Transformers completely lack this recurrency.

So, how do they do it?

They are specifically designed to comprehend context and meaning by analyzing the relationship between different elements, and they rely almost entirely on a mathematical technique called attention to do so.
The shift from RNN models like LSTM to Transformers for NLP problems
At the time of the Transformer model introduction, RNNs were the preferred approach for dealing with sequential data, which is characterized by a specific order in its input.

RNNs function similarly to a feed-forward neural network but process the input sequentially, one element at a time.

Transformers were inspired by the encoder-decoder architecture found in RNNs. However, Instead of using recurrence, the Transformer model is completely based on the Attention mechanism.

Besides improving RNN performance, Transformers have provided a new architecture to solve many other tasks, such as text summarization, image captioning, and speech recognition.

So, what are RNNs' main problems? They are quite ineffective for NLP tasks for two main reasons:

They process the input data sequentially, one after the other. Such a recurrent process does not make use of modern graphics processing units (GPUs), which were designed for parallel computation and, thus, makes the training of such models quite slow.
They become quite ineffective when elements are distant from one another. This is due to the fact that information is passed at each step and the longer the chain is, the more probable the information is lost along the chain.
The shift from Recurrent Neural Networks (RNNs) like LSTM to Transformers in NLP is driven by these two main problems and Transformers' ability to assess both of them by taking advantage of the Attention mechanism improvements:

Pay attention to specific words, no matter how distant they are.
Boost the performance speed.
Thus, Transformers became a natural improvement of RNNs.

Next, let’s take a look at how transformers work.

The Transformer Architecture
Overview
Originally devised for sequence transduction or neural machine translation, transformers excel in converting input sequences into output sequences. It is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. The main core characteristic of the Transformers architecture is that they maintain the encoder-decoder model.

If we start considering a Transformer for language translation as a simple black box, it would take a sentence in one language, English for instance, as an input and output its translation in English.

If we dive a little bit, we observe that this black box is composed of two main parts:

The encoder takes in our input and outputs a matrix representation of that input. For instance, the English sentence “How are you?”
The decoder takes in that encoded representation and iteratively generates an output. In our example, the translated sentence “¿Cómo estás?”
However, both the encoder and the decoder are actually a stack with multiple layers (same number for each). All encoders present the same structure, and the input gets into each of them and is passed to the next one. All decoders present the same structure as well and get the input from the last encoder and the previous decoder.

The original architecture consisted of 6 encoders and 6 decoders, but we can replicate as many layers as we want. So let’s assume N layers of each.

So now that we have a generic idea of the overall Transformer architecture, let’s focus on both Encoders and Decoders to understand better their working flow:

The Encoder WorkFlow
The encoder is a fundamental component of the Transformer architecture. The primary function of the encoder is to transform the input tokens into contextualized representations. Unlike earlier models that processed tokens independently, the Transformer encoder captures the context of each token with respect to the entire sequence.

"""

result = chain.invoke({'text': text})

print(result)