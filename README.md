# unsupervised-speech-representation-learning-
This is a intuitive explanation of Representation Learning with Contrastive Predictive Coding  using code provided by jefflai108 that uses CPC to learn representations of sound files for the purpose of speech recognition

Take the utterance:

"I hate that the good are the ones that starve and the selfish are the ones that feast "

What do you think is greater ? the probability of saying the word "hate" while this person is angry, or saying the word "the" while this person is angry ?

Actually, if the person saying the utterance above is infact angry, based on this utterance alone, 

P("the"|angry) > P("hate"|angry).

So a better way to measure the link between "hate" and angry might be

![equation](https://latex.codecogs.com/gif.latex?\frac{P("hate"|angry)}{P("hate")})

This is the "density ratio" in the CPC paper. 

But what about all the other emotions you can have besides angry, like happy, sad and scared. How much does "hate" tell you about being angry as opposed to all these other emotions?

Define P(word, emotion) as the probability of saying that word while having that emotion among all the other combination of words and emotions. ie P("hate"|scared), P("good"|happy) etc. And P(word|emotion) meaning the probability of someone saying that word when you know for a fact that they have that emotion, not considering words spoken during other emotions.

Then the mutual information is:

![equation](https://latex.codecogs.com/gif.latex?MI(word,&space;emotion)&space;=&space;\sum_{words,emotions}P(word,emotion)log(\frac{P(word|emotion)}{P(word)}))

The CPC paper uses this concept to learn meaningful representations of sequences. It models the density ratio as a log-bilinear model

$$\frac{P(x_{t+k}|c_t)}{P(x_{t+k})} \approx e^{z^T_{t+k} W_k c_t)} $$

![equation](https://latex.codecogs.com/gif.latex?\frac{P(x_{t&plus;k}|c_t)}{P(x_{t&plus;k})}&space;\approx&space;e^{z^T_{t&plus;k}&space;W_k&space;c_t)})


![equation](https://latex.codecogs.com/gif.latex?\frac{P(x_{t&plus;k}|c_t)}{P(x_{t&plus;k})}&space;\approx&space;e^{z^T_{t&plus;k}&space;W_k&space;c_t)})

Where z is an encoding of x, much like texts are an encoding of speech. Here z_t+k means the z's in the future relative to a timepoint t. c_t is a summary of series of z's from z_0 to z_t, and W is a matrix transformation to make z and c the same size so that they can be compared. 

# The Encoder-AutoRegressive Model

The raw waveform is fed to the encoder without preprocessing. The size 20480 audio segment represents a 1.28 second long speech, or audio signal. The 160 downsampling factor for the encoder takes a 20480 sequence and compresses it to sequence length 128. 

When batch_first = True, the GRU input and output shapes are (batch, seq, feature), and this is the shape of z which the GRU takes as input, only that features is equivalent to the channel dimension in 1D CNNs, so z is shpae (batch_size,seq_len,channels). 

The Noise Contrastive Estimation is acheived by using the other samples in the batch as the negative samples. 

What we learn from this implementation is that the expression 

W_k c_t

in the log-bilinear model, is meant to be an approximation of 

z_{t+k}

And so the higher the dot product is between the two, the higher is the density ratio estimate f_k, where

![equation](https://latex.codecogs.com/gif.latex?f_k(x_{t&plus;k},&space;c_t)&space;=&space;exp(z^T_{t&plus;k}&space;W_k&space;c_t))

If you look at the implementation below, you will not see the exponent term, because this exponent term is incorporated into the log softmax term. 

In each training batch, batch_size samples of audio signal is used of size 20480x1. The encoder turns this sequence into shape 128x512 (z). each size 512 vector in z is 0.01 seconds, which is why 128 of them amounts to 1.28 seconds. 

`time_C` is a number chosen between 0 and 128 - K, so if K = 12, then `time_C` is between 0 amd 116. Suppose `time_C` = 100, then the vector c_t of size 256 is an embedding of the first 100 vectors of z or the first 1.0 seconds of the audio signal. 

c_t is used to predict the next K of z_t's in the future. ie z_t+k. If c_t actually holds enough information to predict z_t+k, then some linear transformation W_k should be able to approximate z_t+k from c_t. 

So the entire 128 z's are encoded into c's by the same encoder, then a random timstep is chosen from the first 128 - K timsteps in order to predict the next K timesteps, which is why the random timestep `time_C` cannot be higher than 128 - K, or else there will not be a total of K timesteps in the future to compare to. 

# The InfoNCE loss function

If the encoder and the GRU have generated a good representation c_t, then the dot product of W_k c_t with z_t+k should be higher than tha dot product of W_k c_t with zneg_t+k, where zneg_t+k is the z vector by from a different sample in the batch. This is the "negative sample" in "negative sampling". 

In the implementation below the line `zWc = torch.mm(z_t_k[k], torch.transpose(W_c_k[k],0,1))` is performed for each future timestep k in K. 

The z_t_k tensor is the K number of z_t+k's for each sample in the batch.
W_c_k is the K number of vectors that are a result of matrix multiplying Wk with ct that is supposed to approximate z_t_k. The transpose operation is performed on W_c_k in order for the zWc calculation to be a calculation of shape (batch_size, z_size)x(z_size, batch_size) => (batch_size, batch_size). So W_c_k is not just dot producted with the z_t_k it was meant to approximate, it is also dot producted with the z_t_k's fromt he other samples in the batch. The value in position (k,k) of zWc is proportional to how well W_c_k approximates z_t_k, the values in position (k, not k) is proportional to how well W_c_k approximates not the z_t_k from it's own sequence, but the z_t_k from some other sequece, which it should not be good at it it is working properly. 

Suppose batch size is 3 and the first row of zWc are the elements `[0.1,  1,  -0.1]` then the log_softmax is `[-1.4536, -0.5536, -1.6536]`. This means that W_c_k has a weak preference for it's own z_t_k, over the last, 3rd  z_t_k, which is good, but it has a much stronger preference for the 2nd z_t_k than it's own, which is bad. The softmax function forces the entire row to be less and 1 and as a whole sum to 1. If you want the first element in the softmax to be close to 1.0, which you want, since this means that the negative loss will be close to 0, (the negative loss in this case will be +1.4536) then you want not only for the first element to be large, that is not enough, the other elements need to be low. This is why the loss function log_softmax(z_t_k dot W_c_k / sum of z_t_k dot W_c_k for all samples) not only helps to tune your neural network to move in the right direction, but move away from the wrong directions.

Here I show you which line in the code corresponds to the loss function described int he paper:

![equation](https://latex.codecogs.com/gif.latex?InfoNCELoss(z_{t&space;=&space;t'&space;\rightarrow&space;t'&plus;k},&space;c_t)&space;=&space;-&space;E&space;\left[&space;\log(&space;\frac{f_k(z_{t&plus;k},&space;c_t)}{\sum_{J}&space;f_k(z_{j},&space;c_t)}&space;)&space;\right]\\)

Where i is the index of the correct element in the softmax, the code version of this is:

`nce += torch.sum(torch.diag(logsof_zWc))`

`nce /= -1.*batch_size*self.K` 

The fact that the nce is accumulated over each sample of the batch and each future timestep k, then at the end divided by `batch_size*self.K` represents the expectation. 

The `torch.diag(logsof_zWc)` makes a vector from the diagonal of the `logsof_zWc` matrix, so it is taking only the log softmax values of the correct z_t_k dot W_c_k pairs (this value through the softmax has already incorporated information from the neighboring negative samples). Just like the softmax term is only considering the ratio of the element over the elements including itself:

$$\frac{f_k(z_{t+k}, c_t)}{\sum_{J} f_k(z_{j}, c_t)}$$ 

Here I use z instead of x for simplicity and making the code look like the formula, the paper uses x in the density ratio formula f_k. But z is a direct mapping from x, so hopefully you can see that it is the same. 

# The softmax probability, the density ratio and mutual information
The paper asks you to noitice that the loss function is of the same form as the cross entropy loss 

$$-ylog(P(y|x))$$

where

$$ P(y|x) ~ \frac{f_k(z_{t+k}, c_t)}{\sum_{J} f_k(z_{j}, c_t)}$$  

Returning to the analogy of words and emotions, the loss function is optimized when the model's internal representation of what it means to be angry maximizes P("hate"|angry)/P("hate").

What is the probability that is the angry kind of hate and not one of the other hates like from someone scared or happy? its not P("hate"|angry)/P("hate") cause thats a ratio, not a probability. probabilities need to sum to 1 when you add them together with all the other probabilities, "good", "hate", "the".

The paper presents this as 

$$ P(d=i|X,c_t) = \frac{ \frac{p(word_i|emotion_t)}{p(word_i)} }{ \sum^{N}_{j=1} \frac{p(word_j|emotion_t)}{p(word_j)} } $$

and uses this as a justification for why minimizing the loss function in turns maximizes the density ration, which in turn maximizes the mutual information. The relationship to mutual information is proven in the Appendix. 
