# Introduction

This repository contains the source code associated with my Udacity Capstone Project (based on the proposal found 
at https://review.udacity.com/#!/reviews/541685). 

The code is designed as an extension the work done in the "Snapshot Ensembles: Train 1, get M for Free" research paper 
(https://arxiv.org/abs/1704.00109).  

The main routine for code can be found in the main repo.  Using the "set_up.sh" script (before running "main.py"), it 
pulls the other two repositories (with the state-of-the-art CNN's and Deep Learning utilities) needed to run the project.
 
At a high level, this project aims to improve the original Snapshot ensemble by adding more variety to the local 
minima found in the underlying Snapshot models.  

After the code is run and models are trained, there is a template iPython notebook (called 
"snapshot_graph_generator.ipynb") for producing relavent graphs to demonstrate evaluation metric related to the 
experiments. 


With this code (and its accompanying repositories), a user should be able to:
	1. Train a Snapshot ensemble
	2. Train a triplet loss variant of a Snapshot model
	3. Evaluate Snapshot models and ensembles using a variety of metrics
	4. Build data-rich graphs outlining metrics from evaluations

    

    
# Details of Triplet Loss Training

Currently, the code for triplet loss training assumes that the system has at least two GPUs.  

It however can definitely be modified to use only 1 GPU by simply commenting out the lines containing 
'os.environ["CUDA_VISIBLE_DEVICES"]' in the code (particularly in the "trip_utils.py" file).  In fact, though 
not yet tested, the code should likely work with 1 GPU even without any modification to the code.

In the current implementation, one GPU is used to train the triplet loss Snapshot model (for forward and back 
propagation).  Concurrently, the other GPU is used as a resource for the triplet pair multi-process generator.  

Because producing triplet loss pairs requires first calculating the distance between the model's final layer 
embeddings for various input images, it is most efficient to use a GPU for the inference needed to produce these 
embedding layers.
 
Optimally, a generator will leverage several threads/processes calculate embeddings and use them for triplet pair 
generation since it ultimately ends up being much faster than doing it in a single thread.  And also, the 
DenseNet-100 model (used in training) is small enough that inference for each thread/process only requires a small 
portion of a GPU's available memory. 

For the underlying threads/processes to calculate embeddings with updated model weights, the main training 
thread communicates/shares the model weights via a custom Callback object (called "GetWeights").

Every 25 batches, the GetWeights object uses the "on_batch_end" callback function to update a multi-process list 
with the current weights of the model.  Using this multi-process list, the worker processes (generating the 
triplet pairs) can then update their local models to reflect the one in training.  That way, the embedding 
distances needed for selecting triplet pairs are current. 


Also noteworthy, the generator for triplet pairs is not a normal iterator object.  

Since TensorFlow in Python struggles with handling fork operations (via Python's multiprocessing.Process "start" 
function call), it is necessary to fork and create worker processes before the main thread creates any kind of 
TensorFlow session.  

Therefore, the "main.py" file contains a constructor function (for the "multi_thread_trip_gen" object) 
to produce worker processes each with their own TensorFlow session before the main process generates its personal 
TensorFlow session. The function takes a queue size (for the triplet pairs) and a number of desired worker 
processes.
 
After the worker processes are created, but at some point before triplet training, the workers need the details 
for triplet training (including the input data/labels, batch size, desired margin, etc).  This is done using the 
"start_activity" function for the "multi_thread_trip_gen" object.

Likewise, when the training is done, the "multi_thread_trip_gen" object's "stop_activity" function is called to 
empty the shared triplet pair queue and halt the production of more triplet pairs.


Finally, the triplet generator has three levels of "hardness" in its preparation of pairs - easy, medium, and 
hard - signified with 0, 1, and 2, respectively.  

Without going into too much low level detail, it is advisable to follow a curriculum approach in gradually 
increasing the difficult of the triplet pairs provided to the model.  For instance, one strategy would be start 
with 10 percent of the overall training time dedicated to learning "easy" triplets, then the next 15 percent of 
the training time on "medium" difficulty triplets, and the remainder of training on the "hard" triplets.

Because triplet difficulty is largely predicated on the selection of a good margin value, the generator also 
has a "dynamic_margin" flag.  During "easy" and "medium" training, the generator (with the "dynamic_margin" flag 
turned on) will alter the margin value if it is found to be either too easy or too hard.




# Python Package Requirements

- keras
- tensorflow
- cPickle
- shutil
- h5Py
- numpy
- jupyter
- plotly
- multiprocessing