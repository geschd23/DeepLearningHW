import numpy as np
import codecs
import tensorflow as tf

def load_data(inference_input_file, hparams=None): # from hackathon_9
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]

  return inference_data

def parse_data(sentences, word_map, length):
    data_array = np.empty(shape=[len(sentences),length], dtype=int)
    i=0
    for sentence in sentences:
        words=sentence.split()
        j=0
        for word in words:
            word = word.lower()
            if word in word_map:
                data_array[i][j] = word_map[word]
                j+=1
            else:
                j=0
            if j==length:
                i+=1
                break
    return data_array[:i] 

def setup_data(file, word_map, length):
    src_data = load_data(file)
    npArray = parse_data(src_data, word_map, length)
    np.save("textData",npArray)
    

def split_data(data, proportion): #function taken from hackathon_3 notebook and improved by Jingchao Zhang
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1

    Returns: 
        - train: numpy array; used for training
        - val: numpy array; used for validation
    """
    size = data.shape[0]
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    train = data[s[:split_idx]]
    val = data[s[split_idx:]]
    return train, val

def k_fold_split(data, k, i): #function improved upon Paul Quint's draft
    """
    Partitions a numpy array into 'training set' and 'validation set' for k-fold validation

    Args:
        - data: numpy array; to be aplit along the first axis
        - k: positive integer; the data will be evenly split into k groups. 
        - i: integer; 0 <= i < k; the i-th group is used as the validation set, and the rest is used as 
        the training set. 

    Returns: 
        - train: numpy array; used for training
        - val: numpy array; used for validation
    """
    t = data.shape[0]//k
    val = data[i*t:(i+1)*t]
    first = data[:i*t]
    second = data[(i+1)*t:]
    train = np.concatenate((first, second), axis = 0)
    return train, val

def print_file(string, file):
    """
    Prints to the screen and a file

    Args:
        - string: string to print
        - file: file to print to
    """
    print(string.encode('utf-8'))
    print(string.encode('utf-8'), file=file)
    file.flush()
    
def load_glove(file): 
    """
    Loads a glove file specifying a pretrained embedding
    modified from code at https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    
    Args:
        - file: the filename of the glove file
        
    Returns:
        - embedding: the word embedding vectors unit normalized
        - word_map: dict from word to index
        - indexed_words: dict from index to word
    """
    f = open(file,'r', encoding="utf-8")
    words = sum(1 for line in open(file,'r', encoding="utf-8"))
    dimensions = len(open(file,'r', encoding="utf-8").readline().split())-1
    
    word_map = {}
    indexed_words = {}
    embedding = np.empty(shape=[words,dimensions])
    
    #grab the data
    index = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        word_map[word] = index
        indexed_words[index] = word
        embedding[index] = [float(val) for val in splitLine[1:]]
        index += 1
        
    #normalize the vectors
    norms = np.linalg.norm(embedding,axis=1)
    embedding = embedding / norms[:,None]
    
    return embedding,word_map,indexed_words

def normalize_vector(vector):
    """
    Normalizes the input vector
    """
    return vector / np.linalg.norm(vector)

def get_word_vector(embedding, word_map, word):
    """
    Retrieves the word's glove vector from the embedding
    """
    return embedding[word_map[word]]

def get_closest_word(embedding, indexed_words, vector):
    """
    Finds the closest match to the given vector in the embedding using cosine similarity
    """
    temp = normalize_vector(vector)
    word = indexed_words[np.argmax(np.dot(embedding,temp))]
    return word

def get_sentence(embedding, indexed_words, sentence_vector):
    """
    Finds the closest match for each word in the sentence using cosine similarity
    """
    sentence = ""
    for word in sentence_vector:
        sentence += get_closest_word(embedding,indexed_words,word)+" "
    return sentence