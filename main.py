from sentence_splitter import SentenceSplitter
import nltk
import copy
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import math
import string

nltk.download('punkt')

nltk.download('stopwords')

words_dict = {}
vectors = []
list_of_original_sentences = []
class_counts = []
document = ""

contractions_dict = { "ain't": "are not", "'s":" is", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "‘cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "that'd": "that would", "that'd've": "that would have", "there'd": "there would", "there'd've": "there would have", "they'd": "they would", "they'd've": "they would have","they'll": "they will",
 "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not","what'll": "what will", "what'll've": "what will have", "what're": "what are", "what've": "what have", "when've": "when have", "where'd": "where did", "where've": "where have",
 "who'll": "who will", "who'll've": "who will have", "who've": "who have", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


class Graph:

    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    def deleteEdge(self, v, w):
        self.adj[v].remove(w)
        self.adj[w].remove(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)

def run_process(filename):
    global document
    #create_document(filename)
    split_into_sentences(document)

def create_document(filename):
    global document
    with open(filename, 'r') as file:
        document = file.read()
        document.strip()


def create_document_from_csv():
    global list_of_original_sentences, class_counts
    files = ["sentiment_neutral.csv", "sentiment_positive.csv", "sentiment_negative.csv"]
    for i in range(len(files)):
        df = pd.read_csv(files[i], usecols=["text"])
        count = 0
        for index, row in df.iterrows():
            tweet = row["text"]
            tweet = re.sub(r'[^\w\s]', ' ', tweet)
            tweet = re.sub('\n', '', tweet)
            tweet = ' '.join(tweet.split())
            tweet  = str(i)+ " "+tweet

            #print(tweet)
            list_of_original_sentences.append(tweet)
            count += 1
            if count == 30:
               break

        class_counts.append(count)

    print(class_counts)



def remove_punct(token):
    return [word for word in token if word.isalpha()]

def split_into_sentences(document):


    global list_of_original_sentences

    #print(list_of_original_sentences)
    list_of_sentences = copy.deepcopy(list_of_original_sentences)

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    for i in range(len(list_of_sentences)):
        list_of_sentences[i] = remove_punct(word_tokenize(expand_contractions(list_of_sentences[i].lower())))
        filtered_sentence = []
        for w in list_of_sentences[i]:
            if w not in stop_words:
                filtered_sentence.append(w)
        list_of_sentences[i] = [ps.stem(words_sent) for words_sent in filtered_sentence]

    #print(list_of_sentences)
    create_dictionary(list_of_sentences)
    map_sentence_to_vectors(list_of_sentences)
    create_graph(list_of_sentences)

def create_dictionary(list_of_sentences):
    for sentence in list_of_sentences:
        for word in sentence:
            words_dict[word] = ""

    #print(words_dict)


def map_sentence_to_vectors(list_of_sentences):
    dim = len(words_dict)
    temp = list(words_dict.items())
    for i in range(len(list_of_sentences)):
        sentence = list_of_sentences[i]
        vector = np.zeros((dim,), dtype=int)
        for word in sentence:
            index = [idx for idx, key in enumerate(temp) if key[0] == word]
            vector[index] += 1
        vectors.append(vector)
        #print(list_of_original_sentences[i])
        #print(vector)


    print("no of vectors: "+str(len(vectors)))


def create_graph(list_of_sentences):
    num_of_sentences = len(list_of_sentences)
    graph = np.zeros((num_of_sentences, num_of_sentences))
    #print(graph)
    for i in range(num_of_sentences):
        for j in range(num_of_sentences):
            if (j != i):
                #print(vectors[i])
                #print(vectors[j])
                euclidean_dist = np.linalg.norm(vectors[i] - vectors[j])
                if euclidean_dist == 0:
                    graph[i][j] = float('inf')
                else:
                    graph[i][j] = 1.0/euclidean_dist

                print(str(i) + "---"+str(j))
    #print(graph)
    mst_construction_prim(graph)

def mst_construction_prim(graph):
    num_of_vertices = len(vectors)
    INF = float('inf')
    selected = np.zeros((num_of_vertices,), dtype=int)
    num_of_edges = 0
    selected[0] = True
    mst = []

    while (num_of_edges < num_of_vertices - 1):
        print("no of edges: "+str(num_of_edges)+", no. of vertices: "+str(num_of_vertices))
        minimum = INF
        x = 0
        y = 0
        for i in range(num_of_vertices):
            if selected[i]:
                for j in range(num_of_vertices):
                    if ((not selected[j]) and graph[i][j]):
                        # not in selected and there is an edge
                        if minimum > graph[i][j]:
                            minimum = graph[i][j]
                            x = i
                            y = j
        mst.append((graph[x][y], (x,y)))
        print("mst edge:"+ str(x) + " "+str(y)+" "+str(graph[x][y]))
        selected[y] = True
        num_of_edges += 1

    mst.sort()
    #print("ascending order of weights sorted mst, i.e. descending order of Euclidean distances:")
    #print(mst)
    cluster_and_result(mst)


def cluster_and_result(mst):
    print("---------------------------------------------------------------------------------")
    k = 3  #no of clusters
    num_of_vertices = len(vectors)

    if (k > num_of_vertices):
        print("Invalid: Respecify the no. of clusters (<="+str(num_of_vertices)+")")
        exit()
    graph = Graph(num_of_vertices)

    for i in range(len(mst)):
        u = mst[i][1][0]
        v = mst[i][1][1]
        graph.addEdge(u, v)

    clusters = graph.connectedComponents()
    #at this point, no of conn comp = 1

    least_index_mst = 0
    #mst has already been sorted
    while (len(clusters) < k and least_index_mst < len(mst)):
        u = mst[least_index_mst][1][0]
        v = mst[least_index_mst][1][1]
        graph.deleteEdge(u,v)
        least_index_mst += 1
        clusters = graph.connectedComponents()

    print(clusters)
    average_entropy = 0
    for i in range(len(clusters)):
        class_counts_in_cluster = np.zeros((len(class_counts),), dtype=int)
        cluster_entropy = 0
        #print("Cluster "+str(i+1))
        for indx in clusters[i]:
            print(list_of_original_sentences[indx])
            class_counts_in_cluster[int(list_of_original_sentences[indx][0])] += 1
        for j in range(len(class_counts)):
            fract = class_counts_in_cluster[j] / float(class_counts[j])
            if fract != 0:
                cluster_entropy -= fract*math.log10(fract)

        print("cluster entropy:" + str(cluster_entropy))
        print("--------------------------------------------------------------------------")

        average_entropy += cluster_entropy

    average_entropy /= float(len(class_counts))
    print(average_entropy)


        #print("--------------------------------------------------------------------------")

    #print(list_of_original_sentences[0])


if __name__ == "__main__":
    create_document_from_csv()
    run_process("text2.txt")
