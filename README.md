<H3> Name </H3> :Hemalatha.A
<H3>Register No.</H3> 21224240056
<H3> Experiment 1</H3> 
<H3>DATE:</H3>22.07.2026
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>
## Aim :
    To create a bayesian Network for the given dataset in Python
## Algorithm:
Step 1:Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
Step 2:Set pandas options to display more columns<br/>
Step 3:Read in weather data from a CSV file using pandas<br/>
Step 4:Remove records where the target variable RainTomorrow has missing values<br/>
Step 5:Fill in missing values in other columns with the column mean<br/>
Step 6:Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
Step 7:Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
Step 8:Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
Step 9:Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
Step 10:Convert the BBN to a join tree using the InferenceController<br/>
Step 11:Set node positions for the graph<br/>
Step 12:Set options for the graph appearance<br/>
Step 13:Generate the graph using networkx<br/>
Step 14:Update margins and display the graph using matplotlib.pyplot<br/>

## Program:
```
`!pip install pybbn
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.dag import Edge, EdgeType
from pybbn.graph. jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
pd.options.display.max_columns=50

df=pd.read_csv('/weatherAUS.csv',encoding='utf-8')
df=df[pd.isnull(df['RainTomorrow']) == False]
df = df.drop(columns='Date')

numeric_columns = df.select_dtypes(include=['number']).columns
df.loc[:, numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df['WindGustSpeedCat' ]=df['WindGustSpeed' ].apply(lambda x: '0. <= 40' if x <= 40 else '1. > 40')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0 .<= 60')
df['Humidity3pmCat' ]=df['Humidity3pm' ].apply(lambda x: '1.>60' if x>60 else '0 .<= 60')

print(df)

def probs(data, child, parent1=None, parent2=None):
    if parent1 == None:
        # Calculate marginal probabilities and return as a flattened numpy array
        prob_series = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').iloc[:, 0].sort_index()
        return prob_series.values.astype(float)
    else:
        # Calculate conditional probabilities
        if parent2 == None:
            cpt = pd.crosstab(index=data[parent1], columns=data[child], normalize='index')
        else:
            cpt = pd.crosstab(index=[data[parent1], data[parent2]], columns=data[child], normalize='index')

        # Sort indices and columns to ensure consistent ordering matching BbnNode Variable states
        cpt = cpt.sort_index(axis=0).sort_index(axis=1)

        # Flatten the CPT into a numpy array of probabilities (parent states iterate slowest, child states fastest)
        return cpt.stack().values.astype(float)

H9am = BbnNode(Variable(0, 'H9am', ['0 .<= 60', '1.>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['0 .<= 60', '1.>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['0. <= 40', '1. > 40']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

bbn = Bbn() \
.add_node(H9am) \
.add_node(H3pm) \
.add_node(W) \
.add_node (RT) \
.add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
.add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
.add_edge(Edge(W, RT, EdgeType.DIRECTED))
join_tree = InferenceController.apply(bbn)
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

options = {
"font_size": 16,
"node_size": 4000,
"node_color": "pink",
"edgecolors": "blue",
"edge_color": "green",
"linewidths": 5,
"width": 5,}

n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, ** options)
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt. show( )
print(probs(df, child='Humidity9amCat'))
print(probs(df, child='Humidity3pmCat', parent1='Humidity9amCat' ))
print(probs(df, child='WindGustSpeedCat'))
print(probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))``

```
## Output:

<img width="801" height="142" alt="541433980-ca2ea19b-54d5-4b18-a716-7f9dcd40c894" src="https://github.com/user-attachments/assets/09b36939-6dc0-4e93-a3a5-e2024237aa5e" />

<img width="552" height="842" alt="541433539-7a02b182-1a3c-4691-947c-08e993c18585" src="https://github.com/user-attachments/assets/5c6132da-0d46-44cc-9db7-8727a2657a1f" />



## Result:
   Thus a Bayesian Network is generated using Python

