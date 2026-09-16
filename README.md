### NAME: SURYA P <br>
### REG NO: 212224230280 <br> 
### Date: 18/07/2026

## EX. No. 1 : IMPLEMENTATION OF BAYESIAN NETWORKS

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

```python
import pandas as pd # for data manipulation
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
#Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Drop records where target RainTomorrow=NaN
df=df[pd.isnull(df['RainTomorrow'])==False]
# Drop the 'Date' column as it is not relevant for the model
df = df.drop(columns='Date')

# For other columns with missing values, fill them in with column mean for numeric columns only
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Create bands for variables that we want to use in the model
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40'   if x<=40 else
                                                            '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')

# Show a snaphsot of data
print(df)

# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
'''
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Caclucate probabilities
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                # Caclucate probabilities
                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob
'''
def probs(data, child, parent1=None, parent2=None):
    if parent1 is None:
        # P(child)
        return (
            pd.crosstab(data[child], columns="count", normalize="columns")
            .sort_index()
            .to_numpy()
            .ravel()
            .tolist()
        )

    if parent2 is None:
        # P(child | parent1)
        table = pd.crosstab(data[parent1], data[child], normalize="index")
    else:
        # P(child | parent1, parent2)
        table = pd.crosstab([data[parent1], data[parent2]], data[child], normalize="index")

    return table.sort_index().to_numpy().ravel().tolist()
# Create nodes by using our earlier function to automatically calculate probabilities
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

# Create Network
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)
# Set node positions
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

# Set options for graph looks
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,}

# Generate graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

# Update margins and print the graph
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()

print("CPTs: Humidity 9AM ->{}".format(probs(df, child='Humidity9amCat')))
print("CPTs: Humidity 3PM ->{}".format(probs(df, child='Humidity3pmCat', parent1='Humidity9amCat')))
print("CPTs: Wind Gust Speed ->{}".format(probs(df, child='WindGustSpeedCat')))
print("CPTs: Rain Tomorrow ->{}".format(probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat')))

rain_cpt = pd.crosstab(
    [df["Humidity3pmCat"], df["WindGustSpeedCat"]],
    df["RainTomorrow"],
    normalize="index"
)

print(rain_cpt.round(4))
```


## Output:

<img width="591" height="680" alt="image" src="https://github.com/user-attachments/assets/76b94407-7e16-4b9d-943c-4f08f2e304fe" />

<img width="560" height="669" alt="image" src="https://github.com/user-attachments/assets/0088feda-0600-46ac-a257-39a2c09526a6" />

<img width="660" height="499" alt="image" src="https://github.com/user-attachments/assets/0f266eeb-57e7-4b2a-ac29-45b17e521f3c" />

```
CPTs: Humidity 9AM ->[0.25951567933335423, 0.7404843206666458]
CPTs: Humidity 3PM ->[0.9151376146788991, 0.08486238532110092, 0.5577907517874519, 0.44220924821254814]
CPTs: Wind Gust Speed ->[0.5980232448858118, 0.22305065630775978, 0.17892609880642837]
CPTs: Rain Tomorrow ->[0.9193523214982442, 0.08064767850175575, 0.8901445180919265, 0.10985548190807345, 0.7842639593908629, 0.21573604060913706, 0.6335166148697108, 0.3664833851302893, 0.5402025014889815, 0.4597974985110185, 0.3951322751322751, 0.6048677248677249]
```

<img width="742" height="588" alt="image" src="https://github.com/user-attachments/assets/fceec656-f470-4eed-81b2-1f7e2bfa5f64" />

<img width="735" height="642" alt="image" src="https://github.com/user-attachments/assets/23c95572-48a9-44ef-9888-15bc820bf02e" />

<img width="709" height="153" alt="image" src="https://github.com/user-attachments/assets/0f221092-65ef-462a-a56a-fb525b3d4d50" />

<img width="567" height="155" alt="image" src="https://github.com/user-attachments/assets/bae722e1-5e18-4cb2-91f7-21615ed95255" />



## Result:
   Thus a Bayesian Network is generated using Python
