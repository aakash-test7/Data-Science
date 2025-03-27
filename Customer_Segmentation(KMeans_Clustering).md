**STEP 1**: Importing Dependencies


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

**STEP 2**: Data Collection & Analysis


```python
# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('/content/Mall_Customers.csv')
```


```python
# first 5 rows in the dataframe
customer_data.head()
```





  <div id="df-b20764e4-bf0a-46dd-b899-05fe9c9e92d7" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b20764e4-bf0a-46dd-b899-05fe9c9e92d7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b20764e4-bf0a-46dd-b899-05fe9c9e92d7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b20764e4-bf0a-46dd-b899-05fe9c9e92d7');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8319be51-e823-4ccc-9b03-061fd93dc949">
  <button class="colab-df-quickchart" onclick="quickchart('df-8319be51-e823-4ccc-9b03-061fd93dc949')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8319be51-e823-4ccc-9b03-061fd93dc949 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# finding the number of rows and columns
customer_data.shape
```




    (200, 5)




```python
# getting some informations about the dataset
customer_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 5 columns):
     #   Column                  Non-Null Count  Dtype 
    ---  ------                  --------------  ----- 
     0   CustomerID              200 non-null    int64 
     1   Gender                  200 non-null    object
     2   Age                     200 non-null    int64 
     3   Annual Income (k$)      200 non-null    int64 
     4   Spending Score (1-100)  200 non-null    int64 
    dtypes: int64(4), object(1)
    memory usage: 7.9+ KB



```python
# checking for missing values
customer_data.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CustomerID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Annual Income (k$)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Spending Score (1-100)</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



**STEP 3**: Choosing the Annual Income Column & Spending Score column


```python
X = customer_data.iloc[:,[3,4]].values
```


```python
print(X)
```

    [[ 15  39]
     [ 15  81]
     [ 16   6]
     [ 16  77]
     [ 17  40]
     [ 17  76]
     [ 18   6]
     [ 18  94]
     [ 19   3]
     [ 19  72]
     [ 19  14]
     [ 19  99]
     [ 20  15]
     [ 20  77]
     [ 20  13]
     [ 20  79]
     [ 21  35]
     [ 21  66]
     [ 23  29]
     [ 23  98]
     [ 24  35]
     [ 24  73]
     [ 25   5]
     [ 25  73]
     [ 28  14]
     [ 28  82]
     [ 28  32]
     [ 28  61]
     [ 29  31]
     [ 29  87]
     [ 30   4]
     [ 30  73]
     [ 33   4]
     [ 33  92]
     [ 33  14]
     [ 33  81]
     [ 34  17]
     [ 34  73]
     [ 37  26]
     [ 37  75]
     [ 38  35]
     [ 38  92]
     [ 39  36]
     [ 39  61]
     [ 39  28]
     [ 39  65]
     [ 40  55]
     [ 40  47]
     [ 40  42]
     [ 40  42]
     [ 42  52]
     [ 42  60]
     [ 43  54]
     [ 43  60]
     [ 43  45]
     [ 43  41]
     [ 44  50]
     [ 44  46]
     [ 46  51]
     [ 46  46]
     [ 46  56]
     [ 46  55]
     [ 47  52]
     [ 47  59]
     [ 48  51]
     [ 48  59]
     [ 48  50]
     [ 48  48]
     [ 48  59]
     [ 48  47]
     [ 49  55]
     [ 49  42]
     [ 50  49]
     [ 50  56]
     [ 54  47]
     [ 54  54]
     [ 54  53]
     [ 54  48]
     [ 54  52]
     [ 54  42]
     [ 54  51]
     [ 54  55]
     [ 54  41]
     [ 54  44]
     [ 54  57]
     [ 54  46]
     [ 57  58]
     [ 57  55]
     [ 58  60]
     [ 58  46]
     [ 59  55]
     [ 59  41]
     [ 60  49]
     [ 60  40]
     [ 60  42]
     [ 60  52]
     [ 60  47]
     [ 60  50]
     [ 61  42]
     [ 61  49]
     [ 62  41]
     [ 62  48]
     [ 62  59]
     [ 62  55]
     [ 62  56]
     [ 62  42]
     [ 63  50]
     [ 63  46]
     [ 63  43]
     [ 63  48]
     [ 63  52]
     [ 63  54]
     [ 64  42]
     [ 64  46]
     [ 65  48]
     [ 65  50]
     [ 65  43]
     [ 65  59]
     [ 67  43]
     [ 67  57]
     [ 67  56]
     [ 67  40]
     [ 69  58]
     [ 69  91]
     [ 70  29]
     [ 70  77]
     [ 71  35]
     [ 71  95]
     [ 71  11]
     [ 71  75]
     [ 71   9]
     [ 71  75]
     [ 72  34]
     [ 72  71]
     [ 73   5]
     [ 73  88]
     [ 73   7]
     [ 73  73]
     [ 74  10]
     [ 74  72]
     [ 75   5]
     [ 75  93]
     [ 76  40]
     [ 76  87]
     [ 77  12]
     [ 77  97]
     [ 77  36]
     [ 77  74]
     [ 78  22]
     [ 78  90]
     [ 78  17]
     [ 78  88]
     [ 78  20]
     [ 78  76]
     [ 78  16]
     [ 78  89]
     [ 78   1]
     [ 78  78]
     [ 78   1]
     [ 78  73]
     [ 79  35]
     [ 79  83]
     [ 81   5]
     [ 81  93]
     [ 85  26]
     [ 85  75]
     [ 86  20]
     [ 86  95]
     [ 87  27]
     [ 87  63]
     [ 87  13]
     [ 87  75]
     [ 87  10]
     [ 87  92]
     [ 88  13]
     [ 88  86]
     [ 88  15]
     [ 88  69]
     [ 93  14]
     [ 93  90]
     [ 97  32]
     [ 97  86]
     [ 98  15]
     [ 98  88]
     [ 99  39]
     [ 99  97]
     [101  24]
     [101  68]
     [103  17]
     [103  85]
     [103  23]
     [103  69]
     [113   8]
     [113  91]
     [120  16]
     [120  79]
     [126  28]
     [126  74]
     [137  18]
     [137  83]]


**STEP 4**: Choosing the number of clusters

WCSS  ->  Within Clusters Sum of Squares


```python
# finding wcss value for different number of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)
```


```python
# plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```


    
![png](Customer_Segmentation%28KMeans_Clustering%29_files/Customer_Segmentation%28KMeans_Clustering%29_14_0.png)
    


Optimum Number of Clusters = 5

**STEP 5**: Training the k-Means Clustering Model


```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)
```

    [3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3
     4 3 4 3 4 3 0 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 0 1 2 1 2 1 0 1 2 1 2 1 2 1 2 1 0 1 2 1 2 1
     2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
     1 2 1 2 1 2 1 2 1 2 1 2 1 2 1]


5 Clusters -  0, 1, 2, 3, 4

**STEP 6**: Visualizing all the Clusters


```python
# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
```


    
![png](Customer_Segmentation%28KMeans_Clustering%29_files/Customer_Segmentation%28KMeans_Clustering%29_20_0.png)
    



```python

```
