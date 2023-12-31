K-Means For Pair Selection In Python Part III
Part III: Building a StatArb Strategy Using K-Means
Python Code

---------------------------------------------------------------------------------

#Importing Our Stock Data From Excel 
file=pd.ExcelFile('KMeansStocks.xlsx')

#Parsing the Sheet from Our Excel 
file stockData=file.parse('Example')

#Looking at the head of our Stock Data
stockData.head()

#Looking at the tail of our Stock Data
stockData.tail()

#Making a copy of our stockdata
stockDataCopy=stockData.copy()

#Dropping the Name column from our stockData
stockDataCopy.drop('Name', inplace=True,axis=1)

#Checking the head of our stockData
stockDataCopy.head()

stockDataCopy.reindex(index=stockDataCopy['Symbol'],columns=stockDataCopy.columns)

#Adding back the values to our Columns
stockDataCopy['Symbol']=stockData['Symbol'].values
stockDataCopy['Dividend Yield']=stockData['Dividend Yield'].values
stockDataCopy['P/E']=stockData['P/E'].values
stockDataCopy['EPS']=stockData['EPS'].values
stockDataCopy['MarketCap']=stockData['MarketCap'].values
stockDataCopy['EBITDA']=stockData['EBITDA'].values

#Viewing the head of our stockDataCopy dataframe
stockDataCopy.head()

stock_kmeans=KMeans()

from scipy.spatial.distance import cdist

#creating an object to determine the value for K

class Get_K(object):
  def __init__(self,start,stop,X):
      self.start=start
      self.stop=stop
      self.X=X
      #in our example, we found out that there were some NaN
      #values in our data, thus we must fill those with 0
      #before passing our features into our model
      self.X=self.x.fillna(e)

  def get_k(self):
      #this method will iterate through different
      #values of K and create the SSE
      #initializing a list to hold our error terms
      self.errors=[ ]
      #intializing a range of values for K
      Range=range(self.start,self.stop)
      #iterating over range of values far K
      #and calculating our errors
      for i in Range:
      self.k_means=KMeans(n_clusters=i)
      self.k_means.fit(self.X)
      self.errors.append(sum(np.min(cdist(self.X[0:200],self.k_means.cluster_centers_,'euclidean'),axis=1))/200)
      return

 def plot_elbow(self):
      with plt.style.context(['seaborn-notebook','ggplot‘]):
      plt.figure(figsize=(10,8))
      #we have multiple features, thus we will use the
      #P/E to create our elbow
      plt.plot(self.X['P/E'][0:200],self.errors[0:200])
      plt.xlabel('Clusters')
      plt.ylabel('Errors')
      plt.title('K-Means Elbow Plot')
      plt.tight_layout()
      plt.show()
      return

features=stockDataCopy[[‘Dividend Yield','P/E','EPS','MarketCap','EBITDA']]

#Creating an instance of our Get_K object

#we are setting our range of K from 1 to 266
#note we pass in the first 200 features values in this example
#this was done because otherwise, to plot our elbow, we would
#have to set our range max at 500. To avoid the computational
#time associated with the for loop inside our method
#we pass in a slice of the first 200 features

#this is also the reason we divide by 200 in our class

Find_K=Get_K(1, 200,features [1:200]

#Calling get_k method on our Find_K object
Find_K.get_k()

#Visualizing our K-Means Elbow Plot
Find_K.plot_elbow()


---------------------------------------------------------------------------------

