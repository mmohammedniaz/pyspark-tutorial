**Goal - Make Practical Machine Learning Scalable & Easy**

* ML Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering
* Featurization: feature extraction, transformation, dimensionality reduction, and selection
* Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
* Persistence: saving and load algorithms, models, and Pipelines
* Utilities: linear algebra, statistics, data handling, etc.

### ML Pipelines
MLlib standardizes APIs for machine learning algorithms to make it easier to combine multiple algorithms into a single pipeline.
* DataFrame - Storage mechanism
* Transformer - An algorithm which converts one DataFrame to another DF. Eg. model is a transformer
* Estimator - Creates transformer using fit on a DataFrame. learning algo is a estimator which trans on a DF & produces a model
* Pipeline - Chains multiple Transformer & Estimators to create a flow.
* Parameters - Input to Transformers & Estimators

#### Pipeline components
##### TRansformers
* Transformer implements a method transform(), which converts one DataFrame into another, generally by appending one or more columns. 
* Feature transformer might take a dataframe & converts into another dataframe with new columns using available columns
* New dataframes with new cols only & no previous cols.

#### Estimator
* an Estimator implements a method fit(), which accepts a DataFrame and produces a Model, which is a Transformer
* Eg - LogisticRegression is an estimator, and calling fit on it creates a LogisticRegressionModel, which is a Mode & hence a transformer.
*

[ML](http://spark.apache.org/docs/latest/img/ml-Pipeline.png)


