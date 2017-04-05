### Goals of Big Data Processing Architecture
* A good real-time data processing architecture needs to be fault-tolerant and scalable.
* It needs to support batch and incremental updates, and must be extensible.

### Solutions 
* Nathan Marz, creator of Apache Storm, describing what we have come to know as the Lambda architecture. The Lambda architecture has proven to be relevant to many use-cases and is indeed used by a lot of companies, for example Yahoo and Netflix. But of course, Lambda is not a silver bullet and has received some fair criticism on the coding overhead it can create.
* Jay Kreps from LinkedIn posted an article describing what he called the Kappa architecture, which addresses some of the pitfalls associated with Lambda. Kappa is not a replacement for Lambda, though, as some use-cases deployed using the Lambda architecture cannot be migrated.

## Lambda Architecture

![Lambda](https://www.ericsson.com/research-blog/wp-content/uploads/2015/11/LambdaKappa1_1.png)

### Layers
* Batch
  1. managing historical data
  2. recomputing results such as machine learning models. S
  3. Specifically, the batch layer receives arriving data, combines it with historical data and recomputes results by iterating over the entire combined data set. 
  4. The batch layer operates on the full data and thus allows the system to produce the most accurate results. 
  5. However, the results come at the cost of high latency due to high computation time.
* Speed
  1. The speed layer is used in order to provide results in a low-latency, near real-time fashion. 
  2. The speed layer receives the arriving data and performs incremental updates to the batch layer results. Thanks to the incremental algorithms implemented at the speed layer, computation cost is significantly reduced. 
* Serving
  1. The serving layer enables various queries of the results sent from the batch and speed layers.
***

## Kappa Architecture

![Kappa](https://www.ericsson.com/research-blog/wp-content/uploads/2015/11/LambdaKappa1_2.png)
