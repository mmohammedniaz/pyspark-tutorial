### How Can One Use DataFrames?

Once built, DataFrames provide a domain-specific language for distributed data manipulation.  Here is an example of using DataFrames to manipulate the demographic data of a large population of users:

    # Create a new DataFrame that contains “young users” only
    young = users.filter(users.age < 21)

    # Alternatively, using Pandas-like syntax
    young = users[users.age < 21]

    # Increment everybody’s age by 1
    young.select(young.name, young.age + 1)

    # Count the number of young users by gender
    young.groupBy(“gender”).count()

    # Join young users with another DataFrame called logs
    young.join(logs, logs.userId == users.userId, “left_outer”)

