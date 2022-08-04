The code demonstrates how sensitive data inspection and customer - dataset access validation work. 

In the real-case scenario the flow when a new dataset and a new customer arrives will be like the below;

- When a new dataset arrives, it is inserted into the dataset_table and a unique id is assigned to the dataset
- The columns, containing sensitive data, of the new dataset are detected and stored in the sensitive_columns table  
- When a new customer arrives, the customer is inserted into the customer table 
- Another table (customer_dataset) keeps records of which customer has access right to which datasets
- Customers will receive data in their algorithm via our method called retrieve_data. The methods checks if the customer has access right to the required dataset, then fetches the dataset without the sensitive columns
