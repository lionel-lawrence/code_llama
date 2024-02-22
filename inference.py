import mysql.connector
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import *
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Connect to MySQL server
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_DATABASE
)

cursor = conn.cursor()

# Get a list of tables in the database
cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()


string = ''
# Loop through each table
for table in tables:
    table_name = table[0]
    string += f"CREATE TABLE {table_name} (\n" 
    # print(f"CREATE TABLE {table_name} (")

    # Get column information for the table
    cursor.execute(f"SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_name = '{table_name}';")
    columns = cursor.fetchall()
        
    # Loop through each column and print its definition
    for column in columns:
        column_name = column[0]
        data_type = column[1]
        character_maximum_length = column[2]

        # If the data type has a maximum length, include it in the definition
        if character_maximum_length:
            string += f"    {column_name} {data_type}({character_maximum_length}),\n" 

        else:
            string += f"    {column_name} {data_type},\n"

    string += ")\n\n"


# Close cursor and connection
cursor.close()
conn.close()

while True:
    question = input("Enter your query here (type 'quit'to exit): ") # input here

    start_time = time.time()
    # Check if the user wants to quit
    if question.lower() == 'quit':
        print("Exiting the program...")
        break  # Exit the loop

    # Process the user input (you can add your logic here)
    print("You entered:", question, "\n")

    string += '\n\n' + '-- ' + question + '\n\n' +'SELECT'


    input_ids = tokenizer(string, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    sql_query = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('\n')[-1]

    end_time = time.time()
    print(sql_query,'\n')
    print(f"Total time taken: {end_time - start_time} secs...\n")

    # Connect to MySQL server
    conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_DATABASE
    )

    cursor = conn.cursor()

    cursor.execute(sql_query)
    result = cursor.fetchall()

    print(result)