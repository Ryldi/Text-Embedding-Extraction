import pandas
import os
import mysql.connector
from text_embedding import text_embed_string as embed

class PDF_autosave:
    def auto_save(data):
        # database connection
        _db_context = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="db_test"
        )

        cursor = _db_context.cursor()

        temp_name = str(data["file_name"][0])
        temp_content = str(data["file_content"][0])
        temp_url = 'pdf/' + str(data["file_name"][0]) + '.pdf'
        temp_content_vector = embed([str(data["file_content"][0])])
        
        # insert into database
        cursor.execute("INSERT INTO ms_file (file_name, file_content, file_url, file_content_vector) VALUES (%s, %s, %s, %s)", (temp_name, temp_content, temp_url, temp_content_vector))
        _db_context.commit() 

        # success message
        print("Insertion successful")

        # close db connection
        cursor.close()
        _db_context.close()    