import pandas as pd 
import sqlite3

class forest_fire:

	def __init__(self,db):
		self.db = db

	def to_sql(self):
		db = sqlite3.connect('forest_fire.db')
		#sql_table = self.db.to_sql(name = 'forestfire',con = db)
		table =  pd.read_sql_query('select*from forestfire ',db)
		#count = pd.read_sql_query('select count(*) from work',db)
		#drop = pd.read_sql_query('drop table work',db)
		avg = pd.read_sql_query('select avg(temp) from forestfire',db)
		order_by = pd.read_sql_query('select month from forestfire order by temp asc;',db)
		pd.set_option('display.max_column',None)

		

		return table


csv_file = forest_fire(pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv',encoding = 'latin-1'))

print(csv_file.to_sql())