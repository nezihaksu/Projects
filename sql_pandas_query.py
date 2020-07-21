import pandas as pd 
import sqlite3

class forest_fire:

	def __init__(self,db):
		self.db = db
		#self.db0 = db0

	def to_sql(self):
		db = sqlite3.connect('forest_fire.db')
		db0 = sqlite3.connect('nature_of_work.db')

		#TABLES!
		#sql_table = self.db.to_sql(name = 'forestfire',con = db)
		#sql_table0 = self.db.to_sql(name = 'nature',con = db0)

		table0 = pd.read_sql_query('select*from nature',db0)
		table =  pd.read_sql_query('select*from forestfire',db)
		
		#count = pd.read_sql_query('select count(*) from work',db)
		#drop = pd.read_sql_query('drop table work',db)
		avg = pd.read_sql_query('select avg(temp) from forestfire',db)
		order_by = pd.read_sql_query('select month from forestfire order by temp asc;',db)
		like = pd.read_sql_query('select temp from forestfire where month like "m_rch"; ',db)
		inner_join = pd.read_sql_query('select month,day,month,day from forestfire inner join nature on nature.wind = forestfire.wind ',{db,db0})
		pd.set_option('display.max_column',None)

		return table,table0


csv_file = forest_fire(pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv',encoding = 'latin-1'))

print(csv_file.to_sql())