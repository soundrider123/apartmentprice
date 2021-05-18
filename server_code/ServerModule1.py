import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from io import BytesIO
import anvil.media
import joblib


from bs4 import BeautifulSoup
import requests

@anvil.server.callable
def get_to_client(csvname):
  row = app_tables.lookupdata.get(filename=csvname)
  return row['datadict']

@anvil.server.callable
def load_unegui_data(location1):
  df = loadcsv('unegui')

  df['location'] = df['duureg'] + ' ' + df['bairshil']
  df['price'] = df['price'].str.replace('\n', '')
  df2 = df.loc[df['location'] == location1, ['price','ognoo','talbai','tsonhnii_too','ashiglalt_on','heden_davhart']]
  df2 = df2.sort_values(['ognoo'], ascending=[False])
  return df2.to_dict(orient="records")

@anvil.server.callable
def get_info():
  if os.path.isfile('/unegui.csv'):
    os.remove('/unegui.csv')
  df1 = loadcsv('unegui')
  
  df_info  = pd.DataFrame(columns = ['description', 'value'])
  df_info = df_info.append({'description':'Total row number','value':df1.shape[0]},ignore_index=True)
  df_info = df_info.append({'description':'Min ognoo','value':df1['ognoo'].min()},ignore_index=True)
  df_info = df_info.append({'description':'Max ognoo','value':df1['ognoo'].max()},ignore_index=True)
  
  return df_info.to_dict(orient="records")

def loadcsv(csvname):
  file_exists = False
  if os.path.isfile(f'/{csvname}.csv'):
    file_exists = True

  if not file_exists: 
    row = app_tables.datafile.get(filename=csvname)
    with open(f'/{csvname}.csv', 'w', encoding='utf8') as f:
      bytes_io = BytesIO(row['filemedia'].get_bytes())
      byte_str = bytes_io.read()
      text_obj = byte_str.decode('UTF-8')
      f.write(text_obj)
  
  df = pd.read_csv(f'/{csvname}.csv', delimiter='|', encoding='utf-8')  
  return df

@anvil.server.background_task
def build_model_task(method):
  build_model_method(method)
  return
  
def build_model_method(method):  
  #anvil.server.task_state['n_complete'] = 1
  df = loadcsv('unegui')
  df.drop_duplicates(keep="last", inplace=True)
  
  df['price'] = df['price'].str.replace('\n', '')
  df['price'] = df['price'].str.replace('сая ₮', '')
  df['price'] = df['price'].str.replace(',', '.')
  df['talbai'] = df['talbai'].str.replace('м²', '')  
  
  df['price'] = df['price'].astype(str)
  
  df2 = df[pd.to_numeric(df['price'], errors='coerce').notnull()].copy()
  
  df2['location'] = df2['duureg'] + ' ' + df2['bairshil']
  
  df2['totalprice'] = np.where(df2['price'].astype('float32') < 10 , df2['price'].astype('float32') * df2['talbai'].astype('float32') , df2['price'].astype('float32')  )
  
  unuudur = datetime.today().strftime('%Y-%m-%d')
  uchigdur = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
  df2['ognoo'] = df2['ognoo'].str.replace('Нийтэлсэн: ', '')
  df2['ognoo'] = df2['ognoo'].str.replace('Өнөөдөр', unuudur)
  df2['ognoo'] = df2['ognoo'].str.replace('Өчигдөр', uchigdur)
  df2['ognoo'] = df2['ognoo'].str[0:10]
  
  df2['ashiglalt_on'] = df2['ashiglalt_on'].astype('int')
  df2['davhar'] = df2['davhar'].astype('int')
  df2['heden_davhart'] = df2['heden_davhart'].astype('int')
  df2['tsonhnii_too'] = df2['tsonhnii_too'].astype('int')
  df2['talbai'] = df2['talbai'].astype('float32')
  df2['talbai'] = df2['talbai'].astype('int')

  #anvil.server.task_state['n_complete'] = 2
  #Finding outliers by zscore value (greater than 3)
  arr_zscore = []  
    
  for loc_v in df2['location'].unique():
    loc = df2[df2['location'] == loc_v]
    loc2 = loc['totalprice'].to_frame()
    loc2['zscore'] = abs( (loc2.totalprice - loc2.totalprice.mean())/loc2.totalprice.std(ddof=0) )
    for i in loc2[loc2['zscore'] > 3].index:
      arr_zscore.append(i)  

  df2.drop(index=arr_zscore, inplace = True)

  #delete outlier for talbai
  df2.drop(index=df2[df2.talbai < 10].index, inplace = True)
  df2.drop(index=df2[df2.talbai > 400].index, inplace = True)

  df2.drop_duplicates(keep="last", inplace=True)
  
  def get_days(date2):
    dt1 = pd.to_datetime('2020-09-30', format='%Y-%m-%d')
    dt2 = pd.to_datetime(date2, format='%Y-%m-%d')
    return (dt2-dt1).days  
  
  df2['off_day'] = df2['ognoo'].apply(lambda x: get_days(x))
  
  df_obj = df2.select_dtypes(['object'])
  
  df2[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
  
  #anvil.server.task_state['n_complete'] = 3
  def lookup_to_csv(lookup_name):
    df_lookup = pd.DataFrame({ 'id': pd.unique(df2[lookup_name].values) })
    df_lookup = df_lookup.sort_values('id')
    df_lookup = df_lookup.reset_index()
    del df_lookup['index']    
    df_lookup['numval'] = df_lookup.index + 1
    if os.path.isfile(f'/{lookup_name}.csv'):
      os.remove(f'/{lookup_name}.csv')
    df_lookup.to_csv(f'/{lookup_name}.csv', index=None, sep='|', header=True, encoding="utf-8")
    
  lookup_to_csv('location')
  lookup_to_csv('garaj')
  lookup_to_csv('lising')
  lookup_to_csv('tagt')
  lookup_to_csv('haalga')
  lookup_to_csv('tsonh')
  lookup_to_csv('shal')
  lookup_to_csv('ashiglalt_on')
  lookup_to_csv('heden_davhart')
  lookup_to_csv('talbai')
  lookup_to_csv('davhar')
  lookup_to_csv('tsonhnii_too')  

  #anvil.server.task_state['n_complete'] = 4 

  def update_numval(lookup_name):
    df = pd.read_csv(f'/{lookup_name}.csv', delimiter='|')
    df['id'] = df['id'].astype(str)
    df3 = pd.merge(df2, df,  how='inner', left_on=[lookup_name], right_on = ['id'])
    df3 = df3.rename(columns={'numval': f'{lookup_name}_num'})
    return df3  
  
  df2 = update_numval('location')
  df2 = update_numval('garaj')
  df2 = update_numval('lising')
  df2 = update_numval('tagt')
  df2 = update_numval('haalga')
  df2 = update_numval('tsonh')
  df2 = update_numval('shal')
  
  df2 = df2.reset_index()
  X_prepared = df2[['ashiglalt_on', 'heden_davhart', 'talbai', 'davhar', 'tsonhnii_too', 'off_day', 'location_num', 'garaj_num', 'lising_num', 'tagt_num', 'haalga_num', 'tsonh_num', 'shal_num']]
  
  df2['totalprice'] = df2['totalprice'].astype('int')
  
  #anvil.server.task_state['n_complete'] = 5
  print('building model ...')
  X_train, X_test, y_train, y_test = train_test_split(X_prepared, df2[['totalprice']].values, test_size=0.1, random_state=42)
  
  regr = RandomForestRegressor(n_estimators=1000, random_state=1)
  print('fitting ...')
  regr.fit(X_train, y_train)  
  
  #anvil.server.task_state['n_complete'] = 6
  print('saving model ...')
  # save
  if os.path.isfile('/regr.joblib'):
    os.remove('/regr.joblib')
  joblib.dump(regr, "/regr.joblib", compress=3)

  y_pred = regr.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  
  print('finished saving.')
  
  print('saving to db ...')
  save_to_db()
  print('finished saving to db.')
  
  return rmse

@anvil.server.callable
def predict_price(ashiglalt_on, heden_davhart, talbai, davhar, tsonhnii_too, date_pred, location, garaj, lising, tagt, haalga, tsonh, shal):
  file_exists = False
  if os.path.isfile('/regr.joblib'):
    file_exists = True

  print('start predicting ...')
  if not file_exists:
    print('saving from db to file ...')
    row = app_tables.modelfile.get(filename='model')
    with open('/regr.joblib', 'wb') as f:
      bytes_io = BytesIO(row['filemedia'].get_bytes())
      byte_arr = bytes_io.read()
      f.write(byte_arr)
  
 
  print('loading model ...')
  regr = joblib.load("/regr.joblib")

  col_names =  ['ashiglalt_on', 'heden_davhart', 'talbai', 'davhar', 'tsonhnii_too', 'off_day', 'location', 'garaj', 'lising', 'tagt', 'haalga', 'tsonh', 'shal']
  df2  = pd.DataFrame(columns = col_names)

  def get_days(date2):
    dt1 = pd.to_datetime('2020-09-30', format='%Y-%m-%d')
    dt2 = pd.to_datetime(date2, format='%Y-%m-%d')
    return (dt2-dt1).days  
  
  off_day = get_days(date_pred)
  df2.loc[0] = [ashiglalt_on, heden_davhart, talbai, davhar, tsonhnii_too, off_day, location, garaj, lising, tagt, haalga, tsonh, shal]
  
  print(df2)
  
  df2['ashiglalt_on'] = df2['ashiglalt_on'].astype('int')
  df2['davhar'] = df2['davhar'].astype('int')
  df2['heden_davhart'] = df2['heden_davhart'].astype('int')
  df2['tsonhnii_too'] = df2['tsonhnii_too'].astype('int')
  df2['talbai'] = df2['talbai'].astype('float32')
  df2['talbai'] = df2['talbai'].astype('int')
  
  print('transforming value to numeric ...')
  def update_numval(lookup_name):
    df = pd.read_csv(f'/{lookup_name}.csv', delimiter='|')
    df['id'] = df['id'].astype(str)
    df3 = pd.merge(df2, df,  how='inner', left_on=[lookup_name], right_on = ['id'])
    df3 = df3.rename(columns={'numval': f'{lookup_name}_num'})
    return df3    
  
  df2 = update_numval('location')
  df2 = update_numval('garaj')
  df2 = update_numval('lising')
  df2 = update_numval('tagt')
  df2 = update_numval('haalga')
  df2 = update_numval('tsonh')
  df2 = update_numval('shal')
 
  X_pred = df2[['ashiglalt_on', 'heden_davhart', 'talbai', 'davhar', 'tsonhnii_too', 'off_day', 'location_num', 'garaj_num', 'lising_num', 'tagt_num', 'haalga_num', 'tsonh_num', 'shal_num']]
  
  print('actual predicting started ...')
  y_pred = regr.predict(X_pred)

  return y_pred[0]
  

def csv_to_db(fname):
  with open(f'/{fname}.csv', mode='r', encoding='utf8') as file: 
    fileContent = file.read()
    file_contents = fileContent.encode()
    my_media = anvil.BlobMedia(content_type="text/plain", content=file_contents, name=f'{fname}.csv')
    row = app_tables.datafile.get(filename=fname)
    row.delete()
    app_tables.datafile.add_row(filename=fname, filemedia=my_media)

    row = app_tables.lookupdata.get(filename=fname)
    if row is not None:
      row.delete()
      
    df = pd.read_csv(f'/{fname}.csv', delimiter='|', encoding='utf-8')  
    datadict1 = df.to_dict(orient="records")          
    app_tables.lookupdata.add_row(filename=fname, datadict=datadict1)
    
  return

@anvil.server.callable
def save_to_db():
  with open("/regr.joblib", mode='rb') as file: 
    fileContent = file.read()
    my_media = anvil.BlobMedia(content_type="application/octet-stream", content=fileContent, name="regr.joblib")
    row = app_tables.modelfile.get(filename="model")
    row.delete()
    app_tables.modelfile.add_row(filename='model', filemedia=my_media)
  
  csv_to_db('location')
  csv_to_db('garaj')
  csv_to_db('lising')
  csv_to_db('tagt')
  csv_to_db('haalga')
  csv_to_db('tsonh')
  csv_to_db('shal')
  csv_to_db('ashiglalt_on')
  csv_to_db('heden_davhart')
  csv_to_db('talbai')
  csv_to_db('davhar')
  csv_to_db('tsonhnii_too')  
    
  return

@anvil.server.callable
def build_model(method):
  task = anvil.server.launch_background_task('build_model_task', method)
  return task

@anvil.server.callable
def kill_task(task):
  task.kill()

@anvil.server.background_task
def scheduled_task():
  webscrape()
  build_model_method('randomforest')
  return

@anvil.server.callable
def scrape_data():
  task = anvil.server.launch_background_task('scrape_data_task')
  return task


@anvil.server.background_task
def scrape_data_task():
  webscrape()

@anvil.server.callable
def scrape_data():
  task = anvil.server.launch_background_task('scrape_data_task')
  return task

def webscrape():
  if os.path.isfile('/unegui.csv'):
    os.remove('/unegui.csv')
  
  df_unegui = loadcsv('unegui')
  
  col_names =  ['page_url', 'price','ognoo','shal','tagt','ashiglalt_on','garaj','tsonh','davhar','haalga','talbai','heden_davhart','lising','duureg','tsonhnii_too','bairshil']
  df  = pd.DataFrame(columns = col_names)
  
  def insert(df, row):
    insert_loc = df.index.max()  
    if pd.isna(insert_loc):
      df.loc[0] = row
    else:
      df.loc[insert_loc + 1] = row

  def get_span(src):
    source = '"""'+str(src)+'"""'
    soup = BeautifulSoup(source, "html.parser")
    return soup.text.replace('\n','').replace('"','')      

  for i in range(1, 10):
    print(f'page: {str(i)}')
    if i == 1:
      page = requests.get(f'https://www.unegui.mn/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/').content.decode("utf-8")
    else:
      page = requests.get(f'https://www.unegui.mn/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/?page={str(i)}').content.decode("utf-8")
    soup = BeautifulSoup(page, "html.parser")  
    list_ad = soup.find_all("div", attrs={"class": "list-announcement-block"})  

    for ad in list_ad:
      source_code = '"""'+str(ad)+'"""'
      soup_sm = BeautifulSoup(source_code, "html.parser")
      url = soup_sm.find('a', href=True) 
      
      #set_trace()
      page_url = 'https://www.unegui.mn'+str(url['href'])
      ###############################################################
      # if url is already downloaded then skip
      row = df_unegui.loc[df_unegui['page_url'] == page_url]
      if row.shape[0] > 0:
        continue
      ###############################################################
      #print(page_url)
      page = requests.get(page_url).content.decode("utf-8")
      soup_job = BeautifulSoup(page, "html.parser")
      
      dtl = soup_job.find("div", attrs={"class": "announcement-characteristics clearfix"})
      source_dtl = '"""'+str(dtl)+'"""'
      soup_dtl = BeautifulSoup(source_dtl, "html.parser")        
      list_li = soup_dtl.find_all("li")
      
      price=soup_job.find("div", attrs={"class": "announcement-price__cost"}).text
      ognoo=soup_job.find("span", attrs={"class": "date-meta"}).text
      shal=''
      tagt=''
      ashiglalt_on=''
      garaj=''
      tsonh=''
      davhar=''
      haalga=''
      talbai=''
      heden_davhart=''
      lising=''
      duureg=''
      tsonhnii_too=''
      bairshil=''    
  
      for li in list_li:
        source_span = '"""'+str(li)+'"""'
        soup_span = BeautifulSoup(source_span, "html.parser")             
        span = soup_span.find("span", attrs={"class": "key-chars"})
        
        if "Шал" == span.text:
          span_value = str(li).replace("Шал", "")
          shal = get_span(span_value)
            
        if "Тагт" == span.text:
          span_value = str(li).replace("Тагт", "")
          tagt = get_span(span_value)
            
        if "Ашиглалтанд орсон он" == span.text:
          span_value = str(li).replace("Ашиглалтанд орсон он", "")
          ashiglalt_on = get_span(span_value)
            
        if "Гараж" == span.text:
          span_value = str(li).replace("Гараж", "")
          garaj = get_span(span_value)
            
        if "Цонх" == span.text:
          span_value = str(li).replace("Цонх", "")
          tsonh = get_span(span_value)
            
        if "Барилгын давхар" == span.text:
          span_value = str(li).replace("Барилгын давхар", "")
          davhar = get_span(span_value)
            
        if "Хаалга" == span.text:
          span_value = str(li).replace("Хаалга", "")
          haalga = get_span(span_value)
            
        if "Талбай" == span.text:
          span_value = str(li).replace("Талбай", "")
          talbai = get_span(span_value)
            
        if "Хэдэн давхарт" == span.text:
          span_value = str(li).replace("Хэдэн давхарт", "")
          heden_davhart = get_span(span_value)
                            
        if "Лизинг" == span.text:
          span_value = str(li)
          lising = get_span(span_value)
          lising = lising.replace("ЛизингЛизинг", "Лизинг")
            
        if "Дүүрэг" == span.text:
          span_value = str(li).replace("Дүүрэг", "")
          duureg = get_span(span_value)
            
        if "Цонхны тоо" == span.text:
          span_value = str(li).replace("Цонхны тоо", "")
          tsonhnii_too = get_span(span_value)
            
        if "Байршил" == span.text:
          span_value = str(li).replace("Байршил", "")
          bairshil = get_span(span_value)

          
      insert(df,[page_url, price,ognoo,shal,tagt,ashiglalt_on,garaj,tsonh,davhar,haalga,talbai,heden_davhart,lising,duureg,tsonhnii_too,bairshil])

      
  
  #changing for further use
  print(f'downloaded url count: {df.shape[0]}')

  unuudur = datetime.today().strftime('%Y-%m-%d')
  uchigdur = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
  df['ognoo'] = df['ognoo'].str.replace('Нийтэлсэн: ', '')
  df['ognoo'] = df['ognoo'].str.replace('Өнөөдөр', unuudur)
  df['ognoo'] = df['ognoo'].str.replace('Өчигдөр', uchigdur)
  df['ognoo'] = df['ognoo'].str[0:10]  
  
  df['price'] = df['price'].str.replace('\n', '')

  df3=pd.concat([df, df_unegui],ignore_index=True)
  
  #saving
  if os.path.isfile('/unegui.csv'):
    os.remove('/unegui.csv')
  df3.to_csv("/unegui.csv", index=None, sep='|', header=True)
  csv_to_db('unegui')
  return

