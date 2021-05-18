from ._anvil_designer import Form1Template
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server
import datetime

class Form1(Form1Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    
    self.date_pred.date = datetime.date.today()

    # Any code you write here will run when the form opens.
    dic = anvil.server.call('get_to_client',"location")
    self.drop_down_location.items = [ (r['id'], r['id']) for r in dic]
    
    dic = anvil.server.call('get_to_client',"tsonhnii_too")
    self.drop_down_tsonhnii_too.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"tsonh")
    self.drop_down_tsonh.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"tagt")
    self.drop_down_tagt.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"shal")
    self.drop_down_shal.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"lising")
    self.drop_down_lising.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"heden_davhart")
    self.drop_down_heden_davhart.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"talbai")
    self.drop_down_talbai.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"haalga")
    self.drop_down_haalga.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"garaj")
    self.drop_down_garaj.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"davhar")
    self.drop_down_davhar.items = [ (str(r['id']), r['id']) for r in dic]    

    dic = anvil.server.call('get_to_client',"ashiglalt_on")
    self.drop_down_ashiglalt_on.items = [ (str(r['id']), r['id']) for r in dic]    
    
  def predict_click(self, **event_args):
    """This method is called when the button is clicked"""
    ashiglalt_on = self.drop_down_ashiglalt_on.selected_value
    heden_davhart = self.drop_down_heden_davhart.selected_value
    talbai = self.drop_down_talbai.selected_value
    davhar = self.drop_down_davhar.selected_value
    tsonhnii_too = self.drop_down_tsonhnii_too.selected_value
    date_pred1 = self.date_pred.date
    location = self.drop_down_location.selected_value
    garaj = self.drop_down_garaj.selected_value
    lising = self.drop_down_lising.selected_value
    tagt = self.drop_down_tagt.selected_value
    haalga = self.drop_down_haalga.selected_value
    tsonh = self.drop_down_tsonh.selected_value
    shal = self.drop_down_shal.selected_value

    date_v = date_pred1.strftime('%Y-%m-%d')
    price_pred = anvil.server.call('predict_price',ashiglalt_on, heden_davhart, talbai, davhar, tsonhnii_too, date_v, location, garaj, lising, tagt, haalga, tsonh, shal)    
    self.text_box_result.text = str(price_pred)
    
    pass




