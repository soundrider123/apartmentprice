from ._anvil_designer import Form3Template
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
import anvil.server
from SalePrediction import Globals

class Form3(Form3Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    self.task = None
    # Any code you write here will run when the form opens.
    dic = anvil.server.call('get_info')
    self.repeating_panel_1.items = dic   
    self.data_grid_1.width = "600px"

  def gridrefresh(self):
    dic = anvil.server.call('get_info')
    self.repeating_panel_1.items = dic   
    pass       

  def build_model_click(self, **event_args):
    """This method is called when the button is clicked"""
    #Globals.is_calced = False
    
    method = "randomforest"
    self.task = anvil.server.call('build_model',method)   
    pass

  def stop_click(self, **event_args):
    """This method is called when the button is clicked"""
    if self.task == None:
      return
    else:
      anvil.server.call('kill_task', self.task)
    pass

  def webscrape_click(self, **event_args):
    """This method is called when the button is clicked"""
    self.task = anvil.server.call('scrape_data')
    pass

  def button_refresh_click(self, **event_args):
    """This method is called when the button is clicked"""
    dic = anvil.server.call('get_info')
    self.repeating_panel_1.items = dic      
    pass






