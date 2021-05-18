from ._anvil_designer import Form5Template
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
import anvil.server
from SalePrediction import Globals

class Form5(Form5Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.
    dic = anvil.server.call('get_to_client',"location")
    self.drop_down_location.items = [ (r['id'], r['id']) for r in dic]
    
    self.drop_down_location.selected_value = 'Баянзүрх 5-р хороолол'
    dic2 = anvil.server.call('load_unegui_data','Баянзүрх 5-р хороолол')
    self.repeating_panel_1.items = dic2  


  def form_show(self, **event_args):
    """This method is called when the column panel is shown on the screen"""
    pass

  def button_filter_click(self, **event_args):
    """This method is called when the button is clicked"""
    dic2 = anvil.server.call('load_unegui_data', self.drop_down_location.selected_value)
    self.repeating_panel_1.items = dic2      
    pass






