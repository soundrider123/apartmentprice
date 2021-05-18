from ._anvil_designer import Main_FormTemplate
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import plotly.graph_objects as go
import anvil.server
from SalePrediction.Form1 import Form1
from SalePrediction.Form3 import Form3
from SalePrediction.Form5 import Form5

class Main_Form(Main_FormTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.
    
    form1_instance = Form1(param='an_argument')
    self.flow_panel_1.clear()
    self.flow_panel_1.add_component(form1_instance)
    self.button_1.tag = form1_instance
    
    form3_instance = Form3(param='an_argument')
    self.button_3.tag = form3_instance

    form5_instance = Form5(param='an_argument')
    self.button_5.tag = form5_instance
    

  def button_1_click(self, **event_args):
    """This method is called when the button is clicked"""
    form1 = self.button_1.tag
    if form1 is None:
      return
    self.flow_panel_1.clear()
    self.flow_panel_1.add_component(form1)    
    pass

  def button_2_click(self, **event_args):
    """This method is called when the button is clicked"""
    form2 = self.button_2.tag
    if form2 is None:
      return    
    self.flow_panel_1.clear()
    self.flow_panel_1.add_component(form2)    
    pass

  def button_3_click(self, **event_args):
    """This method is called when the button is clicked"""
    form3 = self.button_3.tag
    if form3 is None:
      return
    self.flow_panel_1.clear()
    self.flow_panel_1.add_component(form3)    
    pass

 
  def button_5_click(self, **event_args):
    """This method is called when the button is clicked"""
    form5 = self.button_5.tag
    if form5 is None:
      return    
    self.flow_panel_1.clear()
    self.flow_panel_1.add_component(form5)    
    pass



