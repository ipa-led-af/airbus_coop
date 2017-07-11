#!/usr/bin/env python
################################################################################
#
# Copyright Airbus Group SAS 2015
# All rigths reserved.
#
# File Name : params.py
# Authors : Martin Matignon
#
# If you find any bug or if you have any question please contact
# Adolfo Suarez Roos <adolfo.suarez@airbus.com>
# Martin Matignon <martin.matignon.external@airbus.com>
#
#
################################################################################
from agi_docgen.common import html
from agi_docgen.common.html import HtmlElement
from agi_docgen.digraph.digraph import *
from agi_docgen.digraph.model.topic import getStandardTDModel


class NodeParameters(HtmlElement):
    
    def __init__(self):
        HtmlElement.__init__(self,
                             tag=html.Sections.article,
                             attrib={"class":"parameters"})
        
    def read(self, node_name, node_xml):
        
        has_params = False
        node_params = node_xml.find("parameters")
        
        if node_params.find("param") is not None:
        
            table = TABLE()
            
            title = TD()
            title.setAttrib(TD.ALIGN, ALIGN.Center)
            title.setAttrib(TD.COLSPAN, 5)
            title.setText("Launch parameters")
            table.addTR(TR(title))
            
            ptr = TR()
            ptr.addTD(getStandardTDModel("Name",        bgcolor=RgbColor.White, align=ALIGN.Center))
            ptr.addTD(getStandardTDModel("Type",        bgcolor=RgbColor.White, align=ALIGN.Center))
            ptr.addTD(getStandardTDModel("Default",     bgcolor=RgbColor.White, align=ALIGN.Center))
            ptr.addTD(getStandardTDModel("Unit",        bgcolor=RgbColor.White, align=ALIGN.Center))
            ptr.addTD(getStandardTDModel("Description", bgcolor=RgbColor.White, align=ALIGN.Center))
            table.addTR(ptr)
            
            for param in node_params.iter('param'):
                
                ptr = TR()
                
                ptr.addTD(getStandardTDModel(self.get_pattrib(param, 'name'),
                                             bgcolor=RgbColor.White,
                                             align = ALIGN.Left))
                
                ptr.addTD(getStandardTDModel(self.get_pattrib(param, "type"),
                                             bgcolor=RgbColor.White,
                                             align = ALIGN.Center))
                
                ptr.addTD(getStandardTDModel(self.get_pattrib(param, "default"),
                                             bgcolor=RgbColor.White,
                                             align = ALIGN.Center))
                
                ptr.addTD(getStandardTDModel(self.get_pattrib(param, "unit"),
                                             bgcolor=RgbColor.White,
                                             align = ALIGN.Center))
                
                ptr.addTD(getStandardTDModel(str(param.text),
                                             bgcolor=RgbColor.White,
                                             align=ALIGN.Left))
                
                table.addTR(ptr)
                has_params = True
            
            if has_params is True:
                self.append(table)
        
        return has_params
    
    def get_pattrib(self, param, key):
        
        if key in param.attrib:
            return param.attrib[key]
        else:
            return "None"
        
    