#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright Airbus Group SAS 2015
# All rigths reserved.
#
# File Name : res.py
# Authors : Martin Matignon
#
################################################################################


########################################
# Module(s) declaration
########################################

import rospy
import os
from roslib.packages import get_pkg_dir
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding import loadUi

########################################
# Constante(s) and Variable(s) declaration
########################################

DIR_LWRDASHBOARD_RESOURCES = os.path.join(get_pkg_dir('lwr_dashboard'),'resources')
DIR_LWRDASHBOARD_IMAGES = DIR_LWRDASHBOARD_RESOURCES+'/images'
DIR_LWRDASHBOARD_LAYOUTS = DIR_LWRDASHBOARD_RESOURCES+'/layouts'
DIR_LWRDASHBOARD_VALUES = DIR_LWRDASHBOARD_RESOURCES+'/values'

########################################
# Class(ies) declaration
########################################

class LwrDashboardImages():
    def __init__(self):
        self.uuid = self.__class__.__name__
        self.dir = DIR_LWRDASHBOARD_IMAGES
        self.icon_robot = DIR_LWRDASHBOARD_IMAGES+'/icon_robot.png'
        self.icon_robot_running = DIR_LWRDASHBOARD_IMAGES+'/icon_robot_running.png'
        self.icon_robot_unknow = DIR_LWRDASHBOARD_IMAGES+'/icon_robot_unknow.png'
        self.icon_robot_in_default = DIR_LWRDASHBOARD_IMAGES+'/icon_robot_in_default.png'
        self.icon_robot_e_stopped = DIR_LWRDASHBOARD_IMAGES+'/icon_robot_e_stopped.png'
        self.icon_robot_collision = DIR_LWRDASHBOARD_IMAGES+'/icon_robot_collision.png'
        self.icon_robot_error = DIR_LWRDASHBOARD_IMAGES+'/icon_robot_error.png'
    def findById(self, id=""):
        try:
            return getattr(self,id)
        except:
            return None

class LwrDashboardLayouts():
    def __init__(self):
        self.uuid = self.__class__.__name__
        self.dir = DIR_LWRDASHBOARD_LAYOUTS
        self.robot = DIR_LWRDASHBOARD_LAYOUTS+'/robot.ui'
        self.popup = DIR_LWRDASHBOARD_LAYOUTS+'/popup.ui'
    def findById(self, id=""):
        try:
            return getattr(self,id)
        except:
            return None

class LwrDashboardValues():
    def __init__(self):
        class LwrDashboardStrings():
            def __init__(self):
                self.uuid = self.__class__.__name__
            def findById(self, id=""):
                try:
                    return getattr(self,id)
                except:
                    return None
            def title(self, lng="en"):
                if lng == "en":
                    return "UR10 Dashboard".decode('utf-8')
                elif lng == "fr":
                    return "UR10 Dashboard".decode('utf-8')
                else:
                    return "UR10 Dashboard".decode('utf-8')
                
            def robot_in_error(self, lng="en"):
                if lng == "en":
                    return "The robot is in error !".decode('utf-8')
                elif lng == "fr":
                    return "Le robot est en erreur !".decode('utf-8')
                else:
                    return "The robot is in error !".decode('utf-8')
                
            def acquit_timeout_exceeded(self, lng="en"):
                if lng == "en":
                    return "Timeout from acquit default exceeded !".decode('utf-8')
                elif lng == "fr":
                    return "Délai d'acquittement dépassé !".decode('utf-8')
                else:
                    return "Timeout from acquit default exceeded !".decode('utf-8')
                
            def unknown(self, lng="en"):
                if lng == "en":
                    return "Unknown".decode('utf-8')
                elif lng == "fr":
                    return "Inconnu".decode('utf-8')
                else:
                    return "Unknown".decode('utf-8')
                
        class LwrDashboardStyles():
            def __init__(self):
                self.uuid = self.__class__.__name__
                self.default_dialog = "QDialog{background-color:#ffffff;border: 5px solid red;}QLabel{background-color: transparent;border: none};"
                self.acquit_button = "background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #dc7575, stop: 1 #8f0000);color:#ffffff;font: 63 16pt;"
            def findById(self, id=""):
                try:
                    return getattr(self,id)
                except:
                    return None
        self.uuid = self.__class__.__name__
        self.dir = DIR_LWRDASHBOARD_VALUES
        self.strings = LwrDashboardStrings()
        self.styles = LwrDashboardStyles()
    def findById(self, id=""):
        try:
            return getattr(self,id)
        except:
            return None

class R:
    DIR = DIR_LWRDASHBOARD_RESOURCES
    images = LwrDashboardImages()
    layouts = LwrDashboardLayouts()
    values = LwrDashboardValues()
    @staticmethod
    def getPixmapById(id=""):
        return QPixmap(R.images.findById(id))
    @staticmethod
    def getIconById(id=""):
        return QIcon(R.images.findById(id))


# End of file